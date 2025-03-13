from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from time import time
import datetime

import math
import random

from models import SyncNet_color_384 as SyncNet
import audio
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

import torch.multiprocessing as mp
import torch.distributed as dist
from pytorch_lightning.loggers import CSVLogger


parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=False,default="checkpoints/syncnet/",type=str)
parser.add_argument('--exp_num', help='ID number of the experiment', required=False, default="actor", type=str)
parser.add_argument('--history_train', help='Save history training', required=False,default="logs/syncnet/",type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
best_loss = 1000
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16
format_video = 'mov'
hparams.set_hparam("img_size", 384)

# mel augmentation
def mask_mel(crop_mel):
    block_size = 0.1
    time_size = math.ceil(block_size * crop_mel.shape[0])
    freq_size = math.ceil(block_size * crop_mel.shape[1])
    time_lim = crop_mel.shape[0] - time_size
    freq_lim = crop_mel.shape[1] - freq_size

    time_st = random.randint(0, time_lim)
    freq_st = random.randint(0, freq_lim)

    mel = crop_mel.copy()
    mel[time_st:time_st+time_size] = -4.
    mel[:, freq_st:freq_st + freq_size] = -4.

    return mel
def get_audio_length(audio_path):
    """Get the length of the audio file in seconds"""
    cmd = 'ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'.format(audio_path)
    audio_length = float(os.popen(cmd).read().strip())
    return audio_length


class Dataset(object):
    def __init__(self, file_list):
        self.all_videos = get_image_list(file_list)
        random.shuffle(self.all_videos)
        
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)
        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, f'{frame_id}.jpg')
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames


    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]
    
    def __len__(self):
        return len(self.all_videos)
    def __getitem__(self, idx):
        # random vid idx
        # vid_idx =  random.randint(0, len(self.all_videos) - 1)
        vidname = self.all_videos[idx % len(self.all_videos)]
        # Check if the video is valid
        img_names = sorted(list(glob(join(vidname, '*.jpg'))))
    
        if len(img_names) <= 3 * syncnet_T:
            # Skip short videos
            return self.__getitem__((idx + 1) % len(self.all_videos))
    
        # Deterministic positive/negative sampling based on index
        is_positive = idx % 2 == 0
    
        if is_positive:  # Positive pair (synchronized)
            # Choose a frame that has enough subsequent frames
            max_start_idx = len(img_names) - syncnet_T
            start_idx = idx % max_start_idx
            img_name = img_names[start_idx]
            chosen = img_name
            y = torch.ones(1).float()
        else:  # Negative pair (not synchronized)
            # Select base frame
            start_idx = idx % (len(img_names) - syncnet_T)
            img_name = img_names[start_idx]
    
            # Create negative example by selecting temporally distant frame
            if len(img_names) > syncnet_T * 3:
                # Use frame from same video but distant enough
                shift = syncnet_T * 2 + (idx % (len(img_names) - syncnet_T * 3))
                wrong_idx = (start_idx + shift) % (len(img_names) - syncnet_T)
                chosen = img_names[wrong_idx]
            else:
                # Use frame from different video if current one is too short
                diff_vid_idx = (idx // 2) % len(self.all_videos)
                if diff_vid_idx == idx % len(self.all_videos):
                    diff_vid_idx = (diff_vid_idx + 1) % len(self.all_videos)
                different_vid = self.all_videos[diff_vid_idx]
                diff_img_names = sorted(list(glob(join(different_vid, '*.jpg'))))
                if len(diff_img_names) > syncnet_T:
                    chosen = diff_img_names[idx % len(diff_img_names)]
                else:
                    # Fallback to using the same video but different frame
                    wrong_idx = (start_idx + syncnet_T) % (len(img_names) - syncnet_T)
                    chosen = img_names[wrong_idx]
    
            y = torch.zeros(1).float()
    
        window_fnames = self.get_window(chosen)
        if window_fnames is None:
            return self.__getitem__((idx + 1) % len(self.all_videos))
    
        window = []
        all_read = True
        # Deterministic augmentation based on index
        is_flip = ((idx // len(self.all_videos)) % 2) == 0
    
        for fname in window_fnames:
            try:
                img = cv2.imread(fname)
                if is_flip:
                    img = cv2.flip(img, 1)
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                all_read = False
                break
            window.append(img)
    
        if not all_read:
            return self.__getitem__((idx + 1) % len(self.all_videos))
    
        try:
            mel_out_path = join(vidname, "mel.npy")
            if os.path.isfile(mel_out_path):
                with open(mel_out_path, "rb") as f:
                    orig_mel = np.load(f)
            else:
                wavpath = os.path.join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)
                orig_mel = audio.melspectrogram(wav).T
                with open(mel_out_path, "wb") as f:
                    np.save(f, orig_mel)
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.all_videos))
    
        if is_positive:
            # For positive examples, use aligned audio
            mel = self.crop_audio_window(orig_mel.copy(), img_name)
        else:
            # For negative examples, use audio from chosen frame (already mismatched)
            mel = self.crop_audio_window(orig_mel.copy(), chosen)
    
        # Apply mel augmentation with fixed pattern
        if idx % 3 == 0:  # Apply to exactly 1/3 of samples
            mel = mask_mel(mel)
    
        if mel.shape[0] != syncnet_mel_step_size:
            return self.__getitem__((idx + 1) % len(self.all_videos))
    
        # Process images as before
        x = (np.concatenate(window, axis=2) / 255.0)
        x = x.transpose(2, 0, 1)
        x = x[:, x.shape[1]//2:]
    
        x = torch.FloatTensor(x)
        mel = torch.FloatTensor(mel.T).unsqueeze(0)
    
        return x, mel, y

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None, scheduler=None):

    global global_step, global_epoch
    resumed_step = global_step
    logger = CSVLogger(args.history_train, name=args.exp_num)

    # Track metrics per epoch
    train_metrics = {'epoch': [], 'train_loss': [], 'eval_loss': [], 'epoch_time': []}
    
    stop_training = False
    while global_epoch < nepochs:
        st_e = time()
        try:
            print(f'Starting Epoch: {global_epoch} | Best Loss: {best_loss:.6f}')
            running_loss = 0.
            correct_preds = 0
            total_preds = 0
            batch_times = []
            
            for step, (x, mel, y) in enumerate(train_data_loader):
                
                st = time()
                model.train()
                optimizer.zero_grad()

                x = x.to(device)
                mel = mel.to(device)
                y = y.to(device)
                
                a, v = model(mel, x)
                loss = cosine_loss(a, v, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                d = nn.functional.cosine_similarity(a, v)
                
                # Calculate accuracy (prediction matches label)
                pred = (d > 0.5).float()
                correct_preds += (pred.squeeze() == y.squeeze()).sum().item()
                total_preds += y.size(0)
                
                global_step += 1

                cur_session_steps = global_step - resumed_step
                running_loss += loss.item()
                
                batch_time = time() - st
                batch_times.append(batch_time)

                if step % 5 == 0:  # Print every 5 steps
                    avg_loss = running_loss/(step+1)
                    accuracy = correct_preds/total_preds if total_preds > 0 else 0
                    print(f"Step {global_step} | Sync Distance: {d.detach().cpu().clone().numpy().mean():.6f} | "
                          f"Loss: {avg_loss:.6f} | Accuracy: {accuracy:.4f} | "
                          f"Batch Time: {batch_time:.3f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
                
                # Log metrics periodically
                if global_step % 20 == 0:
                    with torch.no_grad():
                        model.eval()
                        print(f"\n--- Evaluation at step {global_step} ---")
                        eval_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir)
                        if scheduler:
                            scheduler.step(eval_loss)
                    
                    # Log metrics
                    metrics = {
                        "train_loss": running_loss / (step + 1),
                        "train_accuracy": correct_preds/total_preds if total_preds > 0 else 0,
                        "eval_loss": eval_loss,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "batch_time_avg": sum(batch_times) / len(batch_times) if batch_times else 0
                    }
                    
                    logger.log_metrics(metrics, step=global_step)
                    logger.save()
                    
                    save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch, eval_loss)
                    model.train()
                    
                    # Reset batch timing stats
                    batch_times = []
                
                del x, mel, y

            # End of epoch reporting
            epoch_time = time() - st_e
            avg_epoch_loss = running_loss / len(train_data_loader)
            accuracy = correct_preds / total_preds if total_preds > 0 else 0
            
            print(f"\nEpoch {global_epoch} completed in {epoch_time:.2f}s | "
                  f"Avg Loss: {avg_epoch_loss:.6f} | Accuracy: {accuracy:.4f}\n")
            
            # Store epoch metrics
            train_metrics['epoch'].append(global_epoch)
            train_metrics['train_loss'].append(avg_epoch_loss)
            train_metrics['epoch_time'].append(epoch_time)
            
            # Run evaluation at end of epoch if not recently done
            if global_step % 500 > 100:
                with torch.no_grad():
                    model.eval()
                    print(f"--- End of Epoch {global_epoch} Evaluation ---")
                    eval_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir)
                    train_metrics['eval_loss'].append(eval_loss)
                    save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch, eval_loss)
            
            if stop_training:
                print("The model has converged, stop training.")
                break
                
            global_epoch += 1
            
        except KeyboardInterrupt:
            print("KeyboardInterrupt - Saving checkpoint before exiting")
            save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch, 1000)
            break
            
    # Final save
    save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch, 1000)
    logger.save()
    
    # Print training summary
    print("\n=== Training Summary ===")
    print(f"Total epochs: {global_epoch}")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Total steps: {global_step}")


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir, max_eval_steps=None):
    """
    Evaluate the model on the test dataset
    
    Args:
        test_data_loader: DataLoader for validation data
        global_step: Current training step
        device: Device to run evaluation on
        model: Model to evaluate
        checkpoint_dir: Directory for checkpoints
        max_eval_steps: Maximum number of evaluation steps (None for full evaluation)
    
    Returns:
        Average evaluation loss
    """
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        losses = []
        correct_preds = 0
        total_preds = 0
        print(f'Evaluating model at step {global_step}...')

        for step, (x, mel, y) in enumerate(test_data_loader):
            if max_eval_steps is not None and step >= max_eval_steps:
                break

            # Move data to device
            x = x.to(device)
            mel = mel.to(device)
            y = y.to(device)

            # Forward pass
            a, v = model(mel, x)
            loss = cosine_loss(a, v, y)

            # Calculate similarity
            d = nn.functional.cosine_similarity(a, v)

            # Calculate accuracy
            pred = (d > 0.5).float()
            correct_preds += (pred.squeeze() == y.squeeze()).sum().item()
            total_preds += y.size(0)

            losses.append(loss.item())

            if step % 5 == 0:
                print(f"Eval step {step}/{len(test_data_loader)}, loss: {loss.item():.6f}")

        # Calculate metrics
        averaged_loss = sum(losses) / len(losses)
        accuracy = correct_preds / total_preds if total_preds > 0 else 0

        print(f"Evaluation results - Loss: {averaged_loss:.6f}, Accuracy: {accuracy:.4f}")

    return averaged_loss

def upload_file(path):
    pass

def save_ckpt(model, optimizer, step, checkpoint_dir, epoch, model_name):
    checkpoint_path = join(checkpoint_dir, model_name)
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "best_loss": best_loss,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, loss_val):
    # save best.pth
    global best_loss
    date = str(datetime.datetime.now()).split(" ")[0]
    post_fix = f'checkpoint_{hparams.img_size}_{hparams.syncnet_batch_size}_{global_step:09d}_{date}.pth'
    if loss_val < best_loss:
        best_loss = loss_val
        save_ckpt(model, optimizer, step, checkpoint_dir, epoch, f"best_syncnet_{args.exp_num}.pth")

    # last model
    save_ckpt(model, optimizer, step, checkpoint_dir, epoch, f"last_syncnet_{args.exp_num}.pth")

    prefix = "syncnet_"
    save_ckpt(model, optimizer, step, checkpoint_dir, epoch, f"{prefix}{post_fix}")

    ckpt_list = os.listdir(checkpoint_dir)
    ckpt_list = [file for file in ckpt_list if prefix in file and "checkpoint_" in file and "syncnet_" in file]
    num_ckpts = hparams.num_checkpoints
    if len(ckpt_list) <= num_ckpts*2:
        return

    ckpt_list.sort(key=lambda x: int(x.replace(".pth", "").split("_")[-2]))
    num_elim = len(ckpt_list) - num_ckpts
    elim_ckpt = ckpt_list[:num_elim]
    for ckpt in elim_ckpt:
        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        os.remove(ckpt_path)
        print("Deleted", ckpt_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch
    global best_loss

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    best_loss = checkpoint["best_loss"]

    return model


def run():
    # global global_step

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_num)
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

    train_dataset = Dataset('filelists/train.txt')
    test_dataset = Dataset('filelists/test.txt')
    hparams.set_hparam("syncnet_batch_size", 64)
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers,
        drop_last=True
    )

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=hparams.num_workers,
        drop_last=True
    )

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = nn.DataParallel(SyncNet()).to(device)
    # model = SyncNet().to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))


    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr,
                           weight_decay=1e-5)  # Add weight decay for regularization

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )


    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)
    else:
        print("Training From Scratch !!!")

    # model = nn.DataParallel(model).to(device)

    train(device, model, train_data_loader,test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs, scheduler=None)


if __name__ == "__main__":
    run()