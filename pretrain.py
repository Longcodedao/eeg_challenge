import os
import numpy as np
import torch
from pathlib import Path
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


from build_dataset import (
    load_task_datasets, preprocess_tasks, 
    create_fixed_windows_from_preprocessed,
    preprocess_all_releases
)
from braindecode.datasets import BaseDataset 
from braindecode.preprocessing import create_fixed_length_windows

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from lr_scheduler import CosineLRScheduler
from torch.amp import autocast, GradScaler

import argparse
import copy
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter 

from model.eegmamba_jamba import EegMambaJEPA
from model.loss import VICReg
from model.mdn import MDNHead
from augment import JEPAGPUAugment

def parse_args():
    parser = argparse.ArgumentParser(description="Pretraining dataset loader and utility for JEPA-style EEG pipeline")

    # Data and preprocessing
    parser.add_argument("--data-root", type=str, default="LOL_DATASET/HBN_DATA_FULL", help="Root directory where dataset will be cached/loaded")
    parser.add_argument("--release", nargs="+", default=[f"R{i}" for i in range(1, 12)], 
                        help="Release to load (e.g., R1, R5)")
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS, help="List of task names to load")
    parser.add_argument("--mini", action="store_true", help="Use mini dataset mode (faster for debugging)")
    parser.add_argument("--download", action="store_true", help="Allow dataset download if not present")
    parser.add_argument("--preprocess", action="store_true", help="Run braindecode preprocessing step and save results")
    parser.add_argument("--preload", action="store_true", help="Preload datasets into memory")
    parser.add_argument("--preproc-out", type=str, default="preprocessed", help="Directory to save preprocessed outputs (under data-root)")
    parser.add_argument("--window-sec", type=float, default=2.0, help="Window length in seconds for fixed-length windows")
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap ratio")

    # Model / training (lightweight set to mirror JEMA defaults)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # Augmentation setting
    parser.add_argument("--noise-std", type=float, default=0.01, help="Standard deviation of Gaussian noise for augmentation")
    parser.add_argument("--time-flip-p", type=float, default=0.5, help="Probability of time-flipping augmentation")
    parser.add_argument("--channel-dropout-p", type=float, default=0.1, help="Probability of channel dropout for augmentation")
    parser.add_argument("--padding-percent", type=float, default=0.1, help="Percentage of padding for random cropping augmentation")

    # Training setting
    parser.add_argument("--epochs", type=int, default=10, help="Number of pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for pretraining")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for the batch loader")
    parser.add_argument("--warmup-epochs", type=int, default=20, help="Number of warmup epochs for LR scheduler")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--momentum-decay", type=float, default=0.995, help="EMA momentum decay for target encoder")

    parser.add_argument("--log-dir", type=str, default=".", help="Directory for TensorBoard logs")
    parser.add_argument("--checkpoint", action = "store_true", help="Whether to load from a checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default=".", help="Directory to save model checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="Epoch interval to save checkpoints")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training with torch.cuda.amp (CUDA only)")
    return parser.parse_args()


DEFAULT_TASKS = [
    "RestingState", "DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals",
    "ThePresent", "contrastChangeDetection", "seqLearning6target",
    "seqLearning8target", "surroundSupp", "symbolSearch"
]


def _collate_windows(batch):
    # Each item from braindecode window dataset is typically a tuple (X, info)
    xs = []
    for item in batch:
        if isinstance(item, (tuple, list)):
            x = item[0]
        else:
            x = item
        x = np.asarray(x, dtype=np.float32)
        t = torch.from_numpy(x)
        xs.append(t)
    xs = torch.stack(xs, dim=0)  # (B, C, T) or (B, T, C)
    # Ensure shape is (B, C, T)
    if xs.ndim == 3 and xs.shape[1] < xs.shape[2]:
        # heuristic: if second dim smaller than third, assume (B, C, T)
        pass
    elif xs.ndim == 3 and xs.shape[1] > xs.shape[2]:
        xs = xs.permute(0, 2, 1)
    return xs


def _make_views(x_batch, augmentation):
    # Create two simple stochastic views by adding noise
    view1 = augmentation(x_batch)
    view2 = augmentation(x_batch)

    return view1, view2 


def train_one_epoch(model, loader, criterion, optimizer, device, 
                    augmentation, epoch, writer, scheduler = None,
                    ema_decay: float = None, use_amp: bool = False, 
                    scaler=None):
    
    model.train()
    running_loss = 0.0
    running_inv = 0.0
    running_var = 0.0
    running_cov = 0.0
    running_grad_norm = 0.0
    running_param_norm = 0.0
    running_batch_time = 0.0

    autocast_dtype = None
    if use_amp and device.type == "cuda":
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    use_scaler = (autocast_dtype is not None) and (scaler is not None)

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")
    for step, xb in pbar:
        t0 = time.time()

        xb = xb.to(device)
        x1, x2 = _make_views(xb, augmentation)
        # Choose autocast dtype: prefer bfloat16 on supported devices, otherwise float16.

        # Wrap forward pass and loss calculation with autocast regardless of whether we scale
        with autocast(device_type=device.type, dtype=autocast_dtype if autocast_dtype is not None else torch.float32):
            # Forward pass (online encoder)
            z1 = model(x1)
            z2 = model(x2)
            # Full VICReg loss (this runs projector internally too)
            loss = criterion(z1, z2)
            
            # Compute projector outputs and components for logging inside autocast
            with torch.no_grad():
                z1p = criterion.projector(z1)
                z2p = criterion.projector(z2)
                inv_comp = float(criterion._invariance_loss(z1p, z2p).detach().cpu().item())
                var_comp = float(criterion._variance_loss(z1p, z2p).detach().cpu().item())
                cov_comp = float(criterion._covariance_loss(z1p, z2p).detach().cpu().item())

        optimizer.zero_grad()
        
        if use_scaler:
            # Scale loss for safe backward pass with float16
            scaler.scale(loss).backward()
        else:
            # Standard backward pass for FP32 or when no scaler is used
            loss.backward()

        # Gradient norm (L2) calculation requires unscaling gradients first if using a scaler
        if use_scaler:
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

        # Gradient norm (L2) - handle None grads
        total_norm_sq = 0.0
        for p in list(model.parameters()) + list(criterion.parameters()):
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_sq += param_norm.item() ** 2
        grad_norm = total_norm_sq ** 0.5

        # Parameter norm (L2)
        total_param_sq = 0.0
        for p in list(model.parameters()) + list(criterion.parameters()):
            param_norm = p.data.norm(2)
            total_param_sq += param_norm.item() ** 2
        param_norm = total_param_sq ** 0.5  

        if use_scaler:
            # scaler.step() applies the optimizer step, while skipping if gradients are inf/NaN
            scaler.step(optimizer)
            # Updates the scale factor for the next iteration
            scaler.update()
        else:
            optimizer.step()

        # EMA update for target encoder
        if ema_decay is not None and model.target_model is not None:
            try:
                model.update_target(decay=ema_decay)
            except Exception as e:
                print(f"Warning: EMA update failed. {e}")

        batch_time = time.time() - t0

        # Accumulate stats
        running_loss += loss.item()
        running_inv += inv_comp
        running_var += var_comp
        running_cov += cov_comp
        running_grad_norm += grad_norm
        running_param_norm += param_norm
        running_batch_time += batch_time

        if (step + 1) % 10 == 0:
            avg = running_loss / (step + 1)
            pbar.set_postfix({'avg_loss': f"{avg:.4f}"})
    
    if scheduler is not None:
        try:
            scheduler.step()
        except Exception:
            pass
    
    n_batches = max(1, len(loader))

    # Compute averages
    train_loss = running_loss / n_batches
    avg_inv = running_inv / n_batches
    avg_var = running_var / n_batches
    avg_cov = running_cov / n_batches
    avg_grad_norm = running_grad_norm / n_batches
    avg_param_norm = running_param_norm / n_batches
    avg_batch_time = running_batch_time / n_batches

    # Log to tensorboard
    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/invariance', avg_inv, epoch)
    writer.add_scalar('train/variance', avg_var, epoch)
    writer.add_scalar('train/covariance', avg_cov, epoch)
    writer.add_scalar('train/grad_norm', avg_grad_norm, epoch)
    writer.add_scalar('train/param_norm', avg_param_norm, epoch)
    writer.add_scalar('train/batch_time', avg_batch_time, epoch)

    # Log current learning rate
    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('train/lr', lr, epoch)

    return train_loss

def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_root = Path(args.data_root)
    preproc_root = data_root / args.preproc_out
    sfreq = 100.0  # Target sampling frequency (Default 100 Hz)

    # # 1) Load per-task datasets (lightweight: does not preprocess immediately)
    # data_total = load_task_datasets(cache_dir=data_root, tasks=args.tasks, release=args.release, mini=args.mini, download=args.download)

    # 2) Optionally preprocess and save preprocessed FIF files
    if args.preprocess:
        releases = args.release if isinstance(args.release, list) \
                    else [args.release]
        # preprocess_tasks(data_total, out_dir=preproc_root, sfreq=100.0, overwrite=False)
        preprocess_all_releases(cache_dir=args.data_root, 
                                tasks=args.tasks, 
                                releases=releases,
                                out_dir=preproc_root, 
                                mini = args.mini,
                                download = args.download,
                                sfreq=100.0, 
                                overwrite=False)
        

    # 3) Create fixed-length windows from preprocessed files (if preprocessing was run), otherwise try to create from raw caches
    try:
        windows_ds = create_fixed_windows_from_preprocessed(preproc_root, 
                                                            window_sec=args.window_sec, 
                                                            sfreq=100.0, 
                                                            overlap_ratio=args.overlap,
                                                            preload = args.preload)
        print(f"Windows dataset ready: {len(windows_ds)} examples")
    except Exception as e:
        print(f"Error creating windows from preprocessed data: {e}")

        return
        # print("Attempting to create windows directly from raw cached datasets...")
        # Fallback: create windows directly from raw cached datasets
        # all_datasets = []
        # for task, ds in data_total.items():
        #     all_datasets.append(ds)
        # concat_ds = BaseDataset.concatenate(all_datasets)
        # windows_ds = create_fixed_length_windows(
        #     concat_ds,
        #     window_size_samples=int(args.window_sec * 100.0),
        #     overlap=int(args.overlap * args.window_sec * 100.0),
        #     drop_last_window=True,
        #     shuffle=True,
        #     n_jobs=-1,
        #     preprocessors=None
        # )
        # print(f"Windows dataset ready from raw caches: {len(windows_ds)} examples")


    # Print a small summary
    print("Summary:")
    print(f" - Data root: {data_root}")
    print(f" - Tasks loaded: {args.tasks}")
    print(f" - Window length (s): {args.window_sec}, overlap: {args.overlap}")
    temperal_length = int(args.window_sec * sfreq)
    in_channels = 129  # Assuming standard EEG channel count

    augmentation = JEPAGPUAugment(
        in_channels = in_channels,
        chunk_length = temperal_length,
        padding_percent = args.padding_percent,
        noise_std = args.noise_std,
        time_flip_p = args.time_flip_p,
        channel_dropout_p = args.channel_dropout_p,
        device = device
    )

    # Build a DataLoader from the windows dataset for pretraining
    batch_size = args.batch_size
    n_workers = args.num_workers

    loader = torch.utils.data.DataLoader(windows_ds, batch_size=batch_size,
                                         shuffle=True, num_workers=n_workers,
                                         collate_fn=_collate_windows, drop_last=True,
                                         pin_memory=True)

    # --- Training setup ---
    model = EegMambaJEPA(d_model=args.d_model, n_layer=args.n_layers).to(device)

    # Target encoder for EMA (optional but useful in momentum methods)
    model.attach_target(device=device)

    criterion = VICReg(d_model = args.d_model).to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), 
                            lr = args.lr)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler = CosineLRScheduler(optimizer,
                                  cosine_epochs = args.epochs,
                                  warmup_epochs = args.warmup_epochs,
    )


    # Loading TensorBoard
    log_dir = Path(args.log_dir) / 'runs' / time.strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(log_dir=str(log_dir))

    # EMA and checkpoint params
    ema_decay = args.momentum_decay
    save_dir = Path(args.checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # # Training loop
    epochs = args.epochs
    checkpoint_interval = args.checkpoint_interval

    start_epoch = 1
    ## Load the checkpoint
    if args.checkpoint:
        ckpt_files = sorted(save_dir.glob('pretrain_epoch*.pt'), key=os.path.getmtime)
        if len(ckpt_files) == 0:
            print(f"No checkpoint files found in {save_dir}. Starting from scratch.")
        else:
            latest_ckpt = ckpt_files[-1]
            print(f"Loading checkpoint from {latest_ckpt}")
            ckpt = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            criterion.load_state_dict(ckpt['criterion_state'])

                    # Safe scheduler loading with error handling
            if 'scheduler_state' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state'])
                    print("✅ Successfully loaded scheduler state from checkpoint")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load scheduler state: {e}")
                    print("Scheduler will start from initial state (this may affect LR schedule)")
                    # Optionally: manually set last_epoch to maintain LR continuity
                    scheduler.last_epoch = ckpt['epoch']
            else:
                print("⚠️  No scheduler state found in checkpoint")

            start_epoch = ckpt['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")

    # Setup AMP (automatic mixed precision) if requested and running on CUDA
    use_amp = bool(args.amp) and (device.type == 'cuda')
    if args.amp and not use_amp:
        print("--amp requested but CUDA is not available; proceeding without AMP.")
    scaler = GradScaler(enabled = use_amp) 


    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = train_one_epoch(
            model, loader, criterion, optimizer, device,
            augmentation, epoch, writer, scheduler=scheduler,
            ema_decay=ema_decay, use_amp=use_amp, scaler=scaler,
        )
        print(f"Epoch {epoch} train_loss={train_loss:.4f}")

        # Checkpointing
        if epoch % checkpoint_interval == 0:
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'criterion_state': criterion.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'train_loss': train_loss,
            }
            ckpt_path = save_dir / f'pretrain_epoch{epoch:03d}.pt'
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


    writer.close()
    print("Pretraining finished.")


if __name__ == '__main__':
    main()


