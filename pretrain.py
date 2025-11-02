import os
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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
from torch.amp import autocast, GradScaler
from lr_scheduler import CosineLRScheduler

import argparse
import copy
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter 

from model.eegmamba_jamba import EegMambaJEPA
from model.loss import VICReg
from model.mdn import MDNHead
from augment import JEPAGPUAugment


SFREQ = 100.0  # Target sampling frequency (Default 100 Hz)


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
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 autocast (CUDA only)")
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


def train_one_epoch(args, model, loader, criterion, optimizer, device, 
                    augmentation, epoch, writer, scheduler = None,
                    ema_decay: float = None, use_bf16: bool = False,
                    is_master: bool = True, scaler: GradScaler = None):
    
    model.train()
    running_loss = 0.0
    running_inv = 0.0
    running_var = 0.0
    running_cov = 0.0
    running_grad_norm = 0.0
    running_param_norm = 0.0
    running_batch_time = 0.0

    # If DataLoader uses a DistributedSampler, make sure its epoch is set for shuffling
    try:
        sampler = loader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
    except Exception:
        pass

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}") if is_master else enumerate(loader) 
    autocast_dtype = None
    if use_bf16 and device.type == "cuda":
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    use_scaler = (autocast_dtype is not None) and (scaler is not None)

    for step, xb in pbar:
        t0 = time.time()

        xb = xb.to(device)
        x1, x2 = _make_views(xb, augmentation)
        
        # Wrap forward pass and loss calculation with autocast regardless of whether we scale
        with autocast(device_type=device.type, 
                      dtype=autocast_dtype if autocast_dtype is not None else torch.float32):
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

        # EMA update for target encoder (handle DDP-wrapped model)
        if ema_decay is not None:
            # Unwrap DDP to access module attributes if needed
            inner_model = model.module if isinstance(model, DDP) else model
            target_exists = getattr(inner_model, "target_model", None) is not None
            update_fn = getattr(inner_model, "update_ema", None)
            if target_exists and callable(update_fn):
                try:
                    update_fn(decay=ema_decay)
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

        if (step + 1) % 10 == 0 and is_master:
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

    # Log to tensorboard (only if writer provided - rank > 0 may not have writer)
    if writer is not None:
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


# Add helper to export only the online encoder weights (exclude EMA/target copy)
def save_online_state(model: nn.Module, path: str):
    """
    Save only the online encoder parameters (exclude any attached target_model).
    Works if model is DDP-wrapped or plain module.
    """
    # Unwrap if DDP
    if isinstance(model, DDP):
        m = model.module
    else:
        m = model

    state = {}
    for k, v in m.state_dict().items():
        # Normalize key by stripping leading "module." if present (defensive)
        norm_k = k
        if norm_k.startswith("module."):
            norm_k = norm_k[len("module."):]
        # Exclude any key that belongs to a target model / EMA copy
        if "target_model" in norm_k:
            continue
        state[norm_k] = v.clone().cpu()
    torch.save(state, path)


def main():
    args = parse_args()
    # Derive distributed info from environment variables set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_distributed = world_size > 1 and dist.is_available()

    # Setup device and distributed process group if needed
    if torch.cuda.is_available():
        if use_distributed:
            torch.cuda.set_device(local_rank)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    if use_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        # make printing only from rank 0
        is_master = (dist.get_rank() == 0)
    else:
        is_master = True

    np.random.seed(args.seed + (rank if 'rank' in locals() else 0))
    torch.manual_seed(args.seed + (rank if 'rank' in locals() else 0))

    data_root = Path(args.data_root)
    preproc_root = data_root / args.preproc_out
    print(device, use_distributed, f"RANK {rank}/{world_size}", f"Master: {is_master}")
    # # 1) Load per-task datasets (lightweight: does not preprocess immediately)
    # data_total = load_task_datasets(cache_dir=data_root, tasks=args.tasks, release=args.release, mini=args.mini, download=args.download)

    # 2) Optionally preprocess and save preprocessed FIF files
    if args.preprocess:
        releases = args.release if isinstance(args.release, list) else [args.release]
        # Only master performs the expensive preprocessing step to avoid conflicts.
        preprocess_ok = True
        if is_master:
            try:
                # preprocess is I/O / CPU heavy — run only on rank 0
                preprocess_all_releases(
                    cache_dir=args.data_root,
                    tasks=args.tasks,
                    releases=releases,
                    out_dir=preproc_root,
                    mini=args.mini,
                    download=args.download,
                    sfreq=SFREQ,
                    overwrite=False,
                )
            except Exception as e:
                preprocess_ok = False
                print(f"Master preprocessing failed: {e}")
        # Synchronize so all ranks wait until master preprocessing is complete (or failed)
        if use_distributed:
            try:
                dist.barrier()
            except Exception:
                # If barrier fails, continue — subsequent steps will surface errors when reading data.
                pass
        if not preprocess_ok:
            # If master preprocessing failed, exit on master; other ranks will likely fail shortly after.
            if is_master:
                raise RuntimeError("Preprocessing failed on master. Aborting.")
 

    # 3) Create fixed-length windows from preprocessed files (if preprocessing was run), otherwise try to create from raw caches
    if is_master:
        windows_ds = create_fixed_windows_from_preprocessed(
            preproc_root=str(preproc_root),
            window_sec=args.window_sec,
            sfreq=SFREQ,
            overlap_ratio=args.overlap,
            preload=args.preload,
            min_samples=200,
            required_channels=129,
            cache_metadata=True,
            rank=rank,
            is_master=True
        )
    else:
        print(f"[RANK {rank}] Waiting for master to create windows...")

    dist.barrier()

    # ALL RANKS: Fast load
    windows_ds = create_fixed_windows_from_preprocessed(
        preproc_root=str(preproc_root),
        window_sec=args.window_sec,
        sfreq=SFREQ,
        overlap_ratio=args.overlap,
        preload=args.preload,
        min_samples=200,
        required_channels=129,
        cache_metadata=True,
        rank=rank,
        is_master=is_master
    )
    
    # Print a small summary
    if is_master:
        print("Summary:")
        print(f" - Data root: {data_root}")
        print(f" - Tasks loaded: {args.tasks}")
        print(f" - Window length (s): {args.window_sec}, overlap: {args.overlap}")
    temperal_length = int(args.window_sec * SFREQ)
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

    # Build DataLoader with DistributedSampler when using DDP
    batch_size = args.batch_size
    # When running distributed, split the requested total worker count across ranks
    if use_distributed and world_size > 0:
        n_workers = max(1, args.num_workers // world_size)
    else:
        n_workers = args.num_workers
        
    print(f"Using {n_workers} workers for data loading")

    sampler = DistributedSampler(windows_ds) if use_distributed else None

    loader = torch.utils.data.DataLoader(
         windows_ds,
         batch_size=batch_size,
         shuffle=(sampler is None),
         num_workers=n_workers,
         collate_fn=_collate_windows,
         drop_last=True,
         sampler=sampler,
         pin_memory=(device.type == 'cuda'),
     )

    # --- Training setup ---
    model = EegMambaJEPA(d_model=args.d_model, n_layer=args.n_layers).to(device)

    criterion = VICReg(d_model = args.d_model).to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), 
                            lr = args.lr)
    scheduler = CosineLRScheduler(optimizer,
                                  cosine_epochs = args.epochs,
                                  warmup_epochs = args.warmup_epochs,
    )
    scaler = GradScaler(enabled = args.bf16 and device.type == "cuda")

    # TensorBoard only on master
    log_dir = Path(args.log_dir) / 'runs' / time.strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(log_dir=str(log_dir)) if is_master else None

    # EMA and checkpoint params
    ema_decay = args.momentum_decay
    save_dir = Path(args.checkpoint_dir)
    if is_master:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint (do this before wrapping in DDP so keys match)
    start_epoch = 1
    if args.checkpoint and is_master:
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
            if 'scheduler_state' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state'])
                    print("✅ Successfully loaded scheduler state from checkpoint")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load scheduler state: {e}")
                    scheduler.last_epoch = ckpt['epoch']
            else:
                print("⚠️  No scheduler state found in checkpoint")
            start_epoch = ckpt['epoch'] + 1

    # Broadcast model and optimizer states from master to other ranks if distributed
    model.attach_target(device=device)  # ensure target encoder is created inside module BEFORE wrapping

    if use_distributed:
        # If only master loaded a checkpoint, broadcast parameters and optimizer state to others
        # Wrap model with DDP
        model.to(device)
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, output_device=local_rank if torch.cuda.is_available() else None)
        # Broadcast model params from rank 0 to all ranks for consistency
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

        # NOTE: optimizer state broadcasting is more involved; we rely on all ranks starting with same optimizer init,
        # and master-loaded state is optional. If you need exact optimizer sync, consider saving optimizer state and
        # loading it across ranks via broadcasting (not implemented here for brevity).
    else:
        model.to(device)

    # Training loop
    epochs = args.epochs
    checkpoint_interval = args.checkpoint_interval

    for epoch in range(start_epoch, epochs + 1):
        # Update sampler epoch for DDP shuffling
        if sampler is not None:
            sampler.set_epoch(epoch)

        train_loss = train_one_epoch(args, model, loader, criterion, optimizer, device,
                                     augmentation, epoch, writer, scheduler = scheduler,
                                     ema_decay = ema_decay, use_bf16 = args.bf16, is_master = is_master,
                                     scaler = scaler)
        if is_master:
            print(f"Epoch {epoch} train_loss={train_loss:.4f}")

        # Checkpointing only on master
        if is_master and (epoch % checkpoint_interval == 0):
            # Build an online-only model_state by excluding any target_model keys and stripping module prefix
            # Use helper which handles DDP-unwrapping
            online_path = save_dir / f'online_encoder_epoch{epoch:03d}.pth'
            save_online_state(model, str(online_path))

            # Save full checkpoint (model_state uses online-only dictionary)
            online_state = torch.load(str(online_path), map_location='cpu')
            ckpt = {
                'epoch': epoch,
                'model_state': online_state,
                'optimizer_state': optimizer.state_dict(),
                'criterion_state': criterion.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'train_loss': train_loss,
            }
            ckpt_path = save_dir / f'pretrain_epoch{epoch:03d}.pt'
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
            print(f"Saved online encoder weights: {online_path}")

    if writer is not None:
        writer.close()

    # Clean up distributed
    if use_distributed:
        dist.destroy_process_group()

    if is_master:
        print("Pretraining finished.")


if __name__ == '__main__':
    main()


