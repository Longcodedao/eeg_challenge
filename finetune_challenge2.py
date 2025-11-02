"""Finetune pipeline for Challenge 1 (Contrast Change Detection)

This script adapts the start_kit example to run locally on the provided
dataset root (default: MyNeurIPSData/MyNeurIPSData/HBN_DATA_FULL).

It will:
 - Load the EEGChallengeDataset for the requested task/release
 - Optionally run braindecode preprocessing (or use existing preprocessed files)
 - Create event-locked windows and inject metadata (rt_from_stimulus)
 - Split subjects into train/val/test
 - Build an EEGNeX model and run a simple training loop (MSE regression of RT)

Notes:
 - This script expects the `eegdash` and `braindecode` packages available.
 - If you already ran preprocessing and consolidated files under
   <data_root>/preprocessed (per-subject folders), set --use-preprocessed
   so the script skips the heavy preprocessing step and tries to create
   windows directly.

"""
from pathlib import Path
import argparse
import time
import copy

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from tqdm import tqdm

from lr_scheduler import CosineLRScheduler
import joblib

try:
    from eegdash.dataset import EEGChallengeDataset
    from eegdash.hbn.windows import (
        annotate_trials_with_target,
        add_aux_anchors,
        add_extras_columns,
        keep_only_recordings_with,
    )
except Exception as e:
    EEGChallengeDataset = None

from braindecode.preprocessing import Preprocessor, preprocess, create_windows_from_events
from braindecode.datasets import BaseConcatDataset
import torch.nn as nn
from model.eegmamba_jamba import EegMambaJEPA
from torch.utils.tensorboard import SummaryWriter 
from torchmetrics import MetricCollection, MeanMetric, Accuracy
from torch.amp import autocast, GradScaler


SHIFT_AFTER_STIM = 0.5
SFREQ = 100
WINDOW_SEC = 2.0 
ANCHOR = "stimulus_anchor"


class FinetuneJEPA(nn.Module):
    """Simple wrapper: EegMambaJEPA backbone -> linear regression head."""
    def __init__(self, 
                 n_chans: int = 129, 
                 d_model: int = 256, 
                 n_layer: int = 8, 
                 patch_size: int = 10
                 ):
        super().__init__()
        self.backbone = EegMambaJEPA(
            d_model=d_model, 
            n_layer=n_layer, 
            n_channels=n_chans, 
            patch_size=patch_size
            )
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        z = self.backbone(x)  # (B, d_model)
        out = self.head(z)    # (B, 1)
        return out


class ContrastChangeDataset(torch.utils.data.Dataset):
    def __init__(self, braindecode_dataset):
        self.dataset = braindecode_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y, _ = self.dataset[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, epoch, writer, use_amp=False):
    model.train()
    metrics = MetricCollection({
        'mse': MeanMetric(),
        'rmse': MeanMetric(),
        'parameter_norm': MeanMetric(),
        'gradient_norm': MeanMetric(),
        # Normalize version
        'param_norm_normalized': MeanMetric(),
        'grad_norm_normalized': MeanMetric(),
    }).to(device)

    autocast_dtype = None
    if use_amp and device.type == "cuda":
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _global_l2(tensors):
        if not tensors:
            return torch.tensor(0.0, device=device)

        vec = torch.cat([t.view(-1) for t in tensors])
        return torch.linalg.norm(vec, ord = 2)

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} train")
    for batch_idx, (Xb, yb) in pbar:
        Xb = Xb.to(device)
        yb = yb.to(device)
        
        with autocast(device_type=device.type,
                        dtype=autocast_dtype if autocast_dtype is not None else torch.float32):
            preds = model(Xb)
            loss = criterion(preds, yb)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if scaler is not None:
            scaler.unscale_(optimizer)

        # Gradient norm (L2)
        grad_tensors = [p.grad for p in model.parameters() if p.grad is not None]
        grad_norm = _global_l2(grad_tensors)

        # Parameter norm (L2)
        param_tensors = [p for p in model.parameters() if p.requires_grad]
        param_norm = _global_l2(param_tensors)

        # Normalize norms
        total_elements = sum(p.numel() for p in model.parameters() if p.requires_grad)
        sqrt_N = total_elements ** 0.5
        param_norm_avg = param_norm / sqrt_N if sqrt_N > 0 else torch.tensor(0.0)
        grad_norm_avg = grad_norm / sqrt_N if sqrt_N > 0 else torch.tensor(0.0)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        metrics['mse'].update(loss.detach())
        metrics['rmse'].update(torch.sqrt(loss.detach()))
        metrics['parameter_norm'].update(param_norm)
        metrics['gradient_norm'].update(grad_norm)
        metrics['param_norm_normalized'].update(param_norm_avg)
        metrics['grad_norm_normalized'].update(grad_norm_avg)

        if (batch_idx + 1) % 10 == 0:
            avg_loss = metrics['mse'].compute().item()
            pbar.set_postfix({'AvgMSE': f"{avg_loss:.4f}"})

    if scheduler is not None:
        try:
            scheduler.step()
        except Exception:
            pass

    epoch_loss = metrics['mse'].compute().item()
    epoch_rmse = metrics['rmse'].compute().item()
    parameter_norm = metrics['parameter_norm'].compute().item()
    gradient_norm = metrics['gradient_norm'].compute().item()
    param_norm_avg = metrics['param_norm_normalized'].compute().item()
    grad_norm_avg = metrics['grad_norm_normalized'].compute().item()
    pbar.close()

    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('Train/MSE', epoch_loss, epoch)
    writer.add_scalar('Train/RMSE', epoch_rmse, epoch)
    writer.add_scalar('Train/LR', lr, epoch)
    writer.add_scalar('Train/Param_Norm', parameter_norm, epoch)
    writer.add_scalar('Train/Grad_Norm', gradient_norm, epoch)
    writer.add_scalar('Train/Param_Norm_Normalized', param_norm_avg, epoch)
    writer.add_scalar('Train/Grad_Norm_Normalized', grad_norm_avg, epoch)

    metrics.reset()
    return epoch_loss, epoch_rmse


def validate(model, loader, criterion, device, epoch, writer, use_amp=False):
    model.eval()
    metrics = MetricCollection({
        'mse': MeanMetric(),
        'rmse': MeanMetric(),
    }).to(device)

    autocast_dtype = None
    if use_amp and device.type == "cuda":
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    pbar = tqdm(total=len(loader), desc=f"Epoch {epoch} val")
    with torch.no_grad():
        for batch_idx, (Xb, yb) in enumerate(loader):
            Xb = Xb.to(device)
            yb = yb.to(device)

            with autocast(device_type=device.type,
                            dtype=autocast_dtype if autocast_dtype is not None else torch.float32):
                preds = model(Xb)
                loss = criterion(preds, yb)

            metrics['mse'].update(loss.detach())
            metrics['rmse'].update(torch.sqrt(loss.detach()))

            if (batch_idx + 1) % 10 == 0:
                avg_loss = metrics['mse'].compute().item()
                pbar.set_postfix({'AvgMSE': f"{avg_loss:.4f}"})

    epoch_loss = metrics['mse'].compute().item()
    epoch_rmse = metrics['rmse'].compute().item()
    pbar.close()

    writer.add_scalar('Val/MSE', epoch_loss, epoch)
    writer.add_scalar('Val/RMSE', epoch_rmse, epoch)

    metrics.reset()

    return epoch_loss, epoch_rmse


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune for Challenge 1 (CCD)")
    parser.add_argument("--data-root", type=str,
                        default="MyNeurIPSData/MyNeurIPSData/HBN_DATA_FULL",
                        help="Root dataset folder (raw cache).")
    parser.add_argument("--preproc-root", type=str,
                        default=None,
                        help="If set, use preprocessed files under this path (skips heavy preprocess)."
                        )
    parser.add_argument("--release", nargs='+', default=[f"R{i}" for i in range(1, 12)], help="Releases to use (e.g., R1 R5). Default: R1..R11")
    parser.add_argument("--task", type=str, default="contrastChangeDetection")
    parser.add_argument("--mini", action="store_true", help="Use mini dataset mode for debugging")
    parser.add_argument("--download", action="store_true", help="Download dataset if not found")
    parser.add_argument("--use-preprocessed", action="store_true", help="Use already preprocessed files under <data_root>/preprocessed or --preproc-root")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--pin-memory", action="store_true", help="Use pin_memory in DataLoader")

    parser.add_argument("--weight-path", type=str, required=True, help="Path to pretrained JEPA weights")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=10)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="finetune_challenge1.pt", help="Output model path")

    parser.add_argument("--checkpoint", action="store_true", help="Use checkpoint to resume training")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save intermediate checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Interval (in epochs) to save intermediate checkpoints")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training with torch.cuda.amp (only when CUDA is available)")
    parser.add_argument("--log-dir", type=str, default="runs", help="TensorBoard log directory")

    return parser.parse_args()


def build_offline_preprocessors():
    return [
        Preprocessor(annotate_trials_with_target, 
                     target_field = "rt_from_stimulus", 
                     epoch_length = 2.0, 
                     require_stimulus = True, 
                     require_response = True, 
                     apply_on_array = False),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_root)

    # Try to follow the start_kit flow when possible
    if EEGChallengeDataset is None:
        raise RuntimeError("The package `eegdash` is required for this script. Install it in your environment.")

    print(f"Using data root: {data_root}")

    # 1) Load the per-task EEGChallengeDataset
    # Support multiple releases: load datasets for each release and concatenate their recordings
    releases = args.release if isinstance(args.release, (list, tuple)) else [args.release]
    print(f"Loading EEGChallengeDataset task={args.task}, releases={releases}")

    all_subdatasets = []

    for rel in releases:
        try:
            print(f"  - loading release {rel}")
            cache_dir = Path(data_root) / f"{rel}_mini_L100_bdf" if args.mini else Path(data_root) / f"{rel}_L100_bdf"

            ds_rel = EEGChallengeDataset(task=args.task, 
                                         release=rel, 
                                         cache_dir=cache_dir, 
                                         mini=args.mini,
                                         download = args.download)
            # ds_rel.datasets is a list of per-recording dataset objects
            all_subdatasets.append(ds_rel)
        except Exception as e:
            print(f"Warning: failed to load release {rel}: {e}")

    if len(all_subdatasets) == 0:
        raise RuntimeError(f"No recordings found for task={args.task} in releases={releases}")


    # 2) Optionally preprocess (if user didn't provide preprocessed files)
    preproc_root = Path(args.preproc_root) if args.preproc_root else data_root / 'preprocessed'
    preproc_root.mkdir(parents=True, exist_ok=True)

    # if args.use_preprocessed:
    #     print(f"Using preprocessed files under {preproc_root}. Skipping preprocess step.")
    # else:
    #     print("Running preprocessing (this may take a while).")
    #     preprocessors = build_offline_preprocessors(sfreq=args.sfreq)
    #     for ds in all_subdatasets:
    #         preprocess(ds, preprocessors, n_jobs=-1, save_dir=preproc_root, overwrite=False)
    list_windows = []

    if args.use_preprocessed:
        print(f"Using preprocessed files under {preproc_root}. Skipping preprocess step.")
        for rel in releases:
            try:
                load_path = preproc_root / f"{rel}_windows.pkl"
                windows = joblib.load(load_path)
                list_windows.append(windows)
            except Exception as e:
                print(f"Warning: failed to load preprocessed windows for release {rel}: {e}")
    else:
        print("Running preprocessing (this may take a while).")
        preproc = build_offline_preprocessors()
        
        for idx, ds in tqdm(enumerate(all_subdatasets), desc = "Preprocessing"):
           
            preprocess(ds, preproc, n_jobs = -1)

            ds = keep_only_recordings_with(ANCHOR, ds)
            windows = create_windows_from_events(
                ds,
                mapping={'stimulus_anchor': 0},
                trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
                trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_SEC) * SFREQ),
                window_size_samples=int(WINDOW_SEC * SFREQ),
                window_stride_samples=SFREQ,
                preload=True,
            )

            windows = add_extras_columns(
                windows, 
                ds, 
                desc=ANCHOR,
                keys=("target", "rt_from_stimulus", "rt_from_trialstart",
                      "stimulus_onset", "response_onset", "correct", "response_type")
            )

            list_windows.append(windows)

            save_path = preproc_root / f"{releases[idx]}_windows.pkl"
            joblib.dump(windows, save_path)

    all_windows = BaseConcatDataset(list_windows)

    # 4) Dividing into train and test 
    meta = all_windows.get_metadata()
    subjects = meta['subject'].unique()
    valid_frac = 0.1
    test_frac = 0.1
    seed = 2025
    subjects = list(subjects)
    train_subj, valid_test_subject = train_test_split(subjects, 
                                                      test_size = (valid_frac + test_frac), 
                                                      random_state = check_random_state(seed), 
                                                      shuffle=True)
    valid_subj, test_subj = train_test_split(valid_test_subject, 
                                             test_size = test_frac / (valid_frac+  test_frac), 
                                             random_state = check_random_state(seed + 1), 
                                             shuffle=True)

    subject_split = all_windows.split('subject')
    train_sets = [subject_split[s] for s in subject_split if s in train_subj]
    valid_sets = [subject_split[s] for s in subject_split if s in valid_subj]
    test_sets = [subject_split[s] for s in subject_split if s in test_subj]

    train_ds = BaseConcatDataset(train_sets)
    valid_ds = BaseConcatDataset(valid_sets)
    test_ds = BaseConcatDataset(test_sets)

    train_ds = ContrastChangeDataset(train_ds)
    valid_ds = ContrastChangeDataset(valid_ds)
    test_ds = ContrastChangeDataset(test_ds)

    print(f"Train / Valid / Test sizes: {len(train_ds)} / {len(valid_ds)} / {len(test_ds)}")

    # 5) DataLoaders
    train_loader = DataLoader(train_ds, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=args.num_workers)
    valid_loader = DataLoader(valid_ds, 
                              batch_size=args.batch_size, 
                              shuffle=False, 
                              num_workers=args.num_workers, 
                              pin_memory=True)
    test_loader = DataLoader(test_ds,
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=args.num_workers, 
                                pin_memory=True)

    # 6) Model (EegMamba JEPA backbone + linear head)
    n_chans = 129
    n_times = int(WINDOW_SEC * SFREQ)
    # JEPA expects (B, C, T) input; patch_size chosen to roughly match original code (10)
    model = FinetuneJEPA(n_chans=n_chans, 
                         d_model=args.d_model, 
                         n_layer=args.n_layers, 
                         patch_size=args.patch_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineLRScheduler(optimizer,
                                  cosine_epochs = args.epochs,
                                  warmup_epochs = args.warmup_epochs,
    )
    loss_fn = torch.nn.MSELoss()
    load_finetune = True
    start_epoch = 1
    
    # Optional: load checkpoint to initialize model
    if args.checkpoint:
        ckpt_files = list(Path(args.checkpoint_dir).glob("*.pt"))
        if len(ckpt_files) == 0:
            print(f"No checkpoint files found in {args.checkpoint_dir}")
        else:
            # Sort by modification time (oldest -> newest). newest will be last.
            try:
                ckpt_files.sort(key=lambda p: p.stat().st_mtime)
            except Exception:
                # Fallback: sort by name if stat fails for some reason
                ckpt_files.sort()

            ckpt_path = ckpt_files[-1]
            print(f"Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            start_epoch = ckpt.get('epoch', 1) + 1
            missing, unexpected = model.load_state_dict(ckpt['model_state'], strict = False)
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            if 'scaler_state' in ckpt and ckpt['scaler_state'] is not None:
                scaler.load_state_dict(ckpt['scaler_state'])

            load_finetune = False
            print(f"Checkpoint loaded: epoch {ckpt['epoch']}")

            if len(missing) > 0:
                print(f"  Missing keys when loading model state: {missing}")
            if len(unexpected) > 0:
                print(f"  Unexpected keys when loading model state: {unexpected}")

        print("No checkpoint provided; fine-tuning from scratch.")

    if load_finetune:
        print("Fine-tuning from scratch.")
        weight_path = args.weight_path
        state_dict = torch.load(weight_path, map_location=device)
        model_state = state_dict["model_state"]

        model.backbone.load_state_dict(model_state, strict = False)            


    # Setup AMP (automatic mixed precision) if requested and running on CUDA
    use_amp = bool(args.amp) and (device.type == 'cuda')
    if args.amp and not use_amp:
        print("--amp requested but CUDA is not available; proceeding without AMP.")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    writer = SummaryWriter(log_dir = Path(args.log_dir) / f"finetune_challenge1_{int(time.time())}")

    # use_scaler = (autocast_dtype is not None) and (scaler is not None)
    best_rmse = float('inf')
    best_state = None

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_rmse = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, scaler,
            device, epoch, writer, use_amp=use_amp
        )
        val_loss, val_rmse = validate(
            model, valid_loader, loss_fn, device, epoch, writer, use_amp=use_amp
        )
        print(f"Epoch {epoch}: Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}")
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())

        if epoch % args.checkpoint_interval == 0:
            interim_path = Path(args.checkpoint_dir) / f"finetune_challenge1_epoch{epoch}.pt"
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict() if scaler is not None else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }   
            torch.save(checkpoint, interim_path)
            print(f"Saved interim checkpoint to {interim_path}")

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), args.out)
    print(f"Saved best model (val RMSE={best_rmse:.4f}) to {args.out}")

    # Load the weighst with best validation RMSE and evaluate on test set
    test_loss, test_rmse = validate(
        model, test_loader, loss_fn, device, epoch, writer, use_amp=use_amp
    )
    print(f"Test RMSE (best val model): {test_rmse:.4f}")



if __name__ == '__main__':
    main()
