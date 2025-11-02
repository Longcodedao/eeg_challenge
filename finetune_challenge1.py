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


class FinetuneJEPA(nn.Module):
    """Simple wrapper: EegMambaJEPA backbone -> linear regression head."""
    def __init__(self, n_chans: int = 129, d_model: int = 256, n_layer: int = 8, patch_size: int = 10):
        super().__init__()
        self.backbone = EegMambaJEPA(d_model=d_model, n_layer=n_layer, n_channels=n_chans, patch_size=patch_size)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        z = self.backbone(x)  # (B, d_model)
        out = self.head(z)    # (B, 1)
        return out


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
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--sfreq", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="finetune_challenge1.pt", help="Output model path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path to initialize model weights")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training with torch.cuda.amp (only when CUDA is available)")
    parser.add_argument("--process", action = "store_true", help="Whether to run preprocessing even if preprocessed data is found.")

    return parser.parse_args()


def build_offline_preprocessors(sfreq=100.0):
    return [
        Preprocessor(annotate_trials_with_target, 
                     target_field = "rt_from_stimulus", 
                     epoch_length = 2.0, 
                     require_stimulus = True, 
                     require_response = True, 
                     apply_on_array = False),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]


def collate_xy(batch):
    # Expect each item either (X, meta) or (X, y) depending on dataset
    Xs = []
    Ys = []
    for it in batch:
        if isinstance(it, (tuple, list)) and len(it) >= 2:
            x = np.asarray(it[0], dtype=np.float32)
            meta = it[1]
            # try to extract regression label from metadata
            y = None
            if isinstance(meta, dict) and 'rt_from_stimulus' in meta:
                y = float(meta['rt_from_stimulus'])
            elif hasattr(meta, 'get'):
                y = meta.get('rt_from_stimulus', None)
            if y is None:
                # fallback: zero label
                y = 0.0
        else:
            x = np.asarray(it, dtype=np.float32)
            y = 0.0
        Xs.append(torch.from_numpy(x).float())
        Ys.append(torch.tensor(y, dtype=torch.float32))

    Xb = torch.stack(Xs, dim=0)
    yb = torch.stack([yy.view(-1) if yy.ndim == 0 else yy for yy in Ys], dim=0)
    # Expect network to accept (B, C, T) ; ensure shape
    if Xb.ndim == 3 and Xb.shape[1] > Xb.shape[2]:
        Xb = Xb.permute(0, 2, 1)
    return Xb, yb


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
    import types
    releases = args.release if isinstance(args.release, (list, tuple)) else [args.release]
    print(f"Loading EEGChallengeDataset task={args.task}, releases={releases}")
    all_subdatasets = []

    for rel in releases:
        try:
            print(f"  - loading release {rel}")
            cache_dir = Path(data_root) / f"{rel}_mini_L100_bdf" if args.mini else Path(data_root) / f"{rel}_L100_bdf"

            ds_rel = EEGChallengeDataset(task=args.task, release=rel, 
                                         cache_dir=cache_dir, mini=args.mini,
                                         download = args.download)
            # ds_rel.datasets is a list of per-recording dataset objects
            all_subdatasets.extend(ds_rel)
        except Exception as e:
            print(f"Warning: failed to load release {rel}: {e}")

    if len(all_subdatasets) == 0:
        raise RuntimeError(f"No recordings found for task={args.task} in releases={releases}")

    # Create a lightweight namespace with a .datasets attribute so downstream functions accept it
    # ds = types.SimpleNamespace(datasets=all_subdatasets)

    # 2) Optionally preprocess (if user didn't provide preprocessed files)
    preproc_root = Path(args.preproc_root) if args.preproc_root else data_root / 'preprocessed'
    preproc_root.mkdir(parents=True, exist_ok=True)

    if args.use_preprocessed:
        print(f"Using preprocessed files under {preproc_root}. Skipping preprocess step.")
    else:
        print("Running preprocessing (this may take a while).")
        preprocessors = build_offline_preprocessors(sfreq=args.sfreq)
        for ds in all_subdatasets:
            preprocess(ds, preprocessors, n_jobs=-1, save_dir=preproc_root, overwrite=False)

    # Reload datasets from preprocessed files
    print("Loading preprocessed datasets...")
    preprocessed_datasets = []
    for ds in all_subdatasets:
        subj_id = ds.subject
        preproc_subj_dir = preproc_root / f"subj_{subj_id:03d}"
        if preproc_subj_dir.exists():
            # Expect one file per recording
            rec_files = list(preproc_subj_dir.glob("*.fif"))
            if len(rec_files) == 0:
                print(f"  - Warning: no preprocessed files found for subject {subj_id} in {preproc_subj_dir}")
                continue
            for rf in rec_files:
                try:
                    from braindecode.datasets import read_raw_fif
                    ds_rec = read_raw_fif(rf, preload=True)
                    ds_rec.subject = subj_id
                    preprocessed_datasets.append(ds_rec)
                except Exception as e:
                    print(f"  - Warning: failed to load preprocessed file {rf}: {e}")
        else:
            print(f"  - Warning: preprocessed subject directory not found: {preproc_subj_dir}")
    

    print(preprocessed_datasets)
    # 3) Create windows from events (stimulus-locked windows with RT label)
    print("Creating event-locked windows and injecting RT labels...")
    # Keep only recordings that contain our anchor and create windows
    ds_keep = keep_only_recordings_with('stimulus_anchor', ds)
    windows = create_windows_from_events(
        ds_keep,
        mapping={'stimulus_anchor': 0},
        trial_start_offset_samples=int(0.5 * args.sfreq),
        trial_stop_offset_samples=int((0.5 + args.window_sec) * args.sfreq),
        window_size_samples=int(args.window_sec * args.sfreq),
        window_stride_samples=args.sfreq,
        preload=True,
    )

    # Add extras columns (rt_from_stimulus etc)
    windows = add_extras_columns(windows, ds_keep, desc='stimulus_anchor', 
                                 keys=("target", "rt_from_stimulus", "rt_from_trialstart", "stimulus_onset", "response_onset", "correct", "response_type"))

    # 4) Split subjects
    meta = windows.get_metadata()
    subjects = meta['subject'].unique()
    valid_frac = 0.1
    test_frac = 0.1
    seed = 2025
    subjects = list(subjects)
    train_subj, valid_test_subject = train_test_split(subjects, test_size=(valid_frac + test_frac), random_state=check_random_state(seed), shuffle=True)
    valid_subj, test_subj = train_test_split(valid_test_subject, test_size=test_frac/(valid_frac+test_frac), random_state=check_random_state(seed+1), shuffle=True)

    subject_split = windows.split('subject')
    train_sets = [subject_split[s] for s in subject_split if s in train_subj]
    valid_sets = [subject_split[s] for s in subject_split if s in valid_subj]
    test_sets = [subject_split[s] for s in subject_split if s in test_subj]

    train_ds = BaseConcatDataset(train_sets)
    valid_ds = BaseConcatDataset(valid_sets)
    test_ds = BaseConcatDataset(test_sets)

    print(f"Train / Valid / Test sizes: {len(train_ds)} / {len(valid_ds)} / {len(test_ds)}")

    # 5) DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_xy)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_xy)

    # 6) Model (EegMamba JEPA backbone + linear head)
    n_chans = 129
    n_times = int(args.window_sec * args.sfreq)
    # JEPA expects (B, C, T) input; patch_size chosen to roughly match original code (10)
    model = FinetuneJEPA(n_chans=n_chans, d_model=256, n_layer=8, patch_size=10).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs-1))
    scheduler = CosineLRScheduler(optimizer,
                                  cosine_epochs = args.epochs,
                                  warmup_epochs = args.warmup_epochs,
    )

    loss_fn = torch.nn.MSELoss()

    best_rmse = float('inf')
    best_state = None

    # Optional: load checkpoint to initialize model
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            print(f"Loading checkpoint from {ckpt_path}")
            try:
                ckpt = torch.load(str(ckpt_path), map_location=device)
                # Common checkpoint shapes: full dict with 'model_state' or 'state_dict', or raw state_dict
                if isinstance(ckpt, dict):
                    if 'model_state' in ckpt:
                        sd = ckpt['model_state']
                    elif 'state_dict' in ckpt:
                        sd = ckpt['state_dict']
                    else:
                        sd = ckpt
                else:
                    sd = ckpt

                # Try loading with strict=False to allow missing keys (head/backbone differences)
                missing, unexpected = model.load_state_dict(sd, strict=False)
                print(f"Checkpoint loaded (non-strict).")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
        else:
            print(f"Checkpoint path not found: {ckpt_path}")


    # Setup AMP (automatic mixed precision) if requested and running on CUDA
    use_amp = bool(args.amp) and (device.type == 'cuda')
    if args.amp and not use_amp:
        print("--amp requested but CUDA is not available; proceeding without AMP.")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
 
    autocast_dtype = None
    if use_amp and device.type == "cuda":
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # use_scaler = (autocast_dtype is not None) and (scaler is not None)

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        running_loss = 0.0
        for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            Xb = Xb.to(device).float()
            yb = yb.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            if use_amp:
                # Mixed precision forward/backward
                with torch.amp.autocast(device_type=device.type,
                                            dtype=autocast_dtype if autocast_dtype is not None else torch.float32):
                    preds = model(Xb)
                    loss = loss_fn(preds, yb)
                    
                # Scales the loss, calls backward(), and unscales grads before the step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:

                preds = model(Xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item())

        scheduler.step()

        # val
        model.eval()
        sum_sq = 0.0
        n_samples = 0
        with torch.no_grad():
            for Xb, yb in tqdm(valid_loader, desc=f"Epoch {epoch} val"):
                Xb = Xb.to(device).float()
                yb = yb.to(device).float().view(-1, 1)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        preds = model(Xb)
                else:
                    preds = model(Xb)
                sum_sq += torch.sum((preds.view(-1) - yb.view(-1))**2).item()
                n_samples += yb.numel()

        val_rmse = (sum_sq / max(1, n_samples))**0.5
        print(f"Epoch {epoch} train_loss={running_loss/ max(1,len(train_loader)):.4f} val_rmse={val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.out)

    print(f"Saved best model (val RMSE={best_rmse:.4f}) to {args.out}")


if __name__ == '__main__':
    main()
