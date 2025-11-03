"""Finetune pipeline for Challenge 2 (Predicting Externalizing Factor)

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

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from sklearn.utils import check_random_state
from tqdm import tqdm

from lr_scheduler import CosineLRScheduler
import joblib
import math

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
from braindecode.preprocessing import create_fixed_length_windows

from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset, BaseDataset

import torch.nn as nn
from model.eegmamba_jamba import EegMambaJEPA
from torch.utils.tensorboard import SummaryWriter 
from torchmetrics import MetricCollection, MeanMetric, Accuracy
from torch.amp import autocast, GradScaler
from model.meta_encoder import MetaEncoder
from model.mdn import MDNHead
from model.loss import mdn_loss
import torch.nn.functional as F

SFREQ = 100
CROP_SEC = 2
WINDOW_SEC = 4
STRIDE_SEC = 2
DESCRIPTION_FILEDS = [
    "subject", "session", "run", "task", "age", "gender", "sex", "p_factor"
]
TASK_NAMES = [
    "RestingState", "DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals",
    "ThePresent", "contrastChangeDetection", "seqLearning6target",
    "seqLearning8target", "surroundSupp", "symbolSearch"
]
# TASK_NAMES = [
#     "RestingState",
# ]

class CropMetaWrapper(BaseDataset):
    def __init__(self, windows_ds, 
                        crop_samples, 
                        meta_encoder, 
                        target_name="externalizing"):
        
        self.windows_ds = windows_ds
        self.crop_samples = crop_samples
        self.meta_encoder = meta_encoder
        self.target_name = target_name
        self.rng = np.random.default_rng(2025)  # fixed seed

    def __len__(self):
        return len(self.windows_ds)

    def __getitem__(self, idx):
        X, _, crop_inds = self.windows_ds[idx]  # X: (C, 4*SFREQ)

        # Target
        target = float(self.windows_ds.description[self.target_name])

        # Meta
        desc = self.windows_ds.description
        meta_dict = {
            "task": desc["task"],
            "sex": desc["sex"],
            "age": float(desc["age"]),
        }
        meta_vec = self.meta_encoder.transform(meta_dict)

        # Random 2s crop
        i_win, i_start, i_stop = crop_inds


        assert i_stop - i_start >= self.crop_samples

        # FIXED: .integers instead of .randint
        offset = self.rng.integers(0, i_stop - i_start - self.crop_samples + 1)
        i_start = i_start + offset
        i_stop = i_start + self.crop_samples
        X_crop = X[:, offset : offset + self.crop_samples]  # (C, 2*SFREQ)

        # Infos
        infos = {
            "subject": desc["subject"],
            "session": desc.get("session", ""),
            "run": desc.get("run", ""),
            "task": desc["task"],
            "sex": desc["sex"],
            "age": float(desc["age"]),
        }

        return torch.tensor(X_crop), meta_vec, target, (i_win, i_start, i_stop), infos


# class FinetuneJEPA_Challenge2(nn.Module):
#     def __init__(self, n_channels, d_model, n_layer, patch_size, meta_dim):
#         super(FinetuneJEPA_Challenge2, self).__init__()
#         self.backbone = EegMambaJEPA(
#             n_channels=n_channels,
#             d_model=d_model,
#             n_layer=n_layer,
#             patch_size=patch_size,
#         )

#         # Meta projection
#         self.meta_proj = nn.Linear(meta_dim, d_model)

#         # Heads
#         self.train_head = nn.Linear(d_model * 2, 1)  # Regression head
#         self.submit_head = nn.Linear(d_model, 1)  # Regression head
        
#         # Mode  
#         self.training_with_meta = True

#     # ======= Training with Meta ========
#     def forward(self, x, meta = None):
#         # x: (B, C, T)
#         z = self.backbone(x)  # (B, d_model)

#         if self.training_with_meta and meta is not None:
#             m = self.meta_proj(meta)  # (B, d_model)
#             z = torch.cat([z, m], dim = -1)  # (B, d_model * 2
#             return self.train_head(z)  # (B, 1)
#         else:
#             return self.submit_head(z)  # (B, 1)

#     # ======= Freeze and Switch =========
#     def switch_to_submit(self):
#         print("FREEZING backbone...")
#         for p in self.backbone.parameters():
#             p.requires_grad = False
#         self.training_with_meta = False
#         self.eval()

class FinetuneJEPA_Challenge2(nn.Module):
    def __init__(self, n_channels=129, d_model=256, n_layer=8, patch_size=10, meta_dim=13):
        super().__init__()
        self.backbone = EegMambaJEPA(
            n_channels=n_channels,
            d_model=d_model,
            n_layer=n_layer,
            patch_size=patch_size,
        )
        self.meta_proj = nn.Linear(meta_dim, d_model)

        # MDN Heads
        self.train_head = MDNHead(input_dim=d_model * 2, n_mixtures=3)
        self.submit_head = MDNHead(input_dim=d_model, n_mixtures=3)

        # MODE
        self.mode = "train"  # "train", "val", "submit"

    def forward(self, x, meta=None):
        z = self.backbone(x)  # (B, d_model)

        if self.mode == "train" and meta is not None:
            m = self.meta_proj(meta)
            z = torch.cat([z, m], dim=-1)
            pi, sigma, mu = self.train_head(z)
        else:
            pi, sigma, mu = self.submit_head(z)

        # RETURN BASED ON MODE
        if self.mode == "train":
            return pi, sigma, mu
        else:
            # EVALUATION / SUBMISSION: Return single number
            # Option 1: Mean of mixture (weighted)
            pred = (pi * mu).sum(dim=1)  # (B,)
            # Option 2: Best component (highest pi)
            # best_k = pi.argmax(dim=1)
            # pred = mu.gather(1, best_k.unsqueeze(1)).squeeze(1)
            return pred

    # SWITCH MODES
    def train_mode(self):
        self.mode = "train"
        self.train()

    def eval_mode(self):
        self.mode = "val"
        self.eval()

    def submit_mode(self):
        self.mode = "submit"
        self.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("BACKBONE FROZEN. SUBMISSION READY.")

    # FREEZE BACKBONE
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False



def collate_fn(batch):
    X, meta, y, _, _ = zip(*batch)

    return (
        torch.stack(X),
        torch.stack(meta),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )


def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler,
    device, epoch, writer, use_amp=False, phase="backbone"
):
    model.train_mode()  # <-- important!
    mse_loss_fn = nn.MSELoss()

    metrics = MetricCollection({
        'mdn_loss': MeanMetric(),
        'mse': MeanMetric(),
        'rmse': MeanMetric(),
        'pi_entropy': MeanMetric(),
        'pi_max': MeanMetric(),
        'active_mixtures': MeanMetric(),
        'param_norm': MeanMetric(),
        'grad_norm': MeanMetric(),
    }).to(device)

    autocast_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch} [{phase}]")
    
    for batch_idx, batch in pbar:
        # === UNPACK ===
        if phase == "backbone":
            Xb, meta, yb = batch
            meta = meta.to(device)
        else:
            Xb, _, yb = batch
        Xb, yb = Xb.to(device), yb.to(device).squeeze(-1)

        # === FORWARD ===
        with torch.autocast(device.type, dtype=autocast_dtype if use_amp else torch.float32):
            if phase == "backbone":
                pi, sigma, mu = model(Xb, meta)
            else:
                pi, sigma, mu = model(Xb)

            loss = criterion(pi, sigma, mu, yb)
            pred = (pi * mu).sum(dim=1)
            mse = mse_loss_fn(pred, yb)

        # === BACKWARD ===
        optimizer.zero_grad()
        (scaler.scale(loss) if scaler else loss).backward()
        if scaler: scaler.unscale_(optimizer)
        if scaler: scaler.step(optimizer); scaler.update()
        else: optimizer.step()

        # === MIXTURE MONITORING ===
        with torch.no_grad():
            pi_entropy = -(pi * torch.log(pi + 1e-8)).sum(dim=1).mean()
            pi_max = pi.max(dim=1).values.mean()
            active = (pi > 0.1).float().sum(dim=1).mean()  # how many >10%

        # === UPDATE METRICS ===
        metrics['mdn_loss'].update(loss)
        metrics['mse'].update(mse)
        metrics['rmse'].update(mse.sqrt())
        metrics['pi_entropy'].update(pi_entropy)
        metrics['pi_max'].update(pi_max)
        metrics['active_mixtures'].update(active)

        if (batch_idx + 1) % 10 == 0:
            pbar.set_postfix({
                'MSE': f"{metrics['mse'].compute():.4f}",
                'π-max': f"{metrics['pi_max'].compute():.3f}",
                'mix': f"{metrics['active_mixtures'].compute():.2f}"
            })

    # === LOG ===
    if scheduler: scheduler.step()

    log = {
        f'{phase}/MSE': metrics['mse'].compute(),
        f'{phase}/RMSE': metrics['rmse'].compute(),
        f'{phase}/MDN_Loss': metrics['mdn_loss'].compute(),
        f'{phase}/π_Entropy': metrics['pi_entropy'].compute(),
        f'{phase}/π_Max': metrics['pi_max'].compute(),
        f'{phase}/Active_Mixtures': metrics['active_mixtures'].compute(),
        f'{phase}/LR': optimizer.param_groups[0]['lr'],
    }
    for k, v in log.items():
        writer.add_scalar(k, v, epoch)

    metrics.reset()
    pbar.close()
    return log[f'{phase}/MSE'].item(), log[f'{phase}/RMSE'].item()



def validate(
    model, loader, device, epoch, writer,
    use_amp=False, phase="head"
):
    """
    phase: "backbone" → validate with meta (MDN)
           "head"     → validate without meta (final submission)
    """
    model.eval_mode()  # <-- switches to single-number mode
    metrics = MetricCollection({
        'mse': MeanMetric(),
        'rmse': MeanMetric(),
    }).to(device)

    autocast_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    pbar = tqdm(total=len(loader), desc=f"Epoch {epoch} val [{phase}]")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # === UNPACK ===
            if phase == "backbone":
                Xb, meta, yb = batch
                meta = meta.to(device)
            else:   
                Xb, _, yb = batch
            Xb, yb = Xb.to(device), yb.to(device).squeeze(-1)

            # === FORWARD ===
            with torch.autocast(device.type, dtype=autocast_dtype if use_amp else torch.float32):
                if phase == "backbone":
                    pred = model(Xb, meta)
                else:
                    pred = model(Xb)

                # Single prediction (weighted mean)
                mse = F.mse_loss(pred, yb)

            # === UPDATE ===
            metrics['mse'].update(mse)
            metrics['rmse'].update(mse.sqrt())

            if (batch_idx + 1) % 10 == 0:
                pbar.set_postfix({
                    'MSE': f"{metrics['mse'].compute():.4f}",
                })

            pbar.update()
            

    # === LOG ===
    log = {
        f'Val/{phase}/MSE': metrics['mse'].compute(),
        f'Val/{phase}/RMSE': metrics['rmse'].compute(),
    }
    for k, v in log.items():
        writer.add_scalar(k, v, epoch)

    pbar.close()
    metrics.reset()
    return log[f'Val/{phase}/MSE'].item(), log[f'Val/{phase}/RMSE'].item()


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
    parser.add_argument("--task", nargs='+', default=TASK_NAMES, help="Tasks to use (e.g., rest flanker). Default: all tasks")
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
    parser.add_argument("--epochs-meta", type=int, default=35, help="Number of epochs to train with meta info")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="finetune_challenge1.pt", help="Output model path")

    parser.add_argument("--checkpoint", action="store_true", help="Use checkpoint to resume training")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save intermediate checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Interval (in epochs) to save intermediate checkpoints")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training with torch.cuda.amp (only when CUDA is available)")
    parser.add_argument("--log-dir", type=str, default="runs", help="TensorBoard log directory")

    return parser.parse_args()




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
    meta_for_encoder = []

    for rel in releases:
        try:
            print(f"  - loading release {rel}")
            cache_dir = Path(data_root) / f"{rel}_mini_L100_bdf" if args.mini else Path(data_root) / f"{rel}_L100_bdf"

            for task in args.task:
                print(task)
                ds_rel = EEGChallengeDataset(task=task, 
                                             release=rel, 
                                             cache_dir=cache_dir, 
                                             mini=args.mini,
                                             download = args.download,
                                             description_fields = DESCRIPTION_FILEDS)
                
                # ds_rel.datasets is a list of per-recording dataset objects
                ## Adding the metadata to the encoder training set
                all_subdatasets.append(ds_rel)
                for sub_ds in ds_rel.datasets:
                    desc = sub_ds.description

                    meta_for_encoder.append({
                        "task": desc["task"],
                        "sex": desc["sex"],
                        "age": float(desc["age"]),
                    })

        except Exception as e:
            print(f"Warning: failed to load release {rel}: {e}")

    if len(all_subdatasets) == 0:
        raise RuntimeError(f"No recordings found for task={args.task} in releases={releases}")

    meta_encoder = MetaEncoder().fit(meta_for_encoder)
    META_DIM = meta_encoder.dim
    print(f"Meta encoder dimension: {META_DIM}")

    
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
            for task in args.task:
                try:
                    load_path = preproc_root / f"{rel}_windows_task[{task}].pkl"
                    windows = joblib.load(load_path)
                    list_windows.append(windows)
                except Exception as e:
                    print(f"Warning: failed to load preprocessed windows for release {rel}: {e}")
    else:
        print("Running preprocessing (this may take a while).")
     
        for idx, ds in tqdm(enumerate(all_subdatasets), desc = "Preprocessing"):

            rel_raw = [ds for ds in all_subdatasets if ds.release == ds.release]
            filtered = BaseConcatDataset([
                sub_ds for ds in rel_raw for sub_ds in ds.datasets
                if (sub_ds.raw.n_times >= 4 * SFREQ
                    and len(sub_ds.raw.ch_names) == 129
                    and not math.isnan(sub_ds.description.get("externalizing", math.nan)))
            ])
            windows = create_fixed_length_windows(
                filtered,
                window_size_samples= WINDOW_SEC * SFREQ,
                window_stride_samples= STRIDE_SEC * SFREQ,
                drop_last_window=True,
            )
            windows_ds = BaseConcatDataset(
                [CropMetaWrapper(
                    ds, crop_samples=CROP_SEC * SFREQ, meta_encoder=meta_encoder
                ) for ds in windows.datasets
                ]
            )

            list_windows.append(windows_ds)
            save_directory = preproc_root / f"{ds.release}_windows_task[{ds.description['task'].unique().item()}].pkl"
            joblib.dump(windows_ds, save_directory)

    all_windows = BaseConcatDataset(list_windows)

    # 4) Random split by lengths
    total_len = len(all_windows)
    train_len = int(0.8 * total_len)
    valid_len = int(0.1 * total_len)
    test_len = total_len - train_len - valid_len

    train_ds, valid_ds, test_ds = random_split(
        all_windows,
        lengths=[train_len, valid_len, test_len],
        generator= torch.Generator().manual_seed(2025)
    )

    print(f"Train / Valid / Test sizes: {len(train_ds)} / {len(valid_ds)} / {len(test_ds)}")

    # 5) DataLoaders
    train_loader = DataLoader(train_ds, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=args.num_workers,
                              pin_memory=args.pin_memory,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, 
                              batch_size=args.batch_size, 
                              shuffle=False, 
                              num_workers=args.num_workers, 
                              pin_memory=args.pin_memory,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_ds,
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=args.num_workers, 
                                pin_memory=args.pin_memory,
                                collate_fn=collate_fn)

    ## Declaring the model and training
    # 6) Model (EegMamba JEPA backbone + linear head)
    n_channels = 129

    # JEPA expects (B, C, T) input; patch_size chosen to roughly match original code (10)
    model = FinetuneJEPA_Challenge2(n_channels=n_channels, 
                                    d_model=args.d_model,
                                    n_layer=args.n_layers, 
                                    patch_size=args.patch_size, 
                                    meta_dim=META_DIM).to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineLRScheduler(optimizer,
                                  cosine_epochs = args.epochs,
                                  warmup_epochs = args.warmup_epochs,
                            )
    loss_fn = mdn_loss    
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
    writer = SummaryWriter(log_dir = Path(args.log_dir) / f"finetune_challenge2_{int(time.time())}")

    # use_scaler = (autocast_dtype is not None) and (scaler is not None)
    best_rmse = float('inf')
    best_state = None

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        phase = "backbone" if epoch <= args.epochs_meta else "head"

        train_loss, train_rmse = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, scaler,
            device, epoch, writer, use_amp=use_amp, phase=phase
        )
        val_loss, val_rmse = validate(
            model, valid_loader, device, epoch, writer, use_amp=use_amp, phase=phase
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

    # Check the parent directory exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_path)
    print(f"Saved best model (val RMSE={best_rmse:.4f}) to {out_path}")

    # Load the weighst with best validation RMSE and evaluate on test set
    test_loss, test_rmse = validate(
        model, test_loader, loss_fn, device, epoch, writer, use_amp=use_amp
    )
    print(f"Test RMSE (best val model): {test_rmse:.4f}")



if __name__ == '__main__':
    main()
