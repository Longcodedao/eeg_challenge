from pathlib import Path
import argparse
import time
import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lr_scheduler import CosineLRScheduler
import joblib
import torch.nn as nn


from utils import split_by_subjects, collate_fn_challenge2
from braindecode.datasets.base import BaseConcatDataset, BaseDataset

from torch.utils.tensorboard import SummaryWriter 
from torchmetrics import MetricCollection, MeanMetric
from torch.amp import autocast, GradScaler
import torch.nn.functional as F

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model.eegmamba_jamba import EegMambaJEPA
from model.eegmamba_jamba import FinetuneJEPA_Challenge2
from model.loss import mdn_loss

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





def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler,
    device, epoch, writer, use_amp=False, phase="backbone", 
    rank = 0, world_size = 1
):
    
    # DDP Model access: DDP model handles the distributed training, use standard model.train()
    model.train() # Use standard PyTorch train mode
    
    # Assuming 'train_mode' is a custom method on the inner model:
    model_to_norm = model.module if world_size > 1 else model
    if hasattr(model_to_norm, 'train_mode'):
        model_to_norm.train_mode()

    mse_loss_fn = nn.MSELoss()
    is_main_process = (rank == 0)

    metrics = MetricCollection({
        'mdn_loss': MeanMetric(),
        'mse': MeanMetric(),
        'rmse': MeanMetric(),
        'pi_entropy': MeanMetric(),
        'pi_max': MeanMetric(),
        'active_mixtures': MeanMetric(),
        'grad_norm': MeanMetric(),
    }).to(device)

    autocast_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train Epoch {epoch} [{phase}]") \
           if is_main_process else enumerate(loader)
    
    for batch_idx, batch in pbar:
        # === UNPACK ===
        if phase == "backbone":
            Xb, meta, yb = batch
            meta = meta.to(device)
        else:
            Xb, _, yb = batch
        Xb, yb = Xb.to(device), yb.to(device).squeeze(-1)

        # === FORWARD ===
        with autocast(device.type, dtype=autocast_dtype if use_amp else torch.float32):
            pi, sigma, mu = model(Xb, meta) if phase == "backbone" else model(Xb)

            # NOTE: Loss calculation requires no DDP-specific changes
            loss = criterion(pi, sigma, mu, yb)
            pred = (pi * mu).sum(dim=1)
            mse = mse_loss_fn(pred, yb)

        # === BACKWARD ===
        optimizer.zero_grad()
        (scaler.scale(loss) if scaler else loss).backward()

        # DDP handles gradient reduction automatically during backward() or before step()
        if scaler: scaler.unscale_(optimizer)

        # ---- GRAD NORM CLIPPING ACROSS ALL PROCESSES ----
        local_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm = float('inf')
        )
        global_norm = local_norm.clone()
        if world_size > 1:
            dist.all_reduce(global_norm, op=dist.ReduceOp.SUM)
            global_norm = global_norm / world_size

        
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
        metrics['grad_norm'].update(global_norm)


    # === LOG ===
    if scheduler: scheduler.step()
    # Aggregate metrics to Rank 0 for logging (or compute global average)
    # For simplicity, we compute local average and only Rank 0 logs.

    # 1. Compute final local metric values
    final_metrics = metrics.compute()
    
    # 2. Convert each metric tensor to an average across all processes
    # We must use torchmetrics' sync function if we want the true global average
    
    # Get a list of the metric objects
    metric_objects = [
        metrics['mdn_loss'], metrics['mse'], metrics['rmse'], metrics['pi_entropy'],
        metrics['pi_max'], metrics['active_mixtures'], metrics['grad_norm']
    ]

    # Initialize a dummy tensor on the device for the local average.
    local_averages = torch.tensor([m.compute().item() for m in metric_objects], device=device)
    
    if world_size > 1:
        # Use all_reduce to sum up all local averages
        dist.all_reduce(local_averages, op=dist.ReduceOp.SUM)
        # Divide by world_size to get the global average
        global_averages = local_averages / world_size
    else:
        global_averages = local_averages

    # 3. Log results on Rank 0
    if is_main_process:
        log_names = [
            f'Train/{phase}/MDN_Loss', f'Train/{phase}/MSE', f'Train/{phase}/RMSE', f'Train/{phase}/π_Entropy',
            f'Train/{phase}/π_Max', f'Train/{phase}/Active_Mixtures', f'Train/{phase}/Grad_Norm'
        ]
        log = dict(zip(log_names, global_averages.tolist()))
        log[f'Train/{phase}/LR'] = optimizer.param_groups[0]['lr'] # LR is the same on all ranks
        for k, v in log.items():
            writer.add_scalar(k, v, epoch)

        if pbar: pbar.close()
        
        # Return the final computed MSE and RMSE (from the globally averaged tensor)
        mse_idx = log_names.index(f'Train/{phase}/MSE')
        rmse_idx = log_names.index(f'Train/{phase}/RMSE')
        return global_averages[mse_idx].item(), global_averages[rmse_idx].item()

    # Reset metrics for the next epoch (necessary on all ranks)
    metrics.reset()
    
    # Non-main processes must return consistent, non-None values
    return None, None
    

def validate(
    model, loader, device, epoch, writer,
    use_amp=False, phase="head", rank = 0, world_size = 1
):
    """
    phase: "backbone" → validate with meta (MDN)
           "head"     → validate without meta (final submission)
    """
    model.eval()
    model_to_norm = model.module if world_size > 1 else model
    if hasattr(model_to_norm, 'eval_mode'):
        model_to_norm.eval_mode()  # <-- switches to single-number mode

    metrics = MetricCollection({
        'mse': MeanMetric(),
        'rmse': MeanMetric(),
    }).to(device)

    is_main_process = (rank == 0)
    autocast_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Valid Epoch {epoch} [{phase}]") \
           if is_main_process else enumerate(loader)


    with torch.no_grad():
        for batch_idx, batch in pbar:
            # === UNPACK ===
            if phase == "backbone":
                Xb, meta, yb = batch
                meta = meta.to(device)
            else:   
                Xb, _, yb = batch
            Xb, yb = Xb.to(device), yb.to(device).squeeze(-1)

            # === FORWARD ===
            with autocast(device.type, dtype=autocast_dtype if use_amp else torch.float32):
                if phase == "backbone":
                    pred = model(Xb, meta)
                else:
                    pred = model(Xb)

                # Single prediction (weighted mean)
                mse = F.mse_loss(pred, yb)

            # === UPDATE ===
            metrics['mse'].update(mse)
            metrics['rmse'].update(mse.sqrt())

    
    # 1. Compute final local metric values
    final_metrics = metrics.compute()
    
    # 2. Convert each metric tensor to an average across all processes
    # We must use torchmetrics' sync function if we want the true global average
    
    # Get a list of the metric objects
    metric_objects = [
        metrics['mse'], metrics['rmse'], 
    ]

    # Initialize a dummy tensor on the device for the local average.
    local_averages = torch.tensor([m.compute().item() for m in metric_objects], device=device)
    
    if world_size > 1:
        # Use all_reduce to sum up all local averages
        dist.all_reduce(local_averages, op=dist.ReduceOp.SUM)
        # Divide by world_size to get the global average
        global_averages = local_averages / world_size
    else:
        global_averages = local_averages

    # 3. Log results on Rank 0
    if is_main_process:
        log_names = [
            f'Val/{phase}/MSE', f'Val/{phase}/RMSE'
        ]
        log = dict(zip(log_names, global_averages.tolist()))

        for k, v in log.items():
            writer.add_scalar(k, v, epoch)

        if pbar: pbar.close()
        
        # Return the final computed MSE and RMSE (from the globally averaged tensor)
        mse_idx = log_names.index(f'Val/{phase}/MSE')
        rmse_idx = log_names.index(f'Val/{phase}/RMSE')
        return global_averages[mse_idx].item(), global_averages[rmse_idx].item()

    # Reset metrics for the next epoch (necessary on all ranks)
    metrics.reset()
    
    # Non-main processes must return consistent, non-None values
    return None, None



def evaluate(model, loader, device, use_amp=False):
    model.eval_mode()

    preds = []
    mse = 0.0
    with torch.no_grad():
        for batch in loader:
            Xb, _, y = batch
            Xb = Xb.to(device)
            y = y.to(device).squeeze(-1)

            with autocast(device.type, dtype=torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16):
                pred = model(Xb)    
                preds.append(pred.cpu())


                mse += F.mse_loss(pred, y)

    mse = mse.item() / len(loader)
    preds = torch.cat(preds, dim=0)
    rmse = np.sqrt(mse)
    return preds, rmse

# This is the default after preprocessing the data
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune for Challenge 1 (CCD)")
    parser.add_argument("--data-root", type=str,
                        default="preprocessed_data/challenge2",
                        help="Root dataset folder (raw cache).")
    parser.add_argument("--release", nargs='+', default=[f"R{i}" for i in range(1, 12)], help="Releases to use (e.g., R1 R5). Default: R1..R11")
    parser.add_argument("--task", nargs='+', default=TASK_NAMES, help="Tasks to use (e.g., rest flanker). Default: all tasks")
    parser.add_argument("--mini", action="store_true", help="Use mini dataset mode for debugging")


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
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for reproducibility")

    return parser.parse_args()

def main(args):

    try:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        dist.init_process_group(backend='nccl', init_method='env://')
        print(f"Initialized distributed training: rank {rank}, local_rank {local_rank}, world_size {world_size}")

    except KeyError:
        rank = 0
        local = 0
        world_size = 1
        if torch.cuda.is_available():
            print("Running in single-GPU mode")

    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available; running on CPU.")

    # Only Rank 0 creates the TensorBoard writer and print progress
    is_main_process = (rank == 0)
    writer = SummaryWriter(log_dir = Path(args.log_dir) / f"finetune_challenge2_{int(time.time())}") if is_main_process else None

    data_root = Path(args.data_root)

    # 1) Load the per-task EEGChallengeDataset
    # Support multiple releases: load datasets for each release and concatenate their recordings
    releases = args.release if isinstance(args.release, (list, tuple)) else [args.release]
    tasks = args.task if isinstance(args.task, (list, tuple)) else [args.task]

    if is_main_process:
        print(f"Loading EEGChallengeDataset task={args.task}, releases={releases}")

    list_windows = []
    total = len(releases) * len(tasks)
    pbar = tqdm(total=total, desc="Loading datasets") if is_main_process else None

    for rel in releases:
        for task in args.task:

            if is_main_process:
                pbar.set_description(f"Loading {rel} - {task}")

            preproc_file = f"{rel}_windows_task[{task}].pkl" if not args.mini else f"{rel}_mini_windows_task[{task}].pkl"
            load_path = data_root / preproc_file

            try:
                ds_rel = joblib.load(load_path)
                list_windows.append(ds_rel)
            except Exception as e:
                print(f"Error loading {load_path}: {e}")

            finally:
                if is_main_process:
                    pbar.update(1)
    
    if is_main_process:
        pbar.close()

    all_windows_ds = BaseConcatDataset(list_windows)

    if is_main_process:
        print(f"Total windows loaded: {len(all_windows_ds)}")

    # 2) Load MetaEncoder
    meta_encoder_path = data_root / 'meta_encoder.pkl'
    try: 
        meta_encoder = joblib.load(meta_encoder_path)
        META_DIM = meta_encoder.dim
        if is_main_process:
            print(f"Meta encoder loaded. Dimension: {META_DIM}")
    except Exception as e:
        raise RuntimeError(f"Error loading MetaEncoder from {meta_encoder_path}: {e}")

    
    # 3) Divide to train and test sets (80-20 split by subjects)
    # This split is deterministic due to fixed 'seed', so it's safe to run on all ranks.
    all_subjects = []
    for i in range(len(all_windows_ds.datasets)):
        # Ensure 'windows_ds' exists if the BaseConcatDataset contains wrappers
        ds_item = all_windows_ds.datasets[i]
        subject = ds_item.windows_ds.description['subject'] if hasattr(ds_item, 'windows_ds') else ds_item.description['subject']
        all_subjects.append(subject)

    subjects = list(set(all_subjects))
    total_length = len(subjects)
    train_size = int(0.8 * total_length)
    val_size = int(0.1 * total_length)
    test_size = total_length - train_size - val_size

    seed = args.seed

    train_subjects, val_test_subjects = train_test_split(
        subjects, train_size=train_size, random_state=seed
    )

    val_subjects, test_subjects = train_test_split(
        val_test_subjects, train_size=val_size, random_state=seed
    )

    train_windows_ds = split_by_subjects(all_windows_ds, train_subjects)
    val_windows_ds = split_by_subjects(all_windows_ds, val_subjects)
    test_windows_ds = split_by_subjects(all_windows_ds, test_subjects)

    if is_main_process:
        print(f"Train windows: {len(train_windows_ds)}")
        print(f"Validation windows: {len(val_windows_ds)}")
        print(f"Test windows: {len(test_windows_ds)}")

    # --- DDP Adaptation: Use DistributedSampler if in distributed mode
    train_sampler = DistributedSampler(
        train_windows_ds, num_replicas=world_size, rank=rank, shuffle=True
    ) 

    val_sampler = DistributedSampler(
        val_windows_ds, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_sampler = DistributedSampler(
        test_windows_ds, num_replicas=world_size, rank=rank, shuffle=False
    )

    # Batch size is divided among all GPUs
    batch_size_per_gpu = args.batch_size // world_size

    train_loader = DataLoader(
        train_windows_ds, 
        batch_size = batch_size_per_gpu, 
        sampler = train_sampler if world_size > 1 else None,
        shuffle = False if world_size > 1 else True,
        collate_fn = collate_fn_challenge2,
        num_workers = args.num_workers,
        pin_memory = args.pin_memory
    )

    val_loader = DataLoader(
        val_windows_ds, 
        batch_size=batch_size_per_gpu, 
        sampler = val_sampler if world_size > 1 else None,
        shuffle = False,
        collate_fn = collate_fn_challenge2,
        num_workers = args.num_workers,
        pin_memory = args.pin_memory
    )

    test_loader = DataLoader(
        test_windows_ds, 
        batch_size=batch_size_per_gpu,
        shuffle = False,
        collate_fn=collate_fn_challenge2,
        num_workers = args.num_workers,
        pin_memory = args.pin_memory
    )


    # 4) Create model
    n_channels = 129

    # JEPA expects (B, C, T) input; patch_size chosen to roughly match original code (10)
    model_orig = FinetuneJEPA_Challenge2(n_channels=n_channels, 
                                    d_model=args.d_model,
                                    n_layer=args.n_layers, 
                                    patch_size=args.patch_size, 
                                    meta_dim=META_DIM).to(device)
    model = DDP(model_orig, device_ids = [local_rank], find_unused_parameters=True) if world_size > 1 else model_orig

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineLRScheduler(optimizer,
                                  cosine_epochs = args.epochs,
                                  warmup_epochs = args.warmup_epochs,
                            )
    # Setup AMP (automatic mixed precision) if requested and running on CUDA
    use_amp = bool(args.amp) and (device.type == 'cuda')
    loss_fn = mdn_loss  
    scaler = None

    if use_amp:
        scaler = GradScaler(device)
        if is_main_process:
            print("Using mixed precision training with torch.amp.")
    
      
    load_finetune = True
    start_epoch = 1

    # Get the inner model reference for state_dict operations
    model_to_load = model.module if world_size > 1 else model

    # Optional: load checkpoint to initialize model
    if args.checkpoint:
        if is_main_process:
            print("Loading checkpoint from provided path.")
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

                # Load to CPU first, then map to device to save VRAM
                ckpt = torch.load(ckpt_path, map_location="cpu")

                # Load state into the UNWRAPPED model (model_to_load)
                missing, unexpected = model_to_load.load_state_dict(ckpt['model_state'], strict = False)

                optimizer.load_state_dict(ckpt['optimizer_state'])
                scheduler.load_state_dict(ckpt['scheduler_state'])
                if 'scaler_state' in ckpt and ckpt['scaler_state'] is not None:
                    scaler.load_state_dict(ckpt['scaler_state'])

                load_finetune = False
                print(f"Checkpoint loaded: epoch {ckpt['epoch']}")

                start_epoch = ckpt.get('epoch', 1) + 1

                if len(missing) > 0:
                    print(f"  Missing keys when loading model state: {missing}")
                if len(unexpected) > 0:
                    print(f"  Unexpected keys when loading model state: {unexpected}")


        if world_size > 1:
            dist.barrier()  # Ensure all processes wait for Rank 0 to load checkpoint

    if load_finetune:
        if is_main_process:
            print("Fine-tuning from scratch.")
            weight_path = args.weight_path
            state_dict = torch.load(weight_path, map_location=device)
            model_state = state_dict["model_state"]

            model_to_load.backbone.load_state_dict(model_state, strict = False)            

        if world_size > 1:
            dist.barrier()  # Ensure all processes wait for Rank 0 to load weights

    # use_scaler = (autocast_dtype is not None) and (scaler is not None)
    best_rmse = float('inf')
    best_state = None

    if is_main_process:
        print("Starting training...")
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
            # test_loader.sampler.set_epoch(epoch)

        phase = "backbone" if epoch <= args.epochs_meta else "head"

        # --- TRAINING: All ranks participate (pass rank/world_size to train_one_epoch) ---
        train_loss, train_rmse = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, scaler,
            device, epoch, writer, use_amp=use_amp, phase=phase, rank=rank, world_size=world_size
        )
        # # --- SYNCHRONIZATION: Wait for all ranks to finish training ---
        # if world_size > 1:
        #     dist.barrier()
        # --- VALIDATION: Only Rank 0 logs and prints ---
        val_loss, val_rmse = validate(
            model, val_loader, device, epoch, writer, use_amp=use_amp, phase=phase, rank=rank, world_size=world_size
        )

        if is_main_process:
            print(f"Epoch {epoch}: Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}")

            # Get the state dict from the unwrapped model
            current_model_state = model_to_load.state_dict()

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = copy.deepcopy(current_model_state)

            if epoch % args.checkpoint_interval == 0:
                interim_path = Path(args.checkpoint_dir) / f"finetune_challenge2_epoch{epoch}.pt"
                checkpoint = {
                    'epoch': epoch,
                    'model_state': current_model_state,
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'scaler_state': scaler.state_dict() if scaler is not None else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }   
                torch.save(checkpoint, interim_path)
                print(f"Saved interim checkpoint to {interim_path}")

        if world_size > 1:
            dist.barrier()  # Ensure all ranks wait for Rank 0 to finish saving

    # --- TRAINING COMPLETE ---
    if is_main_process:
        print("Training complete. Testing best model on test set...")
        if best_state is not None:
            model_to_load.load_state_dict(best_state)
            print(f"Loaded best model from training (val RMSE={best_rmse:.4f})")
        
        # Check the parent directory exists
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(model_to_load.state_dict(), out_path)
        print(f"Saved best model (val RMSE={best_rmse:.4f}) to {out_path}")

        # Load the weighst with best validation RMSE and evaluate on test set
        preds, test_rmse = evaluate(
            model_to_load, test_loader, device, use_amp=use_amp
        )


        print(f"Test RMSE (best val model): {test_rmse:.4f}")

    if world_size > 1:
        dist.barrier()  # Ensure all ranks wait for Rank 0 to finish saving


    dist.destroy_process_group()

if __name__ == '__main__':
    args = parse_args()
    main(args)
