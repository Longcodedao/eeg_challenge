# ##########################################################################
# # EEG Foundation Challenge 2025
# # JEMA-EEG25 Full End-to-End Training Script (Single H200 Version)
# #
# # This script runs the complete STAGE 1 (Pre-training) and
# # STAGE 2 (Fine-tuning) on a single high-VRAM GPU (e.g., H200).
# # It uses the corrected EegMambaJEPA architecture with patch embedding.
# ##########################################################################


# =============================================================================
# STEP 1: IMPORTS
# =============================================================================
print("--- Step 1: Importing Libraries ---")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
import os
import gc
import copy
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time # For timing epochs

# Mamba and Vision Imports
try:
    from mamba_ssm import Mamba
    from einops import rearrange
except ImportError:
    print("Mamba-SSM not found. Ensure it was built/installed correctly in Phase 2.")
    raise

# EEG Data Loading Imports
try:
    from eegdash.datasets import EEGDash
    from braindecode.preprocessing import (
        Preprocessor,
        NumpyToTensor,
        scale,
    )
    from braindecode.datautil import create_fixed_length_windows
except ImportError:
    print("eegdash or braindecode not found. Ensure they are installed.")
    raise

print("Libraries imported successfully.")
print(f"PyTorch Version: {torch.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
print(f"Using GPU: {gpu_name}")


# =============================================================================
# STEP 2: CONFIGURATION (CFG) - Updated for H200
# =============================================================================
class CFG:
    # --- [USER ACTION REQUIRED] ---
    # VERIFY this path matches your Vast.ai storage mount point
    RAW_DATA_PATH = "/mnt/storage/HBN_DATA_FULL/"

    # Path to save the final models (e.g., /root/ or a storage path)
    SAVE_PATH = "." # Save in the current directory where the script is run

    # Data & Preprocessing
    SFREQ = 100
    N_CHANNELS = 129
    PRETRAIN_CHUNK_S = 6  # 6 seconds for pre-training chunks
    FINETUNE_CHUNK_S = 2  # 2 seconds for fine-tuning (matches starter kit)
    PRETRAIN_CHUNK_SIZE = PRETRAIN_CHUNK_S * SFREQ
    FINETUNE_CHUNK_SIZE = FINETUNE_CHUNK_S * SFREQ

    # Task names for pre-training (all available tasks)
    TASK_NAMES = [
        "RestingState", "DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals",
        "ThePresent", "contrastChangeDetection", "seqLearning6target",
        "seqLearning8target", "surroundSupp", "symbolSearch"
    ]

    # Release names for pre-training (all 11)
    RELEASE_NAMES = [f"R{i}" for i in range(1, 12)] # R1 to R11

    # Training Hyperparameters
    N_SPLITS = 5  # K-Fold Cross-validation
    SEED = 42

    # Stage 1: Pre-training
    # Increased batch size for H200's large VRAM (141GB)
    # Tune this based on actual memory usage. Start with 256.
    PRETRAIN_BATCH_SIZE = 256
    PRETRAIN_EPOCHS = 50      # Number of epochs for self-supervised pre-training
    PRETRAIN_LR = 1e-4
    EMA_DECAY = 0.996

    # Stage 2: Fine-tuning
    # Increased batch size for H200
    # Tune this based on actual memory usage. Start with 512.
    FINETUNE_BATCH_SIZE = 512
    FINETUNE_EPOCHS_CH1 = 30
    FINETUNE_EPOCHS_CH2 = 30
    FINETUNE_LR = 1e-3

    # Model Architecture
    D_MODEL = 256
    N_LAYERS = 8
    PATCH_SIZE = 10  # 10 samples = 0.1s per patch
    D_STATE = 16
    EXPAND = 2
    D_CONV = 4 # Mamba default conv kernel size

    # Loss
    VICREG_LAMBDA = 25.0
    VICREG_MU = 25.0
    MDN_COMPONENTS = 5

    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4 # Increase for potentially faster data loading on multi-core CPU

# Set random seed for reproducibility
np.random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.SEED)
    # Optional: For better reproducibility, but might slow down training slightly
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# =============================================================================
# STEP 3: MODEL DEFINITIONS (Copied from my_model.py for completeness)
# =============================================================================
print("\n--- Step 3: Defining Model Architecture ---")

# --- Patch Embedding Layer ---
class PatchEmbed(nn.Module):
    def __init__(self, n_channels=CFG.N_CHANNELS, embed_dim=CFG.D_MODEL, patch_size=CFG.PATCH_SIZE):
        super().__init__()
        self.proj = nn.Conv1d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x); x = x.permute(0, 2, 1); return x # (B, NumPatches, D_MODEL)

# --- Bi-Directional Mamba Block ---
class NeuroBiMambaBlock(nn.Module):
    def __init__(self, d_model=CFG.D_MODEL, d_state=CFG.D_STATE, expand=CFG.EXPAND, d_conv=CFG.D_CONV):
        super().__init__()
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False) # Mamba often omits bias here
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner, bias=True
        )
        self.activation = nn.SiLU()
        self.mamba_fwd = Mamba(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=1)
        self.mamba_bwd = Mamba(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=1)
        self.out_proj = nn.Linear(2 * self.d_inner, d_model, bias=False) # Mamba often omits bias here
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_skip = x; x = self.norm(x)
        x_proj = self.in_proj(x); x_conv_in, res = x_proj.chunk(2, dim=-1)
        x_conv_in = rearrange(x_conv_in, "b l d -> b d l")
        x_conv_out = self.conv1d(x_conv_in)[:, :, :x.shape[1]] # Causal crop
        x_conv_out = rearrange(x_conv_out, "b d l -> b l d")
        x_conv_activated = self.activation(x_conv_out)
        x_fwd = self.mamba_fwd(x_conv_activated)
        x_bwd = torch.flip(self.mamba_bwd(torch.flip(x_conv_activated, dims=[1])), dims=[1])
        x_mamba_out = torch.cat([x_fwd, x_bwd], dim=-1)
        x_out = self.out_proj(x_mamba_out * self.activation(res)) # Gated MLP
        return x_out + x_skip

# --- JEPA Backbone ---
class EegMambaJEPA(nn.Module):
    def __init__(self, d_model=CFG.D_MODEL, n_layer=CFG.N_LAYERS, n_channels=CFG.N_CHANNELS,
                 patch_size=CFG.PATCH_SIZE, d_state=CFG.D_STATE, expand=CFG.EXPAND):
        super().__init__()
        self.patch_embed = PatchEmbed(n_channels, d_model, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.mamba_blocks = nn.Sequential(
            *[NeuroBiMambaBlock(d_model=d_model, d_state=d_state, expand=expand) for _ in range(n_layer)]
        )
        self.norm_f = nn.LayerNorm(d_model)
    def forward(self, x):
        B = x.shape[0]; x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1); x = torch.cat((cls_tokens, x), dim=1)
        x = self.mamba_blocks(x); x = self.norm_f(x)
        return x[:, 0] # Return CLS token

# --- VICReg Loss ---
class VICReg(nn.Module):
    def __init__(self, d_model=CFG.D_MODEL, lambda_val=CFG.VICREG_LAMBDA, mu_val=CFG.VICREG_MU, eps=1e-4):
        super().__init__()
        self.lambda_val = lambda_val; self.mu_val = mu_val; self.eps = eps
        # Simplified projector - adjust if needed
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    def forward(self, z1, z2):
        z1p = self.projector(z1); z2p = self.projector(z2)
        repr_loss = nn.functional.mse_loss(z1p, z2p)
        z1_norm = z1p - z1p.mean(dim=0); z2_norm = z2p - z2p.mean(dim=0)
        std_z1 = torch.sqrt(z1_norm.var(dim=0) + self.eps); std_z2 = torch.sqrt(z2_norm.var(dim=0) + self.eps)
        std_loss = (torch.mean(nn.functional.relu(1 - std_z1)) + torch.mean(nn.functional.relu(1 - std_z2))) / 2
        B = z1_norm.shape[0]
        cov_z1 = (z1_norm.T @ z1_norm) / (B - 1); cov_z2 = (z2_norm.T @ z2_norm) / (B - 1)
        off_diag_mask = ~torch.eye(z1_norm.shape[1], device=z1_norm.device).bool()
        cov_loss = (cov_z1[off_diag_mask].pow(2).sum() / z1_norm.shape[1] +
                    cov_z2[off_diag_mask].pow(2).sum() / z1_norm.shape[1])
        loss = self.lambda_val * repr_loss + self.mu_val * std_loss + cov_loss
        return loss

# --- MDN Head & Loss ---
class MDNHead(nn.Module):
    def __init__(self, input_dim=CFG.D_MODEL, n_components=CFG.MDN_COMPONENTS):
        super().__init__()
        self.n_components = n_components
        self.pi = nn.Linear(input_dim, n_components)
        self.sigma = nn.Linear(input_dim, n_components)
        self.mu = nn.Linear(input_dim, n_components)
    def forward(self, x):
        pi = torch.softmax(self.pi(x), dim=1)
        sigma = torch.exp(self.sigma(x)) + 1e-6 # Ensure positive sigma
        mu = self.mu(x)
        return pi, sigma, mu

def mdn_loss(pi, sigma, mu, y, reduce=True):
    # Ensure y has the correct shape for broadcasting
    if y.dim() == 1: y = y.unsqueeze(-1) # (B,) -> (B, 1)
    if y.dim() == 2 and y.shape[1] != 1: y = y.unsqueeze(-1) # (B, D) -> (B, D, 1)

    m = torch.distributions.Normal(loc=mu, scale=sigma)
    # log_prob shape: (B, N_COMPONENTS) if y shape is (B, 1)
    # Need y shape (B, 1) to broadcast correctly against mu/sigma (B, N_COMPONENTS)
    log_prob = m.log_prob(y) # y will broadcast to (B, N_COMPONENTS)
    # Ensure log_prob is numerically stable
    log_prob = torch.clamp(log_prob, min=-1e9) # Prevent -inf
    
    # Calculate log-sum-exp safely
    log_pi = torch.log_softmax(pi, dim=1) # Use log_softmax for stability
    log_likelihood = torch.logsumexp(log_pi + log_prob, dim=1)
    
    loss = -log_likelihood
    if reduce:
        return loss.mean()
    else:
        return loss

# --- Fine-tuning Model Wrapper ---
class EnsembleCompetitionModel(nn.Module):
    def __init__(self, backbones, head):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        self.head = head
    def forward(self, x):
        # Freeze backbones during eval if needed (already handled in training loop)
        features = [bb(x) for bb in self.backbones]
        x = torch.mean(torch.stack(features), dim=0) # Average features
        return self.head(x)

print("Model definitions complete.")

# =============================================================================
# STEP 4: DATA LOADING (Updated num_workers)
# =============================================================================
print("\n--- Step 4: Setting up Data Loaders ---")

# --- Pre-training Dataset (Applies JEPA Augmentations) ---
class JEPADataset(Dataset):
    def __init__(self, eegdash_dataset, chunk_size_s):
        self.eegdash_dataset = eegdash_dataset; self.chunk_size_s = chunk_size_s
        self.sfreq = self.eegdash_dataset.sfreq; self.chunk_size_samples = int(chunk_size_s * self.sfreq)
        # Define two independent augmentation pipelines
        self.transform1 = self.get_transform(); self.transform2 = self.get_transform()
    def get_transform(self):
        # Apply transforms on the GPU within the training loop for efficiency
        return nn.Identity() # Placeholder - augmentations done in training loop
    def __len__(self): return len(self.eegdash_dataset)
    def __getitem__(self, idx):
        X, y, info = self.eegdash_dataset[idx]
        # Return raw data, augmentation happens later
        return torch.tensor(X, dtype=torch.float32), torch.tensor(0.0) # Dummy target

# --- GPU-based Augmentation (efficient for JEPA) ---
class JEPAGpuAugment:
    def __init__(self, n_channels=CFG.N_CHANNELS, chunk_size_samples=CFG.PRETRAIN_CHUNK_SIZE, device=CFG.DEVICE):
        self.chunk_size_samples = chunk_size_samples
        self.device = device
        # Pad slightly more to ensure crop is always possible
        self.pad_size = int(0.1 * chunk_size_samples)
        self.pad = nn.ReplicationPad1d((self.pad_size, self.pad_size)).to(device)

    def __call__(self, x_batch):
        # x_batch shape: (B, C, T_original)
        B, C, T_orig = x_batch.shape
        x_batch = x_batch.to(self.device)
        
        # 1. Pad
        x_padded = self.pad(x_batch) # (B, C, T_orig + 2*pad)
        
        # 2. Random Crop (efficient batch-wise crop)
        max_start = x_padded.shape[2] - self.chunk_size_samples
        start_indices = torch.randint(0, max_start + 1, (B,), device=self.device)
        
        # Use torch.gather or indexing (advanced indexing might be complex)
        # Simpler: Loop (less efficient but clear) - or use unfold/gather
        views = []
        for i in range(B):
            start = start_indices[i]
            views.append(x_padded[i:i+1, :, start : start + self.chunk_size_samples])
        cropped_batch = torch.cat(views, dim=0) # (B, C, chunk_size_samples)
        
        # 3. Add Noise
        noise = torch.randn_like(cropped_batch) * 0.01
        augmented_batch = cropped_batch + noise
        
        return augmented_batch

# --- Function to update target network (EMA) ---
@torch.no_grad()
def update_target_network(online_net, target_net, decay=CFG.EMA_DECAY):
    for online_params, target_params in zip(online_net.parameters(), target_net.parameters()):
        target_params.data = target_params.data * decay + online_params.data * (1 - decay)

# --- Function to fetch pre-training data loaders ---
def fetch_pretrain_loader(fold_idx, kf, all_subjects):
    train_idx, val_idx = list(kf.split(all_subjects))[fold_idx]
    train_subjects = [all_subjects[i] for i in train_idx]
    val_subjects = [all_subjects[i] for i in val_idx]
    print(f"Fold {fold_idx+1}: {len(train_subjects)} train, {len(val_subjects)} val subjects")

    # Load dataset using EEGDash - load slightly larger chunk initially
    eeg_dataset = EEGDash(
        data_path=CFG.RAW_DATA_PATH, tasks=CFG.TASK_NAMES, releases=CFG.RELEASE_NAMES,
        subjects=train_subjects, chunk_size_s=CFG.PRETRAIN_CHUNK_S + 2, # Load larger for padding
        preload=False, bdf=True
    )
    jepa_train_dataset = JEPADataset(eeg_dataset, CFG.PRETRAIN_CHUNK_S)
    train_loader = DataLoader(
        jepa_train_dataset, batch_size=CFG.PRETRAIN_BATCH_SIZE, shuffle=True,
        num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True
    )

    eeg_dataset_val = EEGDash(
        data_path=CFG.RAW_DATA_PATH, tasks=CFG.TASK_NAMES, releases=CFG.RELEASE_NAMES,
        subjects=val_subjects, chunk_size_s=CFG.PRETRAIN_CHUNK_S + 2,
        preload=False, bdf=True
    )
    jepa_val_dataset = JEPADataset(eeg_dataset_val, CFG.PRETRAIN_CHUNK_S)
    val_loader = DataLoader(
        jepa_val_dataset, batch_size=CFG.PRETRAIN_BATCH_SIZE, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=False
    )
    return train_loader, val_loader

# --- Function to fetch fine-tuning data loaders ---
def fetch_finetune_loaders():
    print("Loading fine-tuning data (from R5)...")
    try:
        participants_df = pd.read_csv(os.path.join(CFG.RAW_DATA_PATH, 'participants.tsv'), sep='\t')
    except Exception as e:
        print(f"FATAL ERROR loading participants.tsv from {CFG.RAW_DATA_PATH}: {e}")
        raise
    r5_subjects = participants_df[participants_df['Release'] == 'R5']['participant_id'].tolist()

    # --- CH1 Dataset ---
    eeg_dataset_ch1 = EEGDash(
        data_path=CFG.RAW_DATA_PATH, tasks=["contrastChangeDetection"], releases=["R5"],
        subjects=r5_subjects, chunk_size_s=CFG.FINETUNE_CHUNK_S, preload=True, bdf=True
    )
    labels_ch1 = [info.get('RT_correct', np.nan) for info in eeg_dataset_ch1.infos]
    eeg_dataset_ch1.add_targets(labels_ch1, 'RT_correct')
    valid_indices_ch1 = [i for i, y in enumerate(labels_ch1) if not np.isnan(y)]
    eeg_dataset_ch1 = Subset(eeg_dataset_ch1, valid_indices_ch1)
    train_idx, val_idx = train_test_split(list(range(len(eeg_dataset_ch1))), test_size=0.2, random_state=CFG.SEED)
    train_ds_ch1 = Subset(eeg_dataset_ch1, train_idx); val_ds_ch1 = Subset(eeg_dataset_ch1, val_idx)

    # --- CH2 Dataset ---
    eeg_dataset_ch2 = EEGDash(
        data_path=CFG.RAW_DATA_PATH, tasks=CFG.TASK_NAMES, releases=["R5"],
        subjects=r5_subjects, chunk_size_s=CFG.FINETUNE_CHUNK_S, preload=True, bdf=True
    )
    labels_ch2 = [info.get('Externalizing', np.nan) for info in eeg_dataset_ch2.infos]
    eeg_dataset_ch2.add_targets(labels_ch2, 'Externalizing')
    valid_indices_ch2 = [i for i, y in enumerate(labels_ch2) if not np.isnan(y)]
    eeg_dataset_ch2 = Subset(eeg_dataset_ch2, valid_indices_ch2)
    train_idx, val_idx = train_test_split(list(range(len(eeg_dataset_ch2))), test_size=0.2, random_state=CFG.SEED)
    train_ds_ch2 = Subset(eeg_dataset_ch2, train_idx); val_ds_ch2 = Subset(eeg_dataset_ch2, val_idx)

    # --- DataLoaders ---
    def finetune_collate(batch):
        X = torch.stack([torch.tensor(item[0], dtype=torch.float32) for item in batch])
        y = torch.tensor([item[1] for item in batch], dtype=torch.float32).view(-1, 1)
        return X, y

    ch1_train = DataLoader(train_ds_ch1, batch_size=CFG.FINETUNE_BATCH_SIZE, collate_fn=finetune_collate, shuffle=True, num_workers=CFG.NUM_WORKERS, drop_last=True, pin_memory=True)
    ch1_val = DataLoader(val_ds_ch1, batch_size=CFG.FINETUNE_BATCH_SIZE, collate_fn=finetune_collate, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    ch2_train = DataLoader(train_ds_ch2, batch_size=CFG.FINETUNE_BATCH_SIZE, collate_fn=finetune_collate, shuffle=True, num_workers=CFG.NUM_WORKERS, drop_last=True, pin_memory=True)
    ch2_val = DataLoader(val_ds_ch2, batch_size=CFG.FINETUNE_BATCH_SIZE, collate_fn=finetune_collate, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    print(f"CH1: {len(train_ds_ch1)} train, {len(val_ds_ch1)} val samples")
    print(f"CH2: {len(train_ds_ch2)} train, {len(val_ds_ch2)} val samples")
    return ch1_train, ch1_val, ch2_train, ch2_val

print("Data loading functions defined.")

# =============================================================================
# STEP 5: STAGE 1 - SELF-SUPERVISED PRE-TRAINING
# =============================================================================
def run_pretraining():
    print("\n" + "="*50); print("STARTING STAGE 1: SELF-SUPERVISED PRE-TRAINING"); print("="*50)
    
    try:
        participants_df = pd.read_csv(os.path.join(CFG.RAW_DATA_PATH, 'participants.tsv'), sep='\t')
        all_subjects = participants_df['participant_id'].unique().tolist()
    except Exception as e:
        print(f"FATAL ERROR loading participants.tsv: {e}"); raise
        
    kf = KFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.SEED)
    best_fold_losses = []
    gpu_augment = JEPAGpuAugment() # Initialize GPU augmentation

    for fold in range(CFG.N_SPLITS):
        print(f"\n--- Starting Pre-training Fold {fold+1}/{CFG.N_SPLITS} ---")
        start_fold_time = time.time()
        
        train_loader, val_loader = fetch_pretrain_loader(fold, kf, all_subjects)
        
        online_encoder = EegMambaJEPA(
            d_model=CFG.D_MODEL, n_layer=CFG.N_LAYERS, n_channels=CFG.N_CHANNELS,
            patch_size=CFG.PATCH_SIZE, d_state=CFG.D_STATE, expand=CFG.EXPAND
        ).to(CFG.DEVICE)
        target_encoder = copy.deepcopy(online_encoder).to(CFG.DEVICE)
        
        for param in target_encoder.parameters(): param.requires_grad = False
        criterion = VICReg().to(CFG.DEVICE)
        optimizer = optim.AdamW(online_encoder.parameters(), lr=CFG.PRETRAIN_LR)
        # Adjust T_max for potentially fewer steps per epoch if dataset loading is slow
        scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * CFG.PRETRAIN_EPOCHS)
        best_val_loss = float('inf')
        
        for epoch in range(CFG.PRETRAIN_EPOCHS):
            epoch_start_time = time.time()
            # --- Training ---
            online_encoder.train()
            train_loss = 0; num_batches = 0
            pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{CFG.PRETRAIN_EPOCHS} [Train]")
            for X_batch, _ in pbar:
                # Augment views on GPU
                view1 = gpu_augment(X_batch); view2 = gpu_augment(X_batch)
                
                optimizer.zero_grad()
                z1 = online_encoder(view1)
                with torch.no_grad(): z2 = target_encoder(view2)
                loss = criterion(z1, z2)
                
                # Gradient scaling if using mixed precision (optional but recommended for H200)
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                loss.backward() # Standard precision
                optimizer.step() # Standard precision
                
                scheduler.step()
                update_target_network(online_encoder, target_encoder, CFG.EMA_DECAY)
                
                train_loss += loss.item(); num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            train_loss /= num_batches if num_batches > 0 else 1
            
            # --- Validation ---
            online_encoder.eval()
            val_loss = 0; num_val_batches = 0
            pbar_val = tqdm(val_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{CFG.PRETRAIN_EPOCHS} [Val]")
            with torch.no_grad():
                for X_batch, _ in pbar_val:
                    view1 = gpu_augment(X_batch); view2 = gpu_augment(X_batch)
                    z1 = online_encoder(view1)
                    z2 = target_encoder(view2)
                    loss = criterion(z1, z2)
                    val_loss += loss.item(); num_val_batches += 1
                    pbar_val.set_postfix(loss=f"{loss.item():.4f}")

            val_loss /= num_val_batches if num_val_batches > 0 else 1
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            
            print(f"Fold {fold+1} Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Duration: {epoch_duration:.2f}s")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best val loss. Saving model for fold {fold+1}...")
                torch.save(online_encoder.state_dict(), os.path.join(CFG.SAVE_PATH, f"jepa_backbone_fold{fold}.pth"))

        best_fold_losses.append(best_val_loss)
        fold_end_time = time.time()
        fold_duration = fold_end_time - start_fold_time
        print(f"Fold {fold+1} finished. Best Val Loss: {best_val_loss:.4f}, Duration: {fold_duration/60:.2f}min")
        
        del train_loader, val_loader, online_encoder, target_encoder, criterion, optimizer, scheduler
        gc.collect(); torch.cuda.empty_cache()
    
    print("\n" + "="*50); print("STAGE 1 PRE-TRAINING FINISHED"); print(f"Mean Best Val Loss: {np.mean(best_fold_losses):.4f}"); print("="*50)

# =============================================================================
# STEP 6: STAGE 2 - FINE-TUNING
# =============================================================================
def run_finetuning():
    print("\n" + "="*50); print("STARTING STAGE 2: FINE-TUNING"); print("="*50)
    
    ch1_train, ch1_val, ch2_train, ch2_val = fetch_finetune_loaders()
    
    # --- [ Challenge 1: Cross-Task ] ---
    print("\n--- Fine-tuning for Challenge 1: Cross-Task (Response Time) ---")
    backbones_ch1 = []
    for fold in range(CFG.N_SPLITS):
        backbone = EegMambaJEPA(
            d_model=CFG.D_MODEL, n_layer=CFG.N_LAYERS, n_channels=CFG.N_CHANNELS,
            patch_size=CFG.PATCH_SIZE, d_state=CFG.D_STATE, expand=CFG.EXPAND
        ).to(CFG.DEVICE)
        try:
            backbone.load_state_dict(torch.load(os.path.join(CFG.SAVE_PATH, f"jepa_backbone_fold{fold}.pth")))
            print(f"Loaded backbone fold {fold} for CH1"); backbones_ch1.append(backbone)
        except FileNotFoundError: print(f"WARNING: Could not find backbone fold {fold}. Skipping.")

    if not backbones_ch1: raise RuntimeError("FATAL: No CH1 backbones loaded.")
    for bb in backbones_ch1: bb.eval(); [p.requires_grad_(False) for p in bb.parameters()] # Freeze
    head_ch1 = nn.Linear(CFG.D_MODEL, 1).to(CFG.DEVICE)
    model_ch1 = EnsembleCompetitionModel(backbones_ch1, head_ch1)
    optimizer = optim.AdamW(model_ch1.head.parameters(), lr=CFG.FINETUNE_LR) # Only optimize head
    criterion = nn.MSELoss()
    best_val_loss_ch1 = float('inf')

    for epoch in range(CFG.FINETUNE_EPOCHS_CH1):
        epoch_start_time = time.time(); model_ch1.train() # Train head, backbones are frozen
        train_loss = 0; num_batches = 0
        pbar = tqdm(ch1_train, desc=f"CH1 Epoch {epoch+1}/{CFG.FINETUNE_EPOCHS_CH1} [Train]")
        for X, y in pbar:
            X, y = X.to(CFG.DEVICE), y.to(CFG.DEVICE); optimizer.zero_grad()
            y_pred = model_ch1(X); loss = criterion(y_pred, y)
            loss.backward(); optimizer.step()
            train_loss += loss.item(); num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        model_ch1.eval(); val_loss = 0; num_val_batches = 0
        with torch.no_grad():
            for X, y in ch1_val:
                X, y = X.to(CFG.DEVICE), y.to(CFG.DEVICE)
                y_pred = model_ch1(X); val_loss += criterion(y_pred, y).item(); num_val_batches += 1
        val_loss /= num_val_batches if num_val_batches > 0 else 1
        epoch_end_time = time.time(); epoch_duration = epoch_end_time - epoch_start_time
        print(f"CH1 Epoch {epoch+1}: Train Loss: {train_loss/num_batches:.4f}, Val Loss (MSE): {val_loss:.4f}, Duration: {epoch_duration:.2f}s")
        if val_loss < best_val_loss_ch1:
            best_val_loss_ch1 = val_loss
            print("New best CH1 val loss. Saving model..."); torch.save(model_ch1.state_dict(), os.path.join(CFG.SAVE_PATH, "finetuned_model_ch1.pth"))
    
    print(f"CH1 Fine-tuning finished. Best Val MSE: {best_val_loss_ch1:.4f}")
    del model_ch1, backbones_ch1, head_ch1, optimizer, criterion; gc.collect(); torch.cuda.empty_cache()

    # --- [ Challenge 2: Psychopathology ] ---
    print("\n--- Fine-tuning for Challenge 2: Psychopathology (Externalizing) ---")
    backbones_ch2 = []
    for fold in range(CFG.N_SPLITS):
         backbone = EegMambaJEPA(
            d_model=CFG.D_MODEL, n_layer=CFG.N_LAYERS, n_channels=CFG.N_CHANNELS,
            patch_size=CFG.PATCH_SIZE, d_state=CFG.D_STATE, expand=CFG.EXPAND
        ).to(CFG.DEVICE)
         try:
            backbone.load_state_dict(torch.load(os.path.join(CFG.SAVE_PATH, f"jepa_backbone_fold{fold}.pth")))
            print(f"Loaded backbone fold {fold} for CH2"); backbones_ch2.append(backbone)
         except FileNotFoundError: print(f"WARNING: Could not find backbone fold {fold}. Skipping.")

    if not backbones_ch2: raise RuntimeError("FATAL: No CH2 backbones loaded.")
    for bb in backbones_ch2: bb.eval(); [p.requires_grad_(False) for p in bb.parameters()] # Freeze
    head_ch2 = MDNHead(input_dim=CFG.D_MODEL, n_components=CFG.MDN_COMPONENTS).to(CFG.DEVICE)
    model_ch2 = EnsembleCompetitionModel(backbones_ch2, head_ch2)
    optimizer = optim.AdamW(model_ch2.head.parameters(), lr=CFG.FINETUNE_LR) # Only optimize head
    best_val_loss_ch2 = float('inf')

    for epoch in range(CFG.FINETUNE_EPOCHS_CH2):
        epoch_start_time = time.time(); model_ch2.train() # Train head, backbones are frozen
        train_loss = 0; num_batches = 0
        pbar = tqdm(ch2_train, desc=f"CH2 Epoch {epoch+1}/{CFG.FINETUNE_EPOCHS_CH2} [Train]")
        for X, y in pbar:
            X, y = X.to(CFG.DEVICE), y.to(CFG.DEVICE); optimizer.zero_grad()
            pi, sigma, mu = model_ch2(X); loss = mdn_loss(pi, sigma, mu, y)
            loss.backward(); optimizer.step()
            train_loss += loss.item(); num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        model_ch2.eval(); val_loss = 0; num_val_batches = 0
        with torch.no_grad():
            for X, y in ch2_val:
                X, y = X.to(CFG.DEVICE), y.to(CFG.DEVICE)
                pi, sigma, mu = model_ch2(X); val_loss += mdn_loss(pi, sigma, mu, y).item(); num_val_batches += 1
        val_loss /= num_val_batches if num_val_batches > 0 else 1
        epoch_end_time = time.time(); epoch_duration = epoch_end_time - epoch_start_time
        print(f"CH2 Epoch {epoch+1}: Train Loss: {train_loss/num_batches:.4f}, Val Loss (MDN): {val_loss:.4f}, Duration: {epoch_duration:.2f}s")
        if val_loss < best_val_loss_ch2:
            best_val_loss_ch2 = val_loss
            print("New best CH2 val loss. Saving model..."); torch.save(model_ch2.state_dict(), os.path.join(CFG.SAVE_PATH, "finetuned_model_ch2.pth"))

    print(f"CH2 Fine-tuning finished. Best Val MDN Loss: {best_val_loss_ch2:.4f}")
    del model_ch2, backbones_ch2, head_ch2, optimizer; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "="*50); print("STAGE 2 FINE-TUNING FINISHED"); print("="*50)

# =============================================================================
# STEP 7: MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    script_start_time = time.time()
    # Check if data path exists
    if not os.path.isdir(CFG.RAW_DATA_PATH):
        print(f"FATAL ERROR: Data path not found: {CFG.RAW_DATA_PATH}")
        print("Please update CFG.RAW_DATA_PATH to point to your dataset.")
    elif not torch.cuda.is_available():
        print("FATAL ERROR: CUDA is not available. This script requires a GPU.")
    else:
        print(f"Using data from: {CFG.RAW_DATA_PATH}")
        
        # Run Stage 1: Pre-training
        run_pretraining()
        
        # Run Stage 2: Fine-tuning
        run_finetuning()
        
        script_end_time = time.time()
        total_duration_minutes = (script_end_time - script_start_time) / 60
        
        print("\n\n" + "="*50)
        print("✅ [SUCCESS] ✅")
        print("Full Training Pipeline Finished.")
        print(f"Total script duration: {total_duration_minutes:.2f} minutes.")
        print("Your final models are saved as:")
        print(f" - {os.path.join(CFG.SAVE_PATH, 'finetuned_model_ch1.pth')}")
        print(f" - {os.path.join(CFG.SAVE_PATH, 'finetuned_model_ch2.pth')}")
        print("Your pre-trained backbones are saved as:")
        for f in range(CFG.N_SPLITS): print(f" - {os.path.join(CFG.SAVE_PATH, f'jepa_backbone_fold{f}.pth')}")
        print("\nYou can now proceed to package your submission.")
        print("="*50)