import sys
sys.path.append("../")  # To ensure parent directory is in path

from pathlib import Path
import argparse
import time
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from sklearn.utils import check_random_state
from tqdm import tqdm

import joblib
import math

try:
    # Attempt to import eegdash components
    from eegdash.dataset import EEGChallengeDataset
except Exception as e:
    EEGChallengeDataset = None
    print("Warning: eegdash package not found. Data loading will fail.")

from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset, BaseDataset

from model.meta_encoder import MetaEncoder # Assuming this is available or defined

# --- Constants from Original Script ---
SFREQ = 100
CROP_SEC = 2 # 2 seconds * 100 Hz = 200 samples
WINDOW_SEC = 4 # 4 seconds * 100 Hz = 400 samples
STRIDE_SEC = 2 # 2 seconds * 100 Hz = 200 samples
DESCRIPTION_FILEDS = [
    "subject", "session", "run", "task", "age", "gender", "sex", "p_factor", "externalizing" # Added 'externalizing' for clarity
]
TASK_NAMES = [
    "RestingState", "DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals",
    "ThePresent", "contrastChangeDetection", "seqLearning6target",
    "seqLearning8target", "surroundSupp", "symbolSearch"
]
# ------------------------------------


class CropMetaWrapper(BaseDataset):
    """
    Wraps an EEGWindowsDataset to:
    1. Extract the target variable ("externalizing" score).
    2. Encode the metadata (task, sex, age) using a pre-fitted MetaEncoder.
    3. Perform a random 2-second crop from the 4-second window.
    """
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
        # The underlying braindecode window dataset returns (X, y, crop_inds)
        # where X is the 4s window, y is not used here, and crop_inds is (i_win, i_start, i_stop)
        X, _, crop_inds = self.windows_ds[idx]  # X: (C, 4*SFREQ)

        # Target (Externalizing score)
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

        # Ensure the window is large enough for the crop
        assert i_stop - i_start >= self.crop_samples

        # FIXED: .integers instead of .randint
        offset = self.rng.integers(0, i_stop - i_start - self.crop_samples + 1)
        i_start = i_start + offset
        i_stop = i_start + self.crop_samples
        X_crop = X[:, offset : offset + self.crop_samples]  # (C, 2*SFREQ)

        # Infos (kept for compatibility/debugging, though not strictly needed for dump)
        infos = {
            "subject": desc["subject"],
            "session": desc.get("session", ""),
            "run": desc.get("run", ""),
            "task": desc["task"],
            "sex": desc["sex"],
            "age": float(desc["age"]),
            "externalizing": target,
        }

        # NOTE: For the purpose of dumping, we only need the dataset object itself,
        # not the item getter, but the class is required for window wrapping.
        # We return the original item structure for completeness, but it's unused here.
        return torch.tensor(X_crop), meta_vec, target, (i_win, i_start, i_stop), infos

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing and Windowing for Challenge 2")
    parser.add_argument("--data-root", type=str,
                        default="MyNeurIPSData/MyNeurIPSData/HBN_DATA_FULL",
                        help="Root dataset folder (raw cache).")
    parser.add_argument("--preproc-root", type=str,
                        default=None,
                        help="Path to save the preprocessed window files.")
    parser.add_argument("--release", nargs='+', default=[f"R{i}" for i in range(1, 12)], help="Releases to use (e.g., R1 R5). Default: R1..R11")
    parser.add_argument("--task", nargs='+', default=TASK_NAMES, help="Tasks to use (e.g., rest flanker). Default: all tasks")
    parser.add_argument("--mini", action="store_true", help="Use mini dataset mode for debugging")
    parser.add_argument("--download", action="store_true", help="Download dataset if not found")
    # Removed --use-preprocessed as this script is designed to CREATE them.

    return parser.parse_args()


def preprocess_and_window():
    args = parse_args()
    data_root = Path(args.data_root)

    if EEGChallengeDataset is None:
        raise RuntimeError("The package `eegdash` is required for this script. Install it in your environment.")

    print(f"Using data root: {data_root}")

    # 1) Load the per-task EEGChallengeDataset
    releases = args.release if isinstance(args.release, (list, tuple)) else [args.release]
    print(f"Loading EEGChallengeDataset task={args.task}, releases={releases}")

    all_raw_datasets = []
    meta_for_encoder = []

    for rel in releases:
        try: 
            print(f"  - Loading release {rel}")
            cache_dir = Path(data_root) / f"{rel}_mini_L100_bdf" if args.mini else Path(data_root) / f"{rel}_L100_bdf"

            for task in args.task:
                print(f"    - Loading task {task}")
                raw_ds = EEGChallengeDataset(
                    task=task, 
                    release=rel, 
                    cache_dir=cache_dir, 
                    mini=args.mini,
                    download = args.download,
                    description_fields = DESCRIPTION_FILEDS
                )
                all_raw_datasets.append(raw_ds)

                # Collect metadata for MetaEncoder fitting
                for sub_ds in raw_ds.datasets:
                    desc = sub_ds.description
                    meta_for_encoder.append({
                        "task": desc["task"],
                        "sex": desc["sex"],
                        "age": float(desc["age"]),
                    })
        except Exception as e:
            print(f"Warning: failed to load release {rel}: {e}")

    if len(all_raw_datasets) == 0:
        raise RuntimeError(f"No recordings found for task={args.task} in releases={releases}")

    # 2) Fit the MetaEncoder on the gathered metadata
    meta_encoder = MetaEncoder().fit(meta_for_encoder)
    META_DIM = meta_encoder.dim
    meta_encoder_path = args.preproc_root + "/meta_encoder.pkl" if args.preproc_root else data_root / 'preprocessed' / "meta_encoder.pkl"
    joblib.dump(meta_encoder, meta_encoder_path)
    print(f"Meta encoder fitted. Dimension: {META_DIM}")


    # 3) Setup output directory
    preproc_root = Path(args.preproc_root) if args.preproc_root else data_root / 'preprocessed'
    preproc_root.mkdir(parents=True, exist_ok=True)
    print(f"Preprocessed windows will be saved to: {preproc_root}")

    # 4) Process each loaded raw dataset (per-release, per-task)
    list_windows_to_save = []
    
    for ds in tqdm(all_raw_datasets, desc = "Windowing and Wrapping"):
        # The original script pools all recordings for the same release before windowing.
        # However, for saving modularly, we process per loaded ds_rel object (which is task/release specific)
        print(f"\nProcessing release {ds.release}...")
        # Original dataset size is
        print(f"  -> Original dataset size: {len(ds.datasets)} recordings.")

        # Filter recordings: minimum length, channel count, and non-NaN target
        filtered_datasets = [
            sub_ds for sub_ds in ds.datasets
            if (sub_ds.raw.n_times >= WINDOW_SEC * SFREQ # Must be at least 4s long
                and len(sub_ds.raw.ch_names) == 129 # Must have 129 channels
                and not math.isnan(sub_ds.description.get("externalizing", math.nan))) # Target must be present
        ]
        task = ds.description['task'].unique().item() if 'task' in ds.description else 'unknown'
        if not filtered_datasets:
            print(f"Skipping {ds.release}/{task}: No suitable recordings found after filtering.")
            continue

        filtered = BaseConcatDataset(filtered_datasets)
        print(f"  -> {ds.release}/{task}: {len(filtered)} recordings passed filters.")

        # Create fixed-length windows (4s windows, 2s stride)
        windows = create_fixed_length_windows(
            filtered,
            window_size_samples= WINDOW_SEC * SFREQ,
            window_stride_samples= STRIDE_SEC * SFREQ,
            drop_last_window=True,
        )

        # Wrap each window dataset with the metadata/cropping logic
        windows_ds = BaseConcatDataset(
            [CropMetaWrapper(
                sub_windows_ds,
                crop_samples=CROP_SEC * SFREQ,
                meta_encoder=meta_encoder # Pass the fitted encoder
            ) for sub_windows_ds in windows.datasets
            ]
        )
        print(f"  -> {ds.release}/{task}: Created {len(windows_ds)} total windows.")

        # 5) Save the resulting windows dataset
        save_directory = preproc_root / f"{ds.release}_windows_task[{task}].pkl"

        try:
            joblib.dump(windows_ds, save_directory)
            print(f"  -> Successfully saved windows to {save_directory}")
            list_windows_to_save.append(windows_ds)
        except Exception as e:
            print(f"  -> FAILED to save windows for {ds.release}/{task}: {e}")

    print("\n✅ Preprocessing and windowing complete.")
    print(f"Total datasets saved: {len(list_windows_to_save)}")
    print(f"MetaEncoder object must be saved separately if needed for submission.")


if __name__ == '__main__':
    preprocess_and_window()