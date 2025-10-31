import os
import numpy as np
import torch
from pathlib import Path
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

import mne
from mne.io import read_raw_fif

from eegdash.dataset.dataset import EEGChallengeDataset

from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_fixed_length_windows,
    exponential_moving_standardize
)

import torch.nn as nn
import torch.optim as optim
import argparse

from model.eegmamba_jamba import EegMambaJEPA
from model.loss import VICReg
from model.mdn import MDNHead

# Add convenience imports used in the notebook-style pipeline
from braindecode.preprocessing import Preprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Pretraining dataset loader and utility for JEPA-style EEG pipeline")

    # Data and preprocessing
    parser.add_argument("--data-root", type=str, default="LOL_DATASET/HBN_DATA_FULL", help="Root directory where dataset will be cached/loaded")
    parser.add_argument("--release", type=str, default="R1", help="Release to load (e.g., R1, R5)")
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS, help="List of task names to load")
    parser.add_argument("--mini", action="store_true", help="Use mini dataset mode (faster for debugging)")
    parser.add_argument("--download", action="store_true", help="Allow dataset download if not present")
    parser.add_argument("--preprocess", action="store_true", help="Run braindecode preprocessing step and save results")
    parser.add_argument("--preproc-out", type=str, default="preprocessed", help="Directory to save preprocessed outputs (under data-root)")
    parser.add_argument("--window-sec", type=float, default=2.0, help="Window length in seconds for fixed-length windows")
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap ratio")

    # Model / training (lightweight set to mirror JEMA defaults)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


DEFAULT_TASKS = [
    "RestingState", "DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals",
    "ThePresent", "contrastChangeDetection", "seqLearning6target",
    "seqLearning8target", "surroundSupp", "symbolSearch"
]


def build_preprocessors(sfreq: float = 100.0):
    """Return a list of Preprocessor callables similar to the notebook.

    This is intentionally lightweight: the callables are compatible with
    braindecode.preprocessing.preprocess when available. Users can pass
    --no-preprocess to skip this step.
    """
    preprocessors = [
        Preprocessor("pick", picks="eeg"),
        Preprocessor("set_eeg_reference", ref_channels="average", ch_type="eeg"),
        Preprocessor(
            exponential_moving_standardize,
            factor_new=1e-3,
            init_block_size=int(10 * sfreq),
        ),
    ]
    return preprocessors


def load_task_datasets(cache_dir: str, tasks: list, 
                       release: str = "R1", mini: bool = False, download: bool = False):
    """Load per-task EEGChallengeDataset objects (one per task).

    This mirrors the loop in the notebook: it returns a dict mapping task -> EEGChallengeDataset.
    The datasets themselves contain recordings; further preprocessing / windowing is done separately.
    """
    data_total = {}
    cache_dir = Path(cache_dir) / f"{release}_mini_L100_bdf" if mini else Path(cache_dir) / f"{release}_L100_bdf"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        print(f"Loading task: {task} (release={release}, mini={mini})")
        ds = EEGChallengeDataset(
            cache_dir=str(cache_dir),
            mini=mini,
            task=task,
            download=download,
            release=release,
            n_jobs=-1,
        )
        data_total[task] = ds
    return data_total


def preprocess_tasks(data_total: dict, out_dir: str, sfreq: float = 100.0, overwrite: bool = False):
    """Apply braindecode preprocessing and save preprocessed files per-task.

    This mirrors the notebook steps: it will call braindecode.preprocessing.preprocess
    and write out preprocessed FIF files under out_dir/<task>.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocessors = build_preprocessors(sfreq=sfreq)

    for task, ds in data_total.items():
        task_dir = out_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        print(f"Preprocessing task {task} -> {task_dir} (n_subjects={len(ds.datasets)})")

        # Load raw files if not preloaded (keeps behavior similar to notebook)
        for sub_ds in ds.datasets:
            try:
                if not getattr(sub_ds.raw, "preload", False):
                    sub_ds.raw.load_data()
            except Exception:
                # Some dataset implementations might not provide raw or preload attribute.
                pass

        # Call braindecode preprocessing (may be heavy) - give user control via CLI
        try:
            preproc = preprocess(ds, preprocessors, save_dir=task_dir, n_jobs=-1, overwrite=overwrite)
            print(f"Saved preprocessed files for {task} (found {len(list(task_dir.rglob('*-raw.fif')))} files)")
        except Exception as e:
            print(f"Warning: preprocessing failed for {task}: {e}")


def create_fixed_windows_from_preprocessed(preproc_root: str, window_sec: float = 30.0, sfreq: float = 100.0, overlap_ratio: float = 0.5):
    """Search preprocessed FIF files under preproc_root and create fixed-length windows dataset.

    Returns a braindecode BaseConcatDataset containing all windows.
    """
    preproc_root = Path(preproc_root)
    raw_paths = sorted(preproc_root.rglob("*-raw.fif"))
    print(f"Found {len(raw_paths)} preprocessed raw files under {preproc_root}")
    all_preproc_datasets = []
    for raw_path in raw_paths:
        try:
            raw = read_raw_fif(raw_path, preload=True, verbose=False)
            description = {"task": raw_path.parent.name, "subject": raw_path.parent.name, "filename": raw_path.name}
            all_preproc_datasets.append(BaseDataset(raw, description))
        except Exception as e:
            print(f"Failed to load {raw_path}: {e}")

    if not all_preproc_datasets:
        raise RuntimeError("No preprocessed files found; run preprocessing or point to the correct directory")

    concat_preproc = BaseConcatDataset(all_preproc_datasets)

    window_size_samples = int(window_sec * sfreq)
    window_stride_samples = int(window_size_samples * (1 - overlap_ratio))

    windows_ds = create_fixed_length_windows(
        concat_preproc,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=False,
        preload=True,
    )

    print(f"Created windows dataset with {len(windows_ds)} windows (window_sec={window_sec}, overlap={overlap_ratio})")
    return windows_ds


def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    data_root = Path(args.data_root)
    preproc_root = data_root / args.preproc_out

    # 1) Load per-task datasets (lightweight: does not preprocess immediately)
    data_total = load_task_datasets(cache_dir=data_root, tasks=args.tasks, release=args.release, mini=args.mini, download=args.download)

    # 2) Optionally preprocess and save preprocessed FIF files
    if args.preprocess:
        preprocess_tasks(data_total, out_dir=preproc_root, sfreq=100.0, overwrite=False)

    # 3) Create fixed-length windows from preprocessed files (if preprocessing was run), otherwise try to create from raw caches
    try:
        windows_ds = create_fixed_windows_from_preprocessed(preproc_root, window_sec=args.window_sec, sfreq=100.0, overlap_ratio=args.overlap)
        print(f"Windows dataset ready: {len(windows_ds)} examples")
    except Exception as e:
        print(f"Could not create windows from preprocessed files: {e}")
        print("Attempting to create fixed windows directly from loaded datasets (this may be slower and memory heavy)")
        # Fallback: concatenate raw datasets and create windows directly (like notebook before saving)
        all_preproc_datasets = []
        for task, ds in data_total.items():
            for sub in ds.datasets:
                try:
                    if not getattr(sub.raw, "preload", False):
                        sub.raw.load_data()
                    all_preproc_datasets.append(BaseDataset(sub.raw, sub.description))
                except Exception as exc:
                    print(f"Skipping subject due to error: {exc}")

        if not all_preproc_datasets:
            print("No raw datasets found to create windows from. Exiting.")
            return

        concat_preproc = BaseConcatDataset(all_preproc_datasets)
        windows_ds = create_fixed_length_windows(
            concat_preproc,
            start_offset_samples=0,
            stop_offset_samples=None,
            window_size_samples=int(args.window_sec * 100.0),
            window_stride_samples=int(args.window_sec * 100.0 * (1 - args.overlap)),
            drop_last_window=False,
            preload=True,
        )
        print(f"Created windows dataset directly from raws: {len(windows_ds)} windows")

    # Print a small summary and exit
    print("Summary:")
    print(f" - Data root: {data_root}")
    print(f" - Tasks loaded: {list(data_total.keys())}")
    print(f" - Window length (s): {args.window_sec}, overlap: {args.overlap}")
    print("Script completed. You can now plug this data pipeline into your training loop.")


if __name__ == '__main__':
    main()


