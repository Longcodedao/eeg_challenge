import os
import json
import time
from pathlib import Path
import argparse

import torch
import numpy as np

from build_dataset import (
    create_fixed_windows_from_preprocessed,
    preprocess_all_releases
)

SFREQ = 100.0
DEFAULT_TASKS = [
    "RestingState", "DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals",
    "ThePresent", "contrastChangeDetection", "seqLearning6target",
    "seqLearning8target", "surroundSupp", "symbolSearch"
]


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess EEG releases and create fixed-length windows (single-GPU).")
    p.add_argument("--data-root", type=str, required=True, help="Root directory where raw / cached data lives")
    p.add_argument("--releases", nargs="+", required=True, help="Releases to preprocess (e.g., R1 R2)")
    p.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS, help="List of task names to load")
    p.add_argument("--mini", action="store_true", help="Use mini dataset mode")
    p.add_argument("--download", action="store_true", help="Allow download if missing")
    p.add_argument("--preproc-out", type=str, default="preprocessed", help="Directory under data-root for preprocessed outputs")
    p.add_argument("--window-sec", type=float, default=2.0, help="Window length (seconds)")
    p.add_argument("--overlap", type=float, default=0.5, help="Window overlap ratio")
    p.add_argument("--preload", action="store_true", help="Whether to preload windows into memory (may be large)")
    p.add_argument("--sfreq", type=float, default=SFREQ, help="Target sampling frequency used for preprocessing")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing preprocessed files")
    p.add_argument("--save-info", type=str, default="windows_info.json", help="Filename (under preproc-out) to write windows metadata")
    p.add_argument("--no-cuda", action="store_true", help="Run on CPU even if CUDA is available")
    return p.parse_args()


def main():
    args = parse_args()
    # Force single-process/single-GPU behavior
    use_cuda = torch.cuda.is_available() and (not args.no_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        # Ensure single GPU (device 0)
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass

    data_root = Path(args.data_root)
    preproc_root = data_root / args.preproc_out
    preproc_root.mkdir(parents=True, exist_ok=True)

    releases = args.releases

    print(f"[preprocess_data] Running on device: {device}, targets: {releases}, mini={args.mini}")
    t0 = time.time()

    # 1) Preprocess releases (single-process)
    try:
        print("[preprocess_data] Starting preprocess_all_releases() ... (this can take a long time)")
        preprocess_all_releases(
            cache_dir=str(args.data_root),
            tasks=args.tasks,
            releases=releases,
            out_dir=str(preproc_root),
            mini=args.mini,
            download=args.download,
            sfreq=args.sfreq,
            overwrite=args.overwrite,
        )
        print(f"[preprocess_data] Preprocessing done in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"[preprocess_data] ERROR during preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()
