from eegdash.dataset import EEGChallengeDataset
from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.preprocessing import (
    preprocess, Preprocessor, create_fixed_length_windows,
    exponential_moving_standardize
)
import mne
from mne.io import read_raw_fif
from pathlib import Path
import shutil
import json
from typing import List
from tqdm import tqdm
from collections import defaultdict

import torch
import pandas as pd


def build_preprocessors(sfreq: float = 100.0):
    """Return a list of Preprocessor callables similar to the notebook.

    This is intentionally lightweight: the callables are compatible with
    braindecode.preprocessing.preprocess when available. Users can pass
    --no-preprocess to skip this step.
    """
    preprocessors = [
        Preprocessor("pick", picks="eeg"),
        Preprocessor("set_eeg_reference", ref_channels="average", ch_type="eeg"),
        # Preprocessor(
        #     exponential_moving_standardize,
        #     factor_new=1e-3,
        #     init_block_size=int(10 * sfreq),
        # ),
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
        try:
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
        except Exception as e:
            print(f"There is an error in processing task = {task} with release = {release}")
            continue
        
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



def _find_subject_id_in_folder(folder: Path) -> str:
    """
    Attempt to find a description.json in `folder` (or subfolders) and extract a subject id.
    Falls back to the folder name if no JSON subject id is found.
    """
    # Search for description.json or any .json that contains a subject-like field
    for jf in folder.rglob('description.json'):
        try:
            with open(jf, 'r') as f:
                jd = json.load(f)
            for key in ('subject', 'participant_id', 'participant', 'subject_id'):
                if key in jd:
                    return str(jd[key])
        except Exception:
            continue

    # Fallback: try any json file and look for keys
    for jf in folder.rglob('*.json'):
        try:
            with open(jf, 'r') as f:
                jd = json.load(f)
            for key in ('subject', 'participant_id', 'participant', 'subject_id'):
                if key in jd:
                    return str(jd[key])
        except Exception:
            continue

    # Last resort: folder name
    return folder.name


def _move_folder_to_subject(task_dir: Path, out_root: Path):
    """Move all subject subfolders under a task directory into per-subject folders under out_root.

    Example:
      task_dir = preproc_root / 'contrastChangeDetection'
      out_root = preproc_root  # we will create out_root/<subject_id>/<task>/...
    """
    if not task_dir.exists():
        return

    for child in [p for p in task_dir.iterdir() if p.is_dir()]:
        subject_id = _find_subject_id_in_folder(child)
        dest_dir = out_root / subject_id / task_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Move all files from child into dest_dir (flattening)
        for src in child.rglob('*'):
            if src.is_file():
                dst = dest_dir / src.name
                try:
                    if dst.exists():
                        # Overwrite existing file
                        dst.unlink()
                    shutil.move(str(src), str(dst))
                except Exception:
                    # Fallback to copy then remove
                    try:
                        shutil.copy2(str(src), str(dst))
                        src.unlink()
                    except Exception:
                        print(f"Failed to move {src} -> {dst}")

        # Remove emptied directories (if any)
        try:
            for p in sorted(child.rglob('*'), reverse=True):
                if p.is_dir() and not any(p.iterdir()):
                    p.rmdir()
            if not any(child.iterdir()):
                child.rmdir()
        except Exception:
            pass


def preprocess_all_releases(cache_dir: str, tasks: List[str], releases: List[str], out_dir: str,
                            mini: bool = False, download: bool = False, overwrite: bool = False,
                            sfreq: float = 100.0):
    """Preprocess multiple releases and consolidate results under single out_dir.

    For each release in `releases` this will load datasets, run preprocessing and then
    reorganize the resulting files so that under `out_dir` there will be per-subject
    directories named by the real subject id (e.g., NDARFT581ZW5) and within each subject
    a folder per task containing the preprocessed FIF files.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for rel in releases:
        print(f"Preprocessing release {rel} into temporary folder...")
        rel_out = out_root / rel
        rel_out.mkdir(parents=True, exist_ok=True)

        data_total = load_task_datasets(cache_dir=cache_dir, tasks=tasks, release=rel, mini=mini, download=download)
        preprocess_tasks(data_total, out_dir=rel_out, sfreq=sfreq, overwrite=overwrite)

        # Move preprocessed files from rel_out into per-subject folders
        # Reorganize all task folders into per-subject folders under out_root
        # Expect structure out_root/<task>/<numeric_subject>/* -> move into out_root/<subject_id>/<task>/*
        for task in tasks:
            task_dir = rel_out / task
            if task_dir.exists():
                _move_folder_to_subject(task_dir, out_root)

        # Clean up the temporary release folders
        try:
            if rel_out.exists():
                shutil.rmtree(rel_out)
        except Exception as e:
            print(f"Warning: failed to remove temporary folder {rel_out}: {e}")


    print(f"Preprocessing and consolidation complete. Consolidated preprocessed data at: {out_root}")


def create_fixed_windows_from_preprocessed(
    preproc_root: str,
    window_sec: float = 30.0,
    sfreq: float = 100.0,
    overlap_ratio: float = 0.5,
    preload: bool = False,
    min_samples: int = 200,
    required_channels: int = 129,
    cache_metadata: bool = True,
    rank: int = 0,
    is_master: bool = True
):
    preproc_root = Path(preproc_root)
    metadata_path = preproc_root / "windows_metadata.pt"
    index_path = preproc_root / "windows_index.pt"

    # === FAST PATH: Load from cache ===
    if cache_metadata and metadata_path.exists() and index_path.exists():
        if not is_master:
            print(f"[RANK {rank}] Loading cached window dataset...")
        else:
            print("[MASTER] Loading cached window dataset...")

        metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
        index_data = torch.load(index_path, map_location="cpu", weights_only=False)

        # Reconstruct BaseDatasets from file paths
        all_datasets = []
        for path, desc in tqdm(index_data["datasets"], desc="Loading cached datasets"):
            try:
                raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
                all_datasets.append(BaseDataset(raw, desc))
            except Exception as e:
                print(f"Failed to load cached file {path}: {e}")
                continue

        if not all_datasets:
            raise RuntimeError("No valid cached datasets loaded.")

        concat_ds = BaseConcatDataset(all_datasets)

        windows_ds = create_fixed_length_windows(
            concat_ds,
            start_offset_samples=0,
            stop_offset_samples=None,
            window_size_samples=metadata["window_size"],
            window_stride_samples=metadata["stride"],
            drop_last_window=True,
            preload=preload,
        )
        print(f"[RANK {rank}] Loaded {len(windows_ds)} windows from cache in <2s")
        return windows_ds

    # === SLOW PATH: Full creation (ONLY MASTER) ===
    if not is_master:
        raise RuntimeError("Only master should run full window creation.")

    print("[MASTER] Creating windows from scratch (this may take 10–30 min)...")
    raw_paths = sorted(preproc_root.rglob("*.fif"))
    print(f"Found {len(raw_paths)} preprocessed .fif files")

    all_preproc_datasets = []
    seen_counts = defaultdict(int)

    for raw_path in tqdm(raw_paths, desc="Scanning preprocessed files"):
        try:
            raw = mne.io.read_raw_fif(raw_path, preload=preload, verbose=False)
        except Exception as e:
            print(f"Failed to load {raw_path}: {e}")
            continue

        file_sfreq = float(raw.info.get("sfreq", sfreq) or sfreq)
        n_samples = raw.n_times
        if n_samples < min_samples:
            raw.close()
            continue

        eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False)
        if len(eeg_picks) != required_channels:
            raw.close()
            continue

        # === Subject/Task Inference (same as yours) ===
        try:
            rel = raw_path.relative_to(preproc_root)
            parts = rel.parts
        except:
            parts = raw_path.parts

        if len(parts) >= 3:
            subject, task = parts[0], parts[1]
        elif len(parts) == 2:
            task = parts[0]
            stem = raw_path.stem
            subject = stem.split("-", 1)[0] if "-" in stem else parts[0]
        else:
            subject = task = raw_path.parent.name

        seen_counts[(subject, task)] += 1
        count = seen_counts[(subject, task)]
        task_display = task if count == 1 else f"{task}-{count}"

        description = {
            "task": task_display,
            "task_raw": task,
            "subject": subject,
            "filename": raw_path.name,
            "filepath": str(raw_path),
        }

        try:
            all_preproc_datasets.append(BaseDataset(raw, description))
        except Exception as e:
            print(f"Failed to wrap {raw_path}: {e}")
            raw.close()

    if not all_preproc_datasets:
        raise RuntimeError("No valid preprocessed files found.")

    concat_ds = BaseConcatDataset(all_preproc_datasets)
    window_size_samples = int(window_sec * sfreq)
    stride_samples = int(window_size_samples * (1 - overlap_ratio))

    windows_ds = create_fixed_length_windows(
        concat_ds,
        window_size_samples=window_size_samples,
        window_stride_samples=stride_samples,
        drop_last_window=True,
        preload=preload,
    )

    # === SAVE CACHE ===
    if cache_metadata:
        print("[MASTER] Saving dataset cache for fast reload...")
        torch.save({
            "window_size": window_size_samples,
            "stride": stride_samples,
        }, metadata_path)

        # Save lightweight index: (filepath, description)
        index_data = {
            "datasets": [
                (str(ds.description["filepath"]), ds.description)
                for ds in all_preproc_datasets
            ]
        }
        torch.save(index_data, index_path)

    print(f"[MASTER] Created {len(windows_ds)} windows. Cache saved.")
    return windows_ds

