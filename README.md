# EEG Challenge 2025 

# EEG Challenge 2025

This repository implements Mamba / JEPA-style pretraining and downstream fine-tuning for:
- Challenge 1 (Contrast Change Detection) — regression of response time
- Challenge 2 (Externalizing factor) — MDN-based regression

Quick links
- Pretraining script: [pretrain.py](pretrain.py) — see [`pretrain.main`](pretrain.py)  
- Fine-tune (Challenge 1): [finetune_challenge1.py](finetune_challenge1.py) — uses [`finetune_challenge1.FinetuneJEPA`](finetune_challenge1.py)  
- Fine-tune (Challenge 2): [finetune_challenge2.py](finetune_challenge2.py) (single-GPU) and [finetune_c2_new.py](finetune_c2_new.py) (DDP / multi-GPU) — see [`finetune_challenge2.main`](finetune_challenge2.py) and [`finetune_c2_new.main`](finetune_c2_new.py)  
- Model code and heads: [model/eegmamba_jamba.py](model/eegmamba_jamba.py) (`[`model.EegMambaJEPA`](model/eegmamba_jamba.py)`, `[`model.FinetuneJEPA_Challenge2`](model/eegmamba_jamba.py)`) and [model/mdn.py](model/mdn.py) (`[`model.MDNHead`](model/mdn.py)`)  
- Preprocessing helper: [preprocessing/preprocess_challenge2.py](preprocessing/preprocess_challenge2.py) (`[`preprocessing.preprocess_and_window`](preprocessing/preprocess_challenge2.py)`)  
- Utilities: [utils.py](utils.py) (dataset split & collate)  
- Example submission loader: [submission_longdang.py](submission_longdang.py)  
- Example full training orchestrator: [JEMA-EEG25-Full-Training.py](JEMA-EEG25-Full-Training.py)  
- Useful commands and examples: [commandline.txt](commandline.txt)  
- Notebooks / logs: [experiment_abc.ipynb](experiment_abc.ipynb), [creating_weight.ipynb](creating_weight.ipynb), logs/ and output/

Overview

1. Pretraining (self-supervised JEPA / Mamba)
   - Script: [pretrain.py](pretrain.py) (`[`pretrain.main`](pretrain.py)`)  
   - Goal: train a JEPA / Mamba backbone (encoder) over many EEG releases/tasks to produce transferable backbone weights.  
   - Backbone: [`model.EegMambaJEPA`](model/eegmamba_jamba.py).  
   - Head(s): VICReg / MDN components used depending on config (see [pretrain.py](pretrain.py) and [model/mdn.py](model/mdn.py)).  
   - Output: checkpoint files saved to checkpoint dir (pretrain_epoch*.pt). Use these weights for fine-tuning.

2. Preprocessing & windowing
   - Script: [preprocessing/preprocess_challenge2.py](preprocessing/preprocess_challenge2.py) (`[`preprocessing.preprocess_and_window`](preprocessing/preprocess_challenge2.py)`)  
   - Purpose: convert raw EEG caches (EEGChallengeDataset) into fixed-length windows, fit a MetaEncoder (task/age/sex → vector), and save per-release window pickles.  
   - The produced files (e.g., `R5_windows_task[RestingState].pkl`) are consumed by fine-tuning scripts. See [preprocessing/run_preprocess.txt](preprocessing/run_preprocess.txt).

3. Fine-tuning — Challenge 1 (CCD)
   - Script: [finetune_challenge1.py](finetune_challenge1.py) (`[`finetune_challenge1.FinetuneJEPA`](finetune_challenge1.py)`)  
   - Setup: load backbone weights (from pretraining), attach a simple regression head, split subjects (train/val/test), run MSE-based training.  
   - Typical invocation examples: see [commandline.txt](commandline.txt) — adjust `--weight-path`, `--preproc-root`, batch size, epochs.  
   - Output: best fine-tuned state saved (out path). The submission helper [submission_longdang.py](submission_longdang.py) shows how to load a saved Challenge 1 model.

4. Fine-tuning — Challenge 2 (Externalizing factor)
   - Single-GPU script: [finetune_challenge2.py](finetune_challenge2.py) — supports MDN training with meta-information.  
   - Multi-GPU / DDP: [finetune_c2_new.py](finetune_c2_new.py) — distributed training, uses `torchrun` examples in [commandline.txt](commandline.txt).  
   - Model wrapper: [`model.FinetuneJEPA_Challenge2`](model/eegmamba_jamba.py) implements:
     - MDN heads for training (`[`model.MDNHead`](model/mdn.py)`) and submission-time prediction.
     - Mode switching: `train_mode()`, `eval_mode()`, `submit_mode()` — used by training & evaluation routines.
   - Loss: MDN negative log-likelihood (see [`model.mdn`](model/mdn.py) / [`model.loss.mdn_loss`](model/loss.py) where applicable).
   - Fine-tune flow (DDP script): preprocessing → load meta encoder → create CropMetaWrapper datasets (random 2s crops) → DDP dataloaders (`DistributedSampler`) → training loop with per-epoch validation and checkpointing. See [`finetune_c2_new.main`](finetune_c2_new.py).

Quickstart (local / single-GPU)
- Preprocess (if needed):
  - python preprocessing/preprocess_challenge2.py --data-root <RAW_ROOT> --preproc-root <OUT_DIR> --release R5
  - (See [preprocessing/run_preprocess.txt](preprocessing/run_preprocess.txt))
- Pretrain (example):
  - python pretrain.py --data-root <DATA_ROOT> --release R1 R2 ... --epochs 50 --batch-size 256 --checkpoint-dir checkpoints/
- Fine-tune Challenge 1:
  - python finetune_challenge1.py --data-root <PREPROC_ROOT> --preproc-root <PREPROC_ROOT> --weight-path checkpoints/pretrain_epoch020.pt ...
- Fine-tune Challenge 2 (single GPU):
  - python finetune_challenge2.py --data-root <PREPROC_ROOT> --weight-path checkpoints/pretrain_epoch020.pt ...
- Fine-tune Challenge 2 (DDP / multi-GPU):
  - torchrun --standalone --nproc_per_node=4 finetune_c2_new.py --data-root preprocess_data/challenge2/ --weight-path weight_EEG/pretrain_epoch020.pt ...  
  - Example in [commandline.txt](commandline.txt)

Useful files and symbols
- [pretrain.py](pretrain.py) (`[`pretrain.main`](pretrain.py)`) — pretraining pipeline  
- [preprocessing/preprocess_challenge2.py](preprocessing/preprocess_challenge2.py) (`[`preprocessing.preprocess_and_window`](preprocessing/preprocess_challenge2.py)`) — windowing & meta encoder  
- [finetune_challenge1.py](finetune_challenge1.py) (`[`finetune_challenge1.FinetuneJEPA`](finetune_challenge1.py)`) — Challenge 1 finetune  
- [finetune_challenge2.py](finetune_challenge2.py) (`[`finetune_challenge2.main`](finetune_challenge2.py)`) — Challenge 2 single-GPU  
- [finetune_c2_new.py](finetune_c2_new.py) (`[`finetune_c2_new.main`](finetune_c2_new.py)`) — Challenge 2 DDP multi-GPU  
- [model/eegmamba_jamba.py](model/eegmamba_jamba.py) (`[`model.EegMambaJEPA`](model/eegmamba_jamba.py)`, `[`model.FinetuneJEPA_Challenge2`](model/eegmamba_jamba.py)`) — architectures  
- [model/mdn.py](model/mdn.py) (`[`model.MDNHead`](model/mdn.py)`) — MDN head implementation  
- [utils.py](utils.py) — dataset splitting & collate helpers used in DDP script  
- [submission_longdang.py](submission_longdang.py) — inference / submission helper showing model loading

Repro & logging
- TensorBoard logs are saved under logs/ (see examples in logs/).  
- Checkpoints saved under checkpoint directories specified by script flags. See examples in [commandline.txt](commandline.txt).

Notes & tips
- Pretrained backbone weights from pretraining are used to initialize fine-tuning backbones (loaded leniently with strict=False so heads can differ). See how weight loading is performed in [finetune_challenge2.py](finetune_challenge2.py) and [finetune_c2_new.py](finetune_c2_new.py).
- Challenge 2 uses metadata (task/age/sex) encoded by a MetaEncoder fit during preprocessing. The encoder is saved with the preprocessed outputs and loaded by the fine-tune scripts.
- For DDP runs, ensure CUDA + NCCL available and launch with `torchrun` (examples in [commandline.txt](commandline.txt)).

Contact / further exploration
- Inspect example experiments and EDA in [experiment_abc.ipynb](experiment_abc.ipynb) and [creating_weight.ipynb](creating_weight.ipynb).  
- For a one-file orchestrator, see [JEMA-EEG25-Full-Training.py](JEMA-EEG25-Full-Training.py).

License / citation
- (Add project license and citation here.)
