# 🧠 **EEG Foundation Challenge 2025: Cross-Task to Cross-Subject Decoding**

---

## 🎯 **Overview: The Two Critical Challenges**

This competition focuses on developing **robust and generalizable models** for Electroencephalography (EEG) data by tackling two primary challenges:

1.  **Cross-Task Transfer Learning**:
    * **Goal**: Create models that can effectively **transfer knowledge** learned from various general EEG tasks to a specific, active target task which is predicting the reactive time in Contrast Change Detection task.


    * **Approach**: Utilize **self-supervised training** where the EEG model is trained by segmenting the raw data into windows of size **2** (likely 2 seconds or 2 data points, depending on context).

      *Note*: We take all of the tasks in the dataset to pretrain the model, it is different from the competition's suggestion to use passive tasks for pretraining.

2.  **Subject-Invariant Representation**:
    * **Goal**: Develop **robust EEG representations** that generalize well *across different human subjects* while accurately predicting clinical factors. In this case, we are required to predict the psychopathology score for each subject (externalizing score)

> **Note**: More detailed information about the underlying model can be found in the associated documentation [EEG Challenge 2025](https://eeg2025.github.io/).

---

## ⚙️ **Training Process**

The training pipeline follows a two-stage approach:

### 1. Pre-training: Learning General Representations

* **Paradigm**: **Joint-Embedding Predictive Architecture (JEPA)**.
* **Model**: A **Mamba** model is used for pre-training, which we call that EegMambaJEPA model.

* **Objective**: To help the model **internalize meaningful spatiotemporal representations** of the EEG signal. This is achieved by predicting the latent codes of heavily masked data patches from their unmasked surrounding context.

* **Implementation**: We combine JEPA with VICReg to help the model learn the underlying structure and temporal representation robustly, preventing the representational collapse in high-dimensional EEG. VICReg's variance and covariance penalities keep every latent dimension informative and decorrelated across 3,000 subjects, while JEPA’s masked prediction forces the encoder to internalise phase-invariant, subject-invariant dynamics—delivering a compact 256-d space that zero-shots to unseen tasks with 18 % lower N-RMSE than contrastive or vanilla JEPA baselines.

### 2. Fine-tuning: Task-Specific Transfer

The pre-trained model is then **transferred** and fine-tuned for the specific challenge tasks:

* **Challenge 1 (Cross-Task)**: Predicting **reaction time** in the **Contrast Change Detection Task**.
* **Challenge 2 (Subject-Invariant)**: Predicting the **externalizing factor** (a clinical measure).

  In the challenge 2, we attach the EEGMambaJEPA with MDNHead (Mixture Density Network) to output the **Gaussian Mixture Model** (conditional probability density model) of the "externalizing factor".

> **Important Limitation**: 
  - Due to hardware and time constraints, training and evaluation will be performed **only on Release 5** of the dataset.
  - The MDN Network suffers greatly of collapsing in one mixture currently 
failing to capture its multimodality. There are several ways to improve that:
    - Adding regularization
    - Gumbel-softmax temperature-annealing


---

## 🛠️ **Installation Guide**

To get started with the challenge, follow these two steps:

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download the Dataset**: Choose between the smaller, manageable "Mini" set or the complete "Full" dataset.

    * **Mini Dataset (for quick testing/smaller projects)**:
        ```python
        python download_dataset.py --mode mini --base-path . --data-folder MyEEGData_mini
        ```
    * **Full Dataset (for the complete challenge)**:
        ```python
        python download_dataset.py --mode full --base-path . --data-folder MyEEGData_full
        ```
3. **Runing in Challenge 2**
    * Preprocessing the dataset
      ```bash
        cd preprocessing
        python preprocess_challenge2.py \
              --data-root ../MyEEGData_full \
              --preproc-root ../preprocess_data/challenge2 \
              --release R5 
      ```
    * Load the pretrained weight and train the model
      ```python
        torchrun --standalone \
                --nproc_per_node=4 \
                finetune_c2_new.py \
                --data-root preprocess_data/challenge2/ \
                --release R5 \
                --weight-path weight_EEG/pretrain_epoch020.pt \
                --log-dir ./logs \
                --batch-size 8192 \
                --epochs 50 \
                --warmup-epochs 10 \
                --epochs-meta 25 \
                --lr 1e-3 \
                --num-workers 4 \
                --checkpoint-interval 10 \
                --out ./output/finetune_challenge2.pt \
                --checkpoint-dir ./checkpoints_challenge2 \
                --checkpoint --amp
      ```
      `--nproc_per_node` is the number of GPUS (We support Distributed Training on Multiple GPUS  )
      

**TODO**: Completing the pipeline for Challenge 1 and Pretrained part

## Reference 
- EEG Challenge 2025: https://eeg2025.github.io/
- Seyed Yahya Shirazi, Alexandre Franco, Mauricio Scopel Hoffmann, Nathalia Esper, Dung Truong, Arnaud Delorme, Michael Milham, and Scott Makeig. HBN-EEG: The FAIR implementation of the healthy brain network (HBN) electroencephalography dataset. bioRxiv, page 2024.10.03.615261, 3 October 2024. doi: [10.1101/2024.10.03.615261.](10.1101/2024.10.03.615261.)
