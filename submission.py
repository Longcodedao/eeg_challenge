# ##########################################################################
# # EEG Foundation Challenge 2025 - Submission File
# # JEMA-EEG25 (Mamba + JEPA) Implementation (Single H200 Version)
# ##########################################################################
import torch
import torch.nn as nn # Need nn for Linear head
import sys
import os
from pathlib import Path

# --- Step 1: Add your vendored Mamba library ---
# This line adds your "vendor" folder (built with Python 3.10 and torch 2.2.2)
# to the Python path so the server can find "mamba_ssm"
vendor_path = os.path.join(os.path.dirname(__file__), 'vendor')
sys.path.append(vendor_path)

# --- Step 2: Import your custom model definitions ---
# This imports the classes from your "my_model.py" file
try:
    from my_model import EegMambaJEPA, EnsembleCompetitionModel, MDNHead
    print("Successfully imported model definitions from my_model.py")
except ImportError as e:
    print(f"FATAL ERROR: Could not import model definitions from my_model.py: {e}")
    print("Ensure my_model.py is included in the submission zip at the root level.")
    raise

# --- Step 3: Use the official resolve_path function ---
# This function helps find your saved model weights within the Codabench environment
def resolve_path(name="model_file_name"):
    # Check common locations where Codabench might place resource files
    if Path(f"/app/input/res/{name}").exists():
        return f"/app/input/res/{name}"
    elif Path(f"/app/input/{name}").exists():
        return f"/app/input/{name}"
    # Check relative paths if run locally or if files are directly in root
    elif Path(f"{name}").exists():
        return f"{name}"
    elif Path(__file__).parent.joinpath(f"{name}").exists():
        return str(Path(__file__).parent.joinpath(f"{name}"))
    else:
        # If the file isn't found, raise a clear error
        raise FileNotFoundError(
            f"Could not find required model weight file '{name}'. Searched in "
            f"/app/input/res/, /app/input/, and the current directory "
            f"({Path(__file__).parent}). Ensure the weight file is included in your "
            f"submission zip at the root level and matches this name."
        )

# --- Step 4: Implement the required Submission class ---
class Submission:
    def __init__(self, SFREQ, DEVICE):
        """
        Initialize the Submission class.
        Args:
            SFREQ (int): Sampling frequency provided by the competition environment (should be 100).
            DEVICE (torch.device): The device (CPU or CUDA) provided by the environment.
        """
        self.sfreq = SFREQ
        self.device = DEVICE
        print(f"Submission Initialized. SFREQ={self.sfreq}, Device={self.device}")

        # --- Define Model Hyperparameters (Must match training CFG) ---
        # These parameters are used to instantiate the model architecture correctly.
        # They MUST EXACTLY match the parameters used in JEMA-EEG25-Full-Training-H200.py
        self.D_MODEL = 256
        self.N_LAYERS = 8
        self.N_CHANNELS = 129 # Should match competition data
        self.PATCH_SIZE = 10
        self.D_STATE = 16
        self.EXPAND = 2
        self.MDN_COMPONENTS = 5 # For Challenge 2 head

        # --- Expected Weight File Names ---
        # These names MUST match the files saved by your training script and included in the zip
        self.WEIGHTS_BACKBONE_FOLD_PREFIX = "jepa_backbone_fold" # e.g., jepa_backbone_fold0.pth
        self.WEIGHTS_CH1_FINETUNED = "finetuned_model_ch1.pth"
        self.WEIGHTS_CH2_FINETUNED = "finetuned_model_ch2.pth"
        self.N_FOLDS_EXPECTED = 5 # Should match N_SPLITS in training CFG

    def get_model_challenge_1(self):
        """
        Instantiates and loads the fine-tuned model for Challenge 1.
        Returns:
            torch.nn.Module: The loaded model in evaluation mode.
        """
        print("--- Loading Model for Challenge 1 ---")
        try:
            # 1. Instantiate Backbones (using parameters defined above)
            backbones_ch1 = []
            print(f"Attempting to load {self.N_FOLDS_EXPECTED} backbone folds...")
            for fold in range(self.N_FOLDS_EXPECTED):
                backbone = EegMambaJEPA(
                    d_model=self.D_MODEL, n_layer=self.N_LAYERS, n_channels=self.N_CHANNELS,
                    patch_size=self.PATCH_SIZE, d_state=self.D_STATE, expand=self.EXPAND
                )
                # Load weights for this fold's backbone (only needed if NOT loading the full finetuned model)
                # weight_file = f"{self.WEIGHTS_BACKBONE_FOLD_PREFIX}{fold}.pth"
                # backbone_weights_path = resolve_path(weight_file)
                # backbone.load_state_dict(torch.load(backbone_weights_path, map_location=self.device))
                # print(f"  Loaded backbone weights from: {weight_file}")
                backbones_ch1.append(backbone.to(self.device)) # Move to device

            # 2. Instantiate Head (Simple Linear Regression for Response Time)
            # The output dimension MUST be 1 for response time prediction.
            head_ch1 = nn.Linear(self.D_MODEL, 1).to(self.device)

            # 3. Combine into Ensemble Model
            model_challenge1 = EnsembleCompetitionModel(
                backbones=backbones_ch1, # Pass the list of instantiated backbones
                head=head_ch1
            ).to(self.device)

            # 4. Load the *COMPLETE fine-tuned* model weights
            # This loads both the (potentially updated) backbone weights and the head weights saved during Stage 2.
            finetuned_weights_path = resolve_path(self.WEIGHTS_CH1_FINETUNED)
            print(f"Loading complete fine-tuned weights from: {finetuned_weights_path}")
            model_challenge1.load_state_dict(
                torch.load(finetuned_weights_path, map_location=self.device)
            )

            print("Challenge 1 Model loaded successfully.")
            # Set to evaluation mode is crucial for consistent predictions
            return model_challenge1.eval()

        except FileNotFoundError as e:
            print(f"FATAL ERROR during Challenge 1 model loading: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during Challenge 1 model loading: {e}")
            # You might want to print more details or re-raise depending on debugging needs
            import traceback
            traceback.print_exc()
            raise

    def get_model_challenge_2(self):
        """
        Instantiates and loads the fine-tuned model for Challenge 2.
        Returns:
            torch.nn.Module: The loaded model in evaluation mode.
        """
        print("\n--- Loading Model for Challenge 2 ---")
        try:
            # 1. Instantiate Backbones (same architecture as CH1)
            backbones_ch2 = []
            print(f"Attempting to load {self.N_FOLDS_EXPECTED} backbone folds...")
            for fold in range(self.N_FOLDS_EXPECTED):
                backbone = EegMambaJEPA(
                    d_model=self.D_MODEL, n_layer=self.N_LAYERS, n_channels=self.N_CHANNELS,
                    patch_size=self.PATCH_SIZE, d_state=self.D_STATE, expand=self.EXPAND
                )
                # Load backbone weights individually if needed (usually handled by loading the full finetuned state_dict)
                backbones_ch2.append(backbone.to(self.device))

            # 2. Instantiate Head (MDN for Externalizing Factor distribution)
            head_ch2 = MDNHead(
                input_dim=self.D_MODEL,
                n_components=self.MDN_COMPONENTS
            ).to(self.device)

            # 3. Combine into Ensemble Model
            model_challenge2 = EnsembleCompetitionModel(
                backbones=backbones_ch2,
                head=head_ch2
            ).to(self.device)

            # 4. Load the COMPLETE fine-tuned model weights
            finetuned_weights_path = resolve_path(self.WEIGHTS_CH2_FINETUNED)
            print(f"Loading complete fine-tuned weights from: {finetuned_weights_path}")
            model_challenge2.load_state_dict(
                torch.load(finetuned_weights_path, map_location=self.device)
            )

            print("Challenge 2 Model loaded successfully.")
            # Set to evaluation mode
            return model_challenge2.eval()

        except FileNotFoundError as e:
            print(f"FATAL ERROR during Challenge 2 model loading: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during Challenge 2 model loading: {e}")
            import traceback
            traceback.print_exc()
            raise

# =============================================================================
# Local Testing Snippet (Optional - for debugging outside Codabench)
# =============================================================================
# This part will only run if the script is executed directly (e.g., python submission.py)
# It won't run inside the Codabench environment.
if __name__ == "__main__":
    print("\n--- Running Local Test Snippet ---")
    # Simulate competition environment parameters
    SFREQ_TEST = 100
    # Attempt to use CUDA if available, otherwise CPU
    DEVICE_TEST = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE_TEST = 4
    N_CHANNELS_TEST = 129
    CHUNK_SIZE_S_TEST = 2 # Match fine-tuning chunk size
    N_TIMES_TEST = int(CHUNK_SIZE_S_TEST * SFREQ_TEST)

    print(f"Local Test: SFREQ={SFREQ_TEST}, Device={DEVICE_TEST}, Batch={BATCH_SIZE_TEST}")

    try:
        # Instantiate the submission class
        submission_instance = Submission(SFREQ_TEST, DEVICE_TEST)

        # --- Test Challenge 1 Model ---
        print("\nTesting Challenge 1 Model Loading...")
        model_1 = submission_instance.get_model_challenge_1()
        print("Challenge 1 Model OK.")

        # Create dummy input data
        dummy_input_ch1 = torch.randn(BATCH_SIZE_TEST, N_CHANNELS_TEST, N_TIMES_TEST).to(DEVICE_TEST)

        # Perform a forward pass (inference mode)
        print("Testing Challenge 1 Forward Pass...")
        with torch.no_grad():
            output_ch1 = model_1(dummy_input_ch1)

        # Check output shape (should be [BATCH_SIZE_TEST, 1] for linear head)
        print(f"Challenge 1 Output Shape: {output_ch1.shape}")
        assert output_ch1.shape == (BATCH_SIZE_TEST, 1), "Output shape mismatch for CH1!"
        print("Challenge 1 Forward Pass OK.")
        del model_1, dummy_input_ch1, output_ch1; torch.cuda.empty_cache()

        # --- Test Challenge 2 Model ---
        print("\nTesting Challenge 2 Model Loading...")
        model_2 = submission_instance.get_model_challenge_2()
        print("Challenge 2 Model OK.")

        # Create dummy input data
        dummy_input_ch2 = torch.randn(BATCH_SIZE_TEST, N_CHANNELS_TEST, N_TIMES_TEST).to(DEVICE_TEST)

        # Perform a forward pass
        print("Testing Challenge 2 Forward Pass...")
        with torch.no_grad():
            pi, sigma, mu = model_2(dummy_input_ch2)

        # Check output shapes for MDN head
        print(f"Challenge 2 Output Shapes: pi={pi.shape}, sigma={sigma.shape}, mu={mu.shape}")
        n_components = submission_instance.MDN_COMPONENTS
        assert pi.shape == (BATCH_SIZE_TEST, n_components), "Output shape mismatch for CH2 pi!"
        assert sigma.shape == (BATCH_SIZE_TEST, n_components), "Output shape mismatch for CH2 sigma!"
        assert mu.shape == (BATCH_SIZE_TEST, n_components), "Output shape mismatch for CH2 mu!"
        print("Challenge 2 Forward Pass OK.")
        del model_2, dummy_input_ch2, pi, sigma, mu; torch.cuda.empty_cache()

        print("\n✅ [SUCCESS] Local tests completed successfully.")

    except ImportError:
         print("\n❌ [ERROR] Local test failed due to missing Mamba import.")
         print("   Ensure 'mamba_ssm' and 'causal_conv1d' are installed or provided in 'vendor/'.")
    except FileNotFoundError as e:
        print(f"\n❌ [ERROR] Local test failed: Could not find model weights.")
        print(f"   Details: {e}")
        print(f"   Ensure '{submission_instance.WEIGHTS_CH1_FINETUNED}', '{submission_instance.WEIGHTS_CH2_FINETUNED}', "
              f"and '{submission_instance.WEIGHTS_BACKBONE_FOLD_PREFIX}X.pth' exist in the same directory or specified paths.")
    except Exception as e:
        print(f"\n❌ [ERROR] An unexpected error occurred during local testing:")
        import traceback
        traceback.print_exc()