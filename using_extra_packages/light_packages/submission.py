# ##########################################################################
# # Example of submission files with extra Python packages
# # --------------------------------------------------------------------------
# # This file shows how to include extra Python packages in your submission
# # and how to use them in your submission file.
# #
# # This TUTORIAL WILL ONLY WORK WITH LINUX!
# #
# # This type of installation is suitable for small packages that do not
# # require compilation, example torch_geometric.
# #
# # If you need to install large packages or packages
# # that require C++ compilation, follow the tutorial in preparing_your_model.py
# # in this repository folder heavy_packages.
# # --------------------------------------------------------------------------
# # IMPORTANT: Make sure that the extra packages you include are compatible
# # with the Python version and the packages already installed in the
# # codabench environment. You can check the Codabench environment by
# # running `uv` in the terminal after installing uv.
# # --------------------------------------------------------------------------
# # The codabench environment has Python 3.10 and the following packages
# # already installed: github.com/eeg2025/startkit
# ##########################################################################
# #
# # To include extra Python packages, you need to:
# # 0) Create a fresh environment based on the codabench environment.
# #    We recommend to use uv the environment you will use to install the packages.
# #    You can do this by running:
# #    cd codalab-env
# #    uv sync 
# #    uv activate
# # 1) Create a folder named `python_packages` in the same directory as this
# #    submission file.
# # 2) Install the packages you need in that folder. You can do this by
# #    running pip install with the target in your
# #    terminal.
# #    uv pip install --target PATH_FOR_YOUR_FOLDER/python_packages <package_name>
# # 3) Test locally that your submission file works with the extra packages.
# #    You can run `python submission.py` in your terminal with the uv env activated!
# #    Make sure you are in the same directory as this submission file.
# #    You will need to include the `python_packages` folder in your
# #    PYTHONPATH to test it locally. Code to do this is included in the
# #    `resolve_path` function below.
# # 4) Zip the `python_packages` folder along with your submission file and
# #    any other files you need (e.g., model weights) into a single zip
# #    file.
# #    You can do this by running the following command in your terminal:
# #    (cd PATH_FOR_YOUR_FOLDER && zip -r ../submission.zip .)
# #    Only zip what is needed for your submission to run.
# #    DO NOT zip the entire PROJECT or any unnecessary files.
# # 5) Upload the zip file to Codabench.
# #

# # Note that the `python_packages` folder can be large depending on the
# # packages you install. Make sure to check the size of your zip file before
# # uploading it to Codabench. 
    
from braindecode.models import EEGNeX
import torch

from pathlib import Path

def resolve_path(name="python_packages"):
    if Path(f"/app/input/res/{name}").exists():
        return f"/app/input/res/{name}"
    elif Path(f"/app/input/{name}").exists():
        return f"/app/input/{name}"
    elif Path(f"{name}").exists():
        return f"{name}"
    elif Path(__file__).parent.joinpath(f"{name}").exists():
        return str(Path(__file__).parent.joinpath(f"{name}"))
    else:
        raise FileNotFoundError(
            f"Could not find {name} folder in /app/input/res/ or /app/input/ or current directory"
        )

class ModelWithExtraDeps(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # EXAMPLE OF USING torch_geometric
        # YOU CAN MODIFY THIS TO USE THE PACKAGE YOU INSTALLED
        import sys
        sys.path.append(resolve_path())
        import torch_geometric
        print("Using torch_geometric version", torch_geometric.__version__)
        self.real_model = EEGNeX(
            n_chans=129, n_outputs=1, sfreq=100, n_times=int(2 * 100)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.real_model(x)

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        model_challenge1 = ModelWithExtraDeps().to(self.device)
        # load from the current directory (/app/input/ is where the file resides on Codabench)
        # model_challenge1.load_state_dict(torch.load("os.path.join(os.path.dirname(__file__)", map_location=self.device))
        return model_challenge1

    def get_model_challenge_2(self):
        model_challenge2 = EEGNeX(
            n_chans=129, n_outputs=1, n_times=int(2 * self.sfreq)
        ).to(self.device)
        # model_challenge2.load_state_dict(torch.load("/app/input/weights_challenge_2.pt", map_location=self.device))
        return model_challenge2
