from pathlib import Path

from braindecode.models import EEGNeX
import torch

def resolve_path(name="model_with_extra_deps"):
    if Path(f"/app/input/res/{name}.pt").exists():
        return f"/app/input/res/{name}.pt"
    elif Path(f"/app/input/{name}.pt").exists():
        return f"/app/input/{name}.pt"
    elif Path(f"{name}.pt").exists():
        return f"{name}.pt"
    elif Path(__file__).parent.joinpath(f"{name}.pt").exists():
        return str(Path(__file__).parent.joinpath(f"{name}.pt"))
    else:
        raise FileNotFoundError(
            f"Could not find {name}.pt in /app/input/res/ or /app/input/ or current directory"
        )

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        model_challenge1 =  torch.jit.load(resolve_path(), map_location=self.device)

        # load from the current directory (/app/input/ is where the file resides on Codabench)
        # model_challenge1.load_state_dict(torch.load("/app/input/weights_challenge_1.pt", map_location=self.device))
        return model_challenge1

    def get_model_challenge_2(self):
        model_challenge2 = EEGNeX(
            n_chans=129, n_outputs=1, n_times=int(2 * self.sfreq)
        ).to(self.device)
        # model_challenge2.load_state_dict(torch.load("/app/input/weights_challenge_2.pt", map_location=self.device))
        return model_challenge2


