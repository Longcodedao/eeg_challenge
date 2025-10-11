# -*- coding: utf-8 -*-
# # --------------------------------------------------------------------------
# # This submission file shows how to include extra python packages in your submission
# # and how to use them in your submission file.
# #

# # if you need packages that require C++ compilation, follow the tutorial in this file
# # in this repository folder.
# # --------------------------------------------------------------------------
# # To include extra python packages, you need to:
# # 1) Create a new fresh environment based on the based environment.yml file
# #    conda env create -f environment.yml
# #    conda activate codabench-env
# # 2) Transform your model to a TorchScript file (.pt) using this file
# #    preparing_your_model.py in this repository folder.
# # 3) Upload the .pt file along with your submission file to Codabench.
# #    (cd PATH_FOR_YOUR_FOLDER && zip -r ../submission.zip .)
# # 4) In your submission file, load the .pt file and use it for inference.
# #    See the example in submission.py.

from timm import create_model
import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    """Project EEG tensors (B,129,200) into RGB pseudo-images for ResNet."""

    def __init__(self):
        super().__init__()
        self.channel_proj = torch.nn.Conv1d(
            in_channels=129,
            out_channels=3,
            kernel_size=1,
            bias=True,
        )
        self.backbone = create_model("resnet18", pretrained=True, num_classes=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_proj(x)  # (B, 3, 200)
        x = x.unsqueeze(2)  # (B, 3, 1, 200)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = self.backbone(x)
        return x


model = Model()
zero_input = torch.zeros((1, 129, 200))
out = model(zero_input)
print(out.shape)


scripted = torch.jit.trace(model, zero_input)
scripted = torch.jit.optimize_for_inference(scripted)
scripted.save("model_with_extra_deps.pt")
