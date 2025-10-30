# ##########################################################################
# # EEG Foundation Challenge 2025 - Model Definitions
# # JEMA-EEG25 (Mamba + JEPA) Implementation (Single H200 Version)
# #
# # Contains definitions for:
# # - PatchEmbed
# # - NeuroBiMambaBlock
# # - EegMambaJEPA (Backbone)
# # - VICReg (Loss - included for potential use, though not directly in submission)
# # - MDNHead (Challenge 2 Head)
# # - EnsembleCompetitionModel (Wrapper for submission)
# ##########################################################################

import torch
import torch.nn as nn
from einops import rearrange

# Attempt to import Mamba, provide guidance if it fails
try:
    from mamba_ssm import Mamba
except ImportError:
    print("WARNING: 'mamba_ssm' not found during module definition.")
    print("Ensure it's installed in your environment or provided in the 'vendor' directory for submission.")
    # Define a dummy class to allow script loading, but it will fail at runtime if not installed
    class Mamba(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            print("ERROR: Using dummy Mamba class. Installation required.")
        def forward(self, x):
            raise NotImplementedError("Mamba-SSM is not installed.")

# --- Default Configuration Constants (Mirror CFG from training) ---
# It's better to pass these as arguments during instantiation in submission.py,
# but providing defaults makes the file runnable standalone if needed.
D_MODEL_DEFAULT = 256
N_LAYERS_DEFAULT = 8
N_CHANNELS_DEFAULT = 129
PATCH_SIZE_DEFAULT = 10
D_STATE_DEFAULT = 16
EXPAND_DEFAULT = 2
D_CONV_DEFAULT = 4
VICREG_LAMBDA_DEFAULT = 25.0
VICREG_MU_DEFAULT = 25.0
MDN_COMPONENTS_DEFAULT = 5

# --- Patch Embedding Layer ---
class PatchEmbed(nn.Module):
    """
    EEG Patch Embedding.
    Takes (Batch, Channels, Time) -> (Batch, NumPatches, EmbedDim)
    """
    def __init__(self, n_channels=N_CHANNELS_DEFAULT, embed_dim=D_MODEL_DEFAULT, patch_size=PATCH_SIZE_DEFAULT):
        super().__init__()
        self.proj = nn.Conv1d(
            n_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, D_MODEL, NumPatches)
        x = x.permute(0, 2, 1)  # (B, NumPatches, D_MODEL)
        return x

# --- Bi-Directional Mamba Block ---
class NeuroBiMambaBlock(nn.Module):
    def __init__(self, d_model=D_MODEL_DEFAULT, d_state=D_STATE_DEFAULT, expand=EXPAND_DEFAULT, d_conv=D_CONV_DEFAULT):
        super().__init__()
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False) # Mamba often omits bias here
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner, bias=True
        )
        self.activation = nn.SiLU()
        self.mamba_fwd = Mamba(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=1)
        self.mamba_bwd = Mamba(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=1)
        self.out_proj = nn.Linear(2 * self.d_inner, d_model, bias=False) # Mamba often omits bias here
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_skip = x; x = self.norm(x)
        x_proj = self.in_proj(x); x_conv_in, res = x_proj.chunk(2, dim=-1)
        x_conv_in = rearrange(x_conv_in, "b l d -> b d l")
        # Apply causal padding crop AFTER convolution
        x_conv_out = self.conv1d(x_conv_in)[:, :, :x.shape[1]]
        x_conv_out = rearrange(x_conv_out, "b d l -> b l d")
        x_conv_activated = self.activation(x_conv_out)
        x_fwd = self.mamba_fwd(x_conv_activated)
        # Process backward sequence by flipping, applying mamba, flipping back
        x_bwd = torch.flip(self.mamba_bwd(torch.flip(x_conv_activated, dims=[1])), dims=[1])
        x_mamba_out = torch.cat([x_fwd, x_bwd], dim=-1) # (B, L, 2 * D_EXPAND)
        x_out = self.out_proj(x_mamba_out * self.activation(res)) # Gated MLP
        return x_out + x_skip


# --- JEPA Backbone ---
class EegMambaJEPA(nn.Module):
    def __init__(self,
                 d_model=D_MODEL_DEFAULT,
                 n_layer=N_LAYERS_DEFAULT,
                 n_channels=N_CHANNELS_DEFAULT,
                 patch_size=PATCH_SIZE_DEFAULT,
                 d_state=D_STATE_DEFAULT,
                 expand=EXPAND_DEFAULT
                 ):
        super().__init__()
        self.patch_embed = PatchEmbed(n_channels, d_model, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # Optional: Add positional embeddings if needed, Mamba often doesn't require them.
        # max_len = 1000 # Example max sequence length (adjust based on data/chunks)
        # self.pos_embed = nn.Parameter(torch.randn(1, max_len + 1, d_model))

        self.mamba_blocks = nn.Sequential(
            *[NeuroBiMambaBlock(d_model=d_model, d_state=d_state, expand=expand) for _ in range(n_layer)]
        )
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: (B, C, T)
        B = x.shape[0]
        x = self.patch_embed(x)  # -> (B, NumPatches, D_MODEL)
        num_patches = x.shape[1]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # -> (B, 1 + NumPatches, D_MODEL)

        # Optional: Add positional embedding
        # x = x + self.pos_embed[:, :(num_patches + 1)]

        # Pass through Mamba blocks
        x = self.mamba_blocks(x)

        # Final normalization
        x = self.norm_f(x)

        # Return the CLS token's representation
        return x[:, 0]

# --- VICReg Loss (Included for reference, not used in submission.py) ---
class VICReg(nn.Module):
    def __init__(self, d_model=D_MODEL_DEFAULT, lambda_val=VICREG_LAMBDA_DEFAULT, mu_val=VICREG_MU_DEFAULT, eps=1e-4):
        super().__init__()
        self.lambda_val = lambda_val; self.mu_val = mu_val; self.eps = eps
        # Projector used during pre-training
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    def forward(self, z1, z2):
        z1p = self.projector(z1); z2p = self.projector(z2)
        # Invariance term (MSE)
        repr_loss = nn.functional.mse_loss(z1p, z2p)

        # Variance term (Hinge Loss on std dev)
        z1_norm = z1p - z1p.mean(dim=0); z2_norm = z2p - z2p.mean(dim=0)
        std_z1 = torch.sqrt(z1_norm.var(dim=0) + self.eps); std_z2 = torch.sqrt(z2_norm.var(dim=0) + self.eps)
        std_loss = (torch.mean(nn.functional.relu(1 - std_z1)) + torch.mean(nn.functional.relu(1 - std_z2))) / 2

        # Covariance term (L2 norm of off-diagonal elements)
        B = z1_norm.shape[0]
        cov_z1 = (z1_norm.T @ z1_norm) / (B - 1); cov_z2 = (z2_norm.T @ z2_norm) / (B - 1)
        # Create a mask for off-diagonal elements
        off_diag_mask = ~torch.eye(z1_norm.shape[1], device=z1_norm.device).bool()
        # Sum squared off-diagonal elements and normalize by dimension
        cov_loss = (cov_z1[off_diag_mask].pow(2).sum() / z1_norm.shape[1] +
                    cov_z2[off_diag_mask].pow(2).sum() / z1_norm.shape[1])

        # Combine losses
        loss = self.lambda_val * repr_loss + self.mu_val * std_loss + cov_loss
        return loss

# --- MDN Head & Loss (Loss function included for reference) ---
class MDNHead(nn.Module):
    def __init__(self, input_dim=D_MODEL_DEFAULT, n_components=MDN_COMPONENTS_DEFAULT):
        super().__init__()
        self.n_components = n_components
        # Layers to predict parameters of the Gaussian mixture
        self.pi = nn.Linear(input_dim, n_components)      # Mixture weights (logits)
        self.sigma = nn.Linear(input_dim, n_components)   # Standard deviations (log scale)
        self.mu = nn.Linear(input_dim, n_components)       # Means

    def forward(self, x):
        # Predict parameters
        pi_logits = self.pi(x)
        # Apply softmax for mixture weights, ensuring they sum to 1
        pi = torch.softmax(pi_logits, dim=1)

        # Predict log standard deviations and ensure sigma is positive using exp
        # Add a small epsilon for numerical stability
        sigma = torch.exp(self.sigma(x)) + 1e-6

        # Predict means
        mu = self.mu(x)

        return pi, sigma, mu

def mdn_loss(pi, sigma, mu, y, reduce=True):
    """Calculates the Mixture Density Network loss."""
    # Ensure y has the correct shape for broadcasting: (B, 1)
    if y.dim() == 1: y = y.unsqueeze(-1)
    if y.dim() == 2 and y.shape[1] != 1:
        raise ValueError(f"Target y must be shape (B,) or (B, 1), but got {y.shape}")

    # Create the mixture distribution
    # Normal distribution component: N(mu | sigma^2)
    m = torch.distributions.Normal(loc=mu, scale=sigma)

    # Calculate the log probability density for each component
    # log N(y | mu_k, sigma_k^2)
    # y broadcasts from (B, 1) to (B, N_COMPONENTS)
    log_prob = m.log_prob(y)

    # Ensure log_prob is numerically stable (clamp potential -inf)
    log_prob = torch.clamp(log_prob, min=-1e9, max=1e9) # Also clamp max for stability

    # Calculate log mixture weights (log pi_k) using log_softmax for stability
    log_pi = torch.log_softmax(pi, dim=1)

    # Combine using log-sum-exp for stability: log( sum[ pi_k * N(y | ...) ] )
    # logsumexp( log(pi_k) + log N(y | ...) )
    log_likelihood = torch.logsumexp(log_pi + log_prob, dim=1)

    # Negative log likelihood loss
    loss = -log_likelihood

    if reduce:
        return loss.mean()
    else:
        return loss

# --- Ensemble Wrapper for Submission ---
class EnsembleCompetitionModel(nn.Module):
    def __init__(self, backbones, head):
        """
        Wrapper to average backbone features before passing to the head.
        Used during fine-tuning and submission inference.
        Args:
            backbones (list or nn.ModuleList): List of pre-trained backbone models.
            head (nn.Module): The final head (e.g., nn.Linear or MDNHead).
        """
        super().__init__()
        # Ensure backbones is a ModuleList to register them correctly
        if not isinstance(backbones, nn.ModuleList):
            self.backbones = nn.ModuleList(backbones)
        else:
            self.backbones = backbones
        self.head = head

    def forward(self, x):
        """
        Args:
            x (Tensor): Input EEG data (B, C, T).
        Returns:
            Tensor or Tuple: Output from the head (depends on the head type).
        """
        # Get CLS token features from all backbones
        features = []
        is_head_training = self.head.training # Check head status before changing backbone mode

        for bb in self.backbones:
            # Important: Ensure backbones are in eval mode if their weights are frozen
            # (which they are during fine-tuning and inference)
            original_bb_mode = bb.training
            if not any(p.requires_grad for p in bb.parameters()):
                 bb.eval() # Set to eval if frozen

            features.append(bb(x))

            # Restore original mode if it was changed
            if not any(p.requires_grad for p in bb.parameters()) and original_bb_mode:
                 bb.train() # Set back to train if it was originally training (shouldn't happen here)


        # Average the features across the ensemble dimension
        avg_features = torch.mean(torch.stack(features), dim=0)

        # Pass averaged features through the head
        # Ensure head is in the correct mode (train/eval)
        self.head.train(is_head_training) # Match head mode to original input mode
        output = self.head(avg_features)

        return output

print("my_model.py definitions complete and ready.")