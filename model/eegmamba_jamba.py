import typing
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange
import copy
from .mdn import MDNHead

# --- Patch Embedding Layer ---
class PatchEmbed(nn.Module):
    """
    EEG Patch Embedding.
    Takes (Batch, Channels, Time) -> (Batch, NumPatches, EmbedDim)
    """
    def __init__(self, n_channels = 129, 
                        embed_dim = 256, 
                        patch_size = 10):
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


class NeuroBiMambaBlock(nn.Module):
    """
    Bi-directional Mamba block with a depthwise 1D conv front-end and a gated MLP-style
    projection. Input/Output shape: (B, L, d_model).

    Workflow:
      1. LayerNorm -> linear projection that produces two halves:
         - conv_input (for depthwise conv + activation)
         - gate_tensor (used to gate Mamba output)
      2. Depthwise conv applied along the sequence dimension (causally cropped).
      3. Run Mamba forward on the activated conv output and also on the reversed
         sequence to get a backward context; concatenate them.
      4. Gate the concatenated mamba output, project back to d_model and add residual.
    """
    def __init__(self,
                 d_model: int = 256,
                 d_state: int = 16,
                 expand: int = 2,
                 d_conv: int = 4):
        super().__init__()
        # Hidden dimension after expansion (used for conv & mamba)
        self.hidden_dim = d_model * expand

        # Project input -> [conv_input | gate_residual]  (shape: 2 * hidden_dim)
        self.in_proj = nn.Linear(d_model, 2 * self.hidden_dim, bias=False)

        # Depthwise 1D conv: expects (B, hidden_dim, L)
        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,  # causal-style padding, we'll crop to original length
            groups=self.hidden_dim,
            bias=True,
        )

        self.activation = nn.SiLU()

        # Two Mamba blocks: one for forward context, one for backward (via flip)
        self.mamba_fwd = Mamba(d_model=self.hidden_dim, d_state=d_state, 
                               d_conv=d_conv, expand=1)
        self.mamba_bwd = Mamba(d_model=self.hidden_dim, d_state=d_state, 
                               d_conv=d_conv, expand=1)

        # Project concatenated (fwd | bwd) -> d_model
        self.out_proj = nn.Linear(2 * self.hidden_dim, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)

    def _split_projection(self, x: torch.Tensor):
        # x: (B, L, d_model) -> x_proj: (B, L, 2*hidden_dim)
        x_proj = self.in_proj(x)
        conv_input, gate = x_proj.chunk(2, dim=-1)  # each (B, L, hidden_dim)
        return conv_input, gate

    def _conv_activate(self, conv_input: torch.Tensor, seq_len: int):
        # conv_input: (B, L, hidden_dim) -> conv expects (B, hidden_dim, L)
        y = rearrange(conv_input, "b l d -> b d l")
        # conv1d with padding may extend length; crop to original sequence length
        y = self.conv1d(y)[:, :, :seq_len]
        y = rearrange(y, "b d l -> b l d")
        return self.activation(y)  # (B, L, hidden_dim)

    def _run_bi_mamba(self, activated: torch.Tensor):
        # activated: (B, L, hidden_dim)
        fwd = self.mamba_fwd(activated)  # (B, L, hidden_dim)
        # run backward by flipping sequence dimension
        bwd_in = torch.flip(activated, dims=[1])
        bwd_out = self.mamba_bwd(bwd_in)
        bwd = torch.flip(bwd_out, dims=[1])  # restore original order
        # concat along feature dim -> (B, L, 2*hidden_dim)
        return torch.cat([fwd, bwd], dim=-1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        residual = x
        x = self.norm(x)

        # 1) linear proj -> conv input + gate tensor
        conv_input, gate = self._split_projection(x)

        # 2) depthwise conv + activation (preserve seq length)
        conv_activated = self._conv_activate(conv_input, seq_len=x.shape[1])

        # 3) bi-directional Mamba processing
        mamba_out = self._run_bi_mamba(conv_activated)  # (B, L, 2*hidden_dim)
        fwd_out, bwd_out = mamba_out.chunk(2, dim=-1)  # each (B, L, hidden_dim)

        # 4) gated output and projection back to d_model   
        gate_activation = self.activation(gate)
        gated_fwd = fwd_out * gate_activation
        gated_bwd = bwd_out * gate_activation
        gated = torch.cat([gated_fwd, gated_bwd], dim=-1)  # (B, L, 2*hidden_dim)

        out = self.out_proj(gated)  # (B, L, d_model)

        # residual connection
        return out + residual


# ...existing code...
class EegMambaJEPA(nn.Module):
    """
    JEPA-style backbone using Patch embedding + stacked NeuroBiMambaBlock layers.

    Input: (B, C, T): (Batch, Channels, Time)
    Output: CLS token embedding -> (B, d_model)
    """
    def __init__(
        self,
        d_model: int = 256,
        n_layer: int = 8,
        n_channels: int = 129,
        patch_size: int = 10,
        d_state: int = 16,
        expand: int = 2,
        use_pos_embed: bool = False,
        max_len: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_pos_embed = use_pos_embed

        # Patch embedding: (B, C, T) -> (B, NumPatches, d_model)
        self.patch_embed = PatchEmbed(n_channels, d_model, patch_size)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Optional positional embeddings (applied after adding CLS)
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.randn(1, max_len + 1, d_model))

        # Stack of NeuroBiMamba blocks
        self.mamba_blocks = self._build_mamba_stack(n_layer, d_model, d_state, expand)

        # Final layer norm
        self.norm_f = nn.LayerNorm(d_model)

    def _build_mamba_stack(self, n_layer: int, d_model: int, d_state: int, expand: int) -> nn.Sequential:
        """Create a sequential stack of NeuroBiMambaBlock modules."""
        blocks = [
            NeuroBiMambaBlock(d_model=d_model, d_state=d_state, expand=expand)
            for _ in range(n_layer)
        ]
        return nn.Sequential(*blocks)

    def _prepend_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend the CLS token to a batch of patch embeddings."""
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        return torch.cat((cls_tokens, x), dim=1)

    def get_num_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters in EegMambaJEPA: {total_params}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        returns: (B, d_model)  -- embedding for CLS token
        """

        B, C, T = x.shape

        # --- START: Per-Window Standardization ---
        # This is the "transformation inside the model"
        # It ensures train/test data are identically normalized (z-score).
        # We calculate mean/std per sample (B), across channels (C) and time (T)
        mean = torch.mean(x, dim=(1, 2), keepdim=True)
        std = torch.std(x, dim=(1, 2), keepdim=True)
        x = (x - mean) / (std + 1e-6)  # Add epsilon to prevent division by zero
        
        # --- END: Per-Window Standardization ---

        # Embed patches
        x = self.patch_embed(x)  # (B, NumPatches, d_model)

        # Prepend CLS token
        x = self._prepend_cls_token(x)  # (B, 1 + NumPatches, d_model)

        # Optional positional embeddings (truncate to current length)
        if self.use_pos_embed:
            seq_len = x.shape[1]
            x = x + self.pos_embed[:, :seq_len]

        # Pass through Mamba blocks and final norm
        x = self.mamba_blocks(x)
        x = self.norm_f(x)

        # Return CLS representation
        return x[:, 0]


    def attach_target(self, 
                      device: typing.Optional[torch.device] = None) -> 'EegMambaJEPA':
        """
        Create an EMA target copy of this model. Call once after instantiation.

        Args:
            device: optional device to move the target model to (e.g. torch.device('cuda')).
        """
        self.target_model = copy.deepcopy(self)

        # Freeze target parameters
        for param in self.target_model.parameters():
            param.requires_grad = False

        # Put target in eval mode
        self.target_model.eval()

        if device is not None:
            self.target_model = self.target_model.to(device)

    @torch.no_grad()
    def update_target(self, decay: float = 0.999):
        if self.target_model is None:
            raise ValueError("Target model not attached. Call attach_target() first.")
        
        for param_o, param_t in zip(self.parameters(), self.target_model.parameters()):
            param_t.data.mul_(decay).add_(param_o.data, alpha=1.0 - decay)

    @torch.no_grad()
    def target_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_model is None:
            raise ValueError("Target model not attached. Call attach_target() first.")
        return self.target_model(x)


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


class FinetuneJEPA_Challenge2(nn.Module):
    def __init__(self, n_channels=129, d_model=256, n_layer=8, patch_size=10, meta_dim=13):
        super().__init__()
        self.backbone = EegMambaJEPA(
            n_channels=n_channels,
            d_model=d_model,
            n_layer=n_layer,
            patch_size=patch_size,
        )
        self.meta_proj = nn.Linear(meta_dim, d_model)

        # MDN Heads
        self.train_head = MDNHead(input_dim=d_model * 2, n_mixtures=3)
        self.submit_head = MDNHead(input_dim=d_model, n_mixtures=3)

        # MODE
        self.mode = "train"  # "train", "val", "submit"

        # New: Dummy parameter to ensure submission head is always used in training
        # This is a common DDP trick for models with conditional branches.
        self.dummy_param = nn.Parameter(torch.empty(1))
        nn.init.constant_(self.dummy_param, 0.0)
        
        # New: Register the metadata tensor for backward pass in the submit phase
        self.register_buffer('_meta_zero', torch.zeros(1, meta_dim))

    def forward(self, x, meta=None):
        z = self.backbone(x)  # (B, d_model)
        batch_size = z.shape[0]

        # Prepare the metadata
        if meta is None:
            meta_input = self._meta_zero.expand(batch_size, -1)  # (B, meta_dim)
        else:
            meta_input = meta

        m = self.meta_proj(meta_input)  # (B, d_model)
        z_train = torch.cat([z, m], dim=1)  # (B, d_model * 2)

        # ALways call both heads to ensure all parameters are used in backward
        pi_train, sigma_train, mu_train = self.train_head(z_train)
        pi_submit, sigma_submit, mu_submit = self.submit_head(z)

        # Return the appropriate head output based on the mode
        if self.mode == "train":
            if meta is not None:
                return pi_train, sigma_train, mu_train
            else:
                return pi_submit, sigma_submit, mu_submit
        else:
            # EVALUATION / SUBMISSION: Return single number
            # Option 1: Mean of mixture (weighted)
            pred = (pi_submit * mu_submit).sum(dim=1)  # (B,)
            # Option 2: Best component (highest pi)
            # best_k = pi_submit.argmax(dim=1)
            # pred = mu_submit.gather(1, best_k.unsqueeze(1)).squeeze(1)
            return pred

    # SWITCH MODES
    def train_mode(self):
        self.mode = "train"
        self.train()

    def eval_mode(self):
        self.mode = "val"
        self.eval()

    def submit_mode(self):
        self.mode = "submit"
        self.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("BACKBONE FROZEN. SUBMISSION READY.")

    # FREEZE BACKBONE
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False


