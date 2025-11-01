import torch 
import torch.nn as nn

class JEPAGPUAugment:
    def __init__(self, in_channels = 129, 
                 chunk_length = 3000,
                padding_percent = 0.1, 
                noise_std = 0.01,
                time_flip_p = 0.5,
                channel_dropout_p = 0.1,
                device = "cuda"):
        
        self.in_channels = in_channels
        self.chunk_length = chunk_length
        self.device = device

        self.pad_size = int(padding_percent * self.chunk_length)  # padding percent padding
        self.pad = nn.ReplicationPad1d((self.pad_size, self.pad_size))
        
        self.noise_std = noise_std
        self.time_flip_p = time_flip_p
        self.channel_dropout_p = channel_dropout_p

    def __call__(self, x_batch):

        # Shape of x_batch: (B, N, T)
        B, N, T = x_batch.shape
        x_batch = x_batch.to(self.device)

        # 1. Pad (B, N, T + 2 * pad_size)
        x_padded = self.pad(x_batch)
        
        # 2. Random crop
        unfold = x_padded.unfold(
            dimension = 2, 
            size = self.chunk_length,
            step = 1
        )
        L = unfold.shape[2]  # Number of possible starting positions
        start_idx = torch.randint(0, L, (B,), device = self.device)
        batch_idx = torch.arange(B, device = self.device)
        cropped = unfold[batch_idx, :, start_idx, :]

        # 3. Time Flip
        # Since EEG is non-directional (alpha waves same forward/backward)
        if torch.rand(1).item() < self.time_flip_p:
            cropped = torch.flip(cropped, dims=[2])

        # 4. Channel Dropout
        # Simulates bad electrodes, forces spatial robustness
        if self.channel_dropout_p > 0:
            channel_mask = (torch.rand(N, device=self.device) > self.channel_dropout_p).float()
            cropped = cropped * channel_mask[None, :, None]

        # 5. Gaussian Noise
        # Models muscle/eye artifacts
        noise = torch.randn_like(cropped) * self.noise_std
        augmented = cropped + noise

        # 6. CLAMP to [-10, 10] µV
        augmented = torch.clamp(augmented, min=-10.0, max=10.0)

        return augmented