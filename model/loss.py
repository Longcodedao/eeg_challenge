import torch
import torch.nn as nn
import torch.nn.functional as F

class VICReg(nn.Module):
    def __init__(self, d_model = 256, 
                       lambda_val = 25.0, 
                       mu_val = 25.0,
                       nu_val = 1.0,  
                       eps = 1e-4):
        
        super().__init__()

        # Invariance weight
        self.lambda_val = lambda_val
        # Variance weight
        self.mu_val = mu_val
        # Covariance weight
        self.nu_val = nu_val
        self.eps = eps

        # Projector used during pre-training
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def _invariance_loss(self, z1p, z2p):
        """MSE between paired projections (encourages invariance)."""
        return F.mse_loss(z1p, z2p)

    def _variance_loss(self, z1p, z2p):
        """
        Hinge loss on per-dimension standard deviation to avoid collapse.
        Penalizes dimensions with std < 1.
        """
        z1_norm = z1p - z1p.mean(dim = 0)
        z2_norm = z2p - z2p.mean(dim = 0)
        std_z1 = torch.sqrt(z1_norm.var(dim=0) + self.eps) 
        std_z2 = torch.sqrt(z2_norm.var(dim=0) + self.eps)

        hinge_1 = torch.mean(F.relu(1 - std_z1))
        hinge_2 = torch.mean(F.relu(1 - std_z2))

        return 0.5 * (hinge_1 + hinge_2)

    def _covariance_loss(self, z1p, z2p):
        """
        Off-diagonal covariance penalty: encourages feature decorrelation.
        Returns 0 if batch size < 2 to avoid divide-by-zero.
        """
        z1_norm = z1p - z1p.mean(dim = 0)
        z2_norm = z2p - z2p.mean(dim = 0)

        B, D = z1_norm.shape
        if B < 2:
            return torch.tensor(0.0, device=z1p.device)

        cov_z1 = (z1_norm.T @ z1_norm) / (B - 1)
        cov_z2 = (z2_norm.T @ z2_norm) / (B - 1)

        # Create a mask for off-diagonal elements
        off_diag_mask = ~torch.eye(D, device=z1_norm.device).bool()

        # Sum squared off-diagonal elements and normalize by dimension
        cov_z1 = cov_z1[off_diag_mask].pow(2).sum() / D 
        cov_z2 = cov_z2[off_diag_mask].pow(2).sum() / D

        cov_loss = 0.5 * (cov_z1 + cov_z2)

        return cov_loss

    def forward(self, z1, z2):

        z1p = self.projector(z1)
        z2p = self.projector(z2)

        inv = self._invariance_loss(z1p, z2p)    
        var = self._variance_loss(z1p, z2p)
        cov = self._covariance_loss(z1p, z2p)

        loss = self.lambda_val * inv + self.mu_val * var + self.nu_val * cov

        return loss




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