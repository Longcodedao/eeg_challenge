import torch
import torch.nn as nn
import torch.nn.functional as F

class MDNHead(nn.Module):
    """
    Mixture Density Network head for predicting parameters of a Gaussian Mixture Model.
    Given input features, predicts mixture weights (pi), means (mu), and standard deviations (sigma).
    """
    def __init__(self, input_dim: int, n_mixtures: int, min_sigma: float = 1e-3):
        super(MDNHead, self).__init__()
        self.n_mixtures = n_mixtures

        # Linear layers to predict mixture weights, means, and log standard deviations
        self.pi = nn.Linear(input_dim, n_mixtures)
        self.mu = nn.Linear(input_dim, n_mixtures)
        self.sigma = nn.Linear(input_dim, n_mixtures)

        # Softplus activation to ensure positive standard deviations
        self.softplus = nn.Softplus()
        self.min_sigma = min_sigma  # Minimum standard deviation for numerical stability

    def forward(self, x: torch.Tensor):
        """
        Forward pass to predict GMM parameters.
        
        Args:
            x: Input tensor of shape (B, input_dim)
        
        Returns:
            pi: Mixture weights of shape (B, n_mixtures)
            sigma: Standard deviations of shape (B, n_mixtures)
            mu: Means of shape (B, n_mixtures)
        """
        # Mixture weights
        pi_logits = self.pi(x)
        pi = torch.softmax(pi_logits, dim=1)

        # Means
        mu = self.mu(x)

        # Positive scales via softplus for numerical stability
        sigma = self.softplus(self.sigma(x)) + self.min_sigma

        return pi, sigma, mu