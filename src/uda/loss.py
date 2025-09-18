# src/uda/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AmplifiedCosineLoss(nn.Module):
    """
    Calculates the cosine distance (1 - similarity) and amplifies the loss
    using a power function. This creates a stronger gradient when vectors are
    already very similar, preventing the alignment loss from vanishing prematurely.
    """
    def __init__(self, amplification_power: float = 0.25):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.power = amplification_power

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalize to focus on direction, not magnitude
        source = F.normalize(source, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)
        
        cos_sim = self.cosine_similarity(source, target)
        
        # The raw loss (1 - cos_sim) approaches 0 for similar vectors.
        # Raising this small value to a power < 1 amplifies it.
        raw_loss = 1.0 - cos_sim
        amplified_loss = (raw_loss + 1e-6).pow(self.power) # Epsilon for stability
        
        return amplified_loss.mean()