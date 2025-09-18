# src/training/loss.py
import torch
from monai.losses import DiceLoss, FocalLoss
from monai.data import DataLoader

def calculate_loss_weights(dataloader: DataLoader, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates class weights for CE and Dice loss based on voxel frequencies.
    """
    # This function remains unchanged from the previous version.
    print("Calculating loss weights from training set...")
    voxel_counts = torch.zeros(num_classes, dtype=torch.float64)
    for batch in dataloader:
        label = batch["label"].flatten()
        for c in range(num_classes):
            voxel_counts[c] += (label == c).sum()
    beta = 0.999
    w_ce = (1 - beta) / (1 - beta ** voxel_counts.clamp(min=1))
    w_ce = w_ce / w_ce.sum()
    w_dice = w_ce.clone()
    w_dice[0] *= 0.01
    print("Loss weights calculated.")
    return w_ce, w_dice

def get_loss_function(
    device: torch.device,
    ce_weights: torch.Tensor = None,
    dice_weights: torch.Tensor = None,
    include_background: bool = True,
):
    """
    Creates the combined Dice and Focal loss function.
    Can be configured to include/exclude background and use class weights.
    """
    dice_loss = DiceLoss(
        to_onehot_y=True, softmax=True,
        weight=dice_weights.to(device) if dice_weights is not None else None,
        include_background=include_background
    )

    focal_loss = FocalLoss(
        to_onehot_y=True,
        gamma=2.0,
        weight=ce_weights.to(device) if ce_weights is not None else None,
        include_background=include_background
    )

    def combined_loss(logits, labels):
        return 0.5 * dice_loss(logits, labels) + 0.5 * focal_loss(logits, labels)
    
    return combined_loss