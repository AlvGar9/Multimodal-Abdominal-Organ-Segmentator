# src/training/engine.py
from typing import Tuple
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torchmetrics.segmentation import Dice
from monai.data import DataLoader

def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: callable,
    scaler: GradScaler,
    device: torch.device
) -> float:
    # This function remains unchanged.
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        img = batch['image'].to(device)
        lab_ch = batch["label"].to(device)
        optimizer.zero_grad()
        with autocast(device_type=str(device)):
            out = model(img)
            loss = loss_fn(out, lab_ch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    return running_loss / len(loader)

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: callable,
    num_classes: int,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Runs validation, returning average loss and mean foreground Dice score.
    """
    model.eval()
    running_loss = 0.0
    dice_metric = Dice(num_classes=num_classes, average='none').to(device)

    for batch in tqdm(loader, desc="Validating", leave=False):
        img = batch["image"].to(device)
        lab_ch = batch["label"].to(device)
        lab = lab_ch.squeeze(1).long()
        with autocast(device_type=str(device)):
            out = model(img)
            loss = loss_fn(out, lab_ch)
        running_loss += loss.item()
        preds_classes = out.softmax(dim=1).argmax(dim=1)
        dice_metric.update(preds_classes, lab)
        
    avg_loss = running_loss / len(loader)
    dice_scores = dice_metric.compute().cpu()
    
    # Use nanmean for robust calculation of foreground dice, ignoring potential NaN values.
    mean_fg_dice = torch.nanmean(dice_scores[1:]).item()
    
    return avg_loss, mean_fg_dice