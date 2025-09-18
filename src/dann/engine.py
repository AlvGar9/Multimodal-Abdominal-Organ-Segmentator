# src/dann/engine.py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast
from torchmetrics.segmentation import Dice

def get_lambda_p(epoch: int, max_epochs: int) -> float:
    """Calculates the lambda parameter for the GRL, which adapts over epochs."""
    p = epoch / max_epochs
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

def train_epoch_dann(
    model, loader, optimizer, loss_fn_seg, loss_fn_domain, scaler, device, lambda_p
):
    """Runs a single DANN training epoch."""
    model.train()
    running_seg_loss, running_dom_loss = 0.0, 0.0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        img = batch['image'].to(device)
        lab_ch = batch["label"].to(device)
        domain_labels = batch["domain"].to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type=str(device)):
            seg_logits, domain_logits = model(img, lambda_p=lambda_p, return_domain_logits=True)
            seg_loss = loss_fn_seg(seg_logits, lab_ch)
            dom_loss = loss_fn_domain(domain_logits, domain_labels)
            total_loss = seg_loss + lambda_p * dom_loss
            
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_seg_loss += seg_loss.item()
        running_dom_loss += dom_loss.item()
    
    avg_seg_loss = running_seg_loss / len(loader)
    avg_dom_loss = running_dom_loss / len(loader)
    return avg_seg_loss, avg_dom_loss

@torch.no_grad()
def validate_dann(model, loader, loss_fn_seg, device, num_classes):
    """Runs DANN validation, returning metrics for each domain."""
    model.eval()
    running_seg_loss = 0.0
    
    ct_dice_metric = Dice(num_classes=num_classes, average='macro').to(device)
    mri_dice_metric = Dice(num_classes=num_classes, average='macro').to(device)

    for batch in tqdm(loader, desc="Validating", leave=False):
        img = batch["image"].to(device)
        lab_ch = batch["label"].to(device)
        domain = batch["domain"]
        
        with autocast(device_type=str(device)):
            seg_logits = model(img, return_domain_logits=False)
            seg_loss = loss_fn_seg(seg_logits, lab_ch)
        
        running_seg_loss += seg_loss.item()
        preds = seg_logits.softmax(dim=1).argmax(dim=1)
        
        # Separate updates for CT and MRI samples
        is_ct = (domain == 0)
        is_mri = (domain == 1)

        if torch.any(is_ct):
            ct_dice_metric.update(preds[is_ct], lab_ch[is_ct].squeeze(1).long())
        if torch.any(is_mri):
            mri_dice_metric.update(preds[is_mri], lab_ch[is_mri].squeeze(1).long())
            
    avg_loss = running_seg_loss / len(loader)
    ct_dice = ct_dice_metric.compute().item()
    mri_dice = mri_dice_metric.compute().item()
    
    return avg_loss, ct_dice, mri_dice