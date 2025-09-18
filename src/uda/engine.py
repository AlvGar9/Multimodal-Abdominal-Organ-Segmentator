# src/uda/engine.py
import numpy as np
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast

def train_epoch_uda(
    model, loader, loss_fn_seg, loss_fn_align, optimizer, scaler, epoch, max_epochs
):
    """Runs a single UDA training epoch."""
    model.train()
    
    # Define weight schedule for segmentation vs. alignment loss
    if epoch <= max_epochs * 0.2: # First 20%
        seg_weight, align_weight = 0.9, 0.1
    elif epoch <= max_epochs * 0.6: # Next 40%
        seg_weight, align_weight = 0.5, 0.5
    else: # Final 40%
        seg_weight, align_weight = 0.3, 0.7
        
    total_losses, seg_losses, align_losses = [], [], []
    
    for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        ct_img = batch['ct_image'].to(torch.device('cuda'))
        ct_label = batch['ct_label'].to(torch.device('cuda'))
        mri_img = batch['mri_image'].to(torch.device('cuda'))

        combined_img = torch.cat([ct_img, mri_img], dim=0)
        
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            combined_seg, all_features = model.forward_with_features(combined_img)
            
            # Segmentation loss (on source/CT only)
            ct_seg_output, _ = torch.split(combined_seg, ct_img.size(0), dim=0)
            seg_loss = loss_fn_seg(ct_seg_output, ct_label)

            # Alignment loss (between source/CT and target/MRI)
            align_loss = 0
            level_weights = [0.1, 0.2, 0.3, 0.4, 1.0] # Deeper layers get more weight
            
            for i, features in enumerate(all_features):
                ct_feats, mri_feats = torch.split(features, ct_img.size(0), dim=0)
                # Align centroids of the feature batches
                ct_centroid = torch.mean(ct_feats, dim=0, keepdim=True)
                mri_centroid = torch.mean(mri_feats, dim=0, keepdim=True)
                align_loss += loss_fn_align(ct_centroid, mri_centroid) * level_weights[i]

            total_loss = seg_weight * seg_loss + align_weight * align_loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_losses.append(total_loss.item())
        seg_losses.append(seg_loss.item())
        align_losses.append(align_loss.item())
        
    avg_total = np.mean(total_losses)
    avg_seg = np.mean(seg_losses)
    avg_align = np.mean(align_losses)
    
    print(f"Epoch {epoch}: Total Loss={avg_total:.4f}, Seg Loss={avg_seg:.4f}, Align Loss={avg_align:.4f}")
    return avg_total