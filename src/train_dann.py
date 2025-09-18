# src/train_dann.py
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

# DANN-specific imports
from dann.data import get_dann_dataloaders
from dann.model import DANNUNet, load_unet_weights
from dann.engine import get_lambda_p, train_epoch_dann, validate_dann

# Reusable imports from standard training
from training import config as cfg
from training.utils import set_seed
from training.loss import calculate_loss_weights, get_loss_function
from training.optimizer import get_scratch_optimizer_scheduler

def main(args):
    """Main script for DANN experiments."""
    set_seed(cfg.SEED)
    num_mri = args.num_mri_samples
    
    # --- Data ---
    train_loader, val_loader, _, _ = get_dann_dataloaders(cfg, num_mri) # Test loaders are not needed here
    
    # --- Model ---
    model = DANNUNet().to(cfg.DEVICE)
    load_unet_weights(model, cfg.PRETRAINED_CT_PATH, cfg.DEVICE)
    
    # --- Loss, Optimizer, Scheduler ---
    ce_weights, dice_weights = calculate_loss_weights(train_loader, cfg.NUM_CLASSES)
    loss_fn_seg = get_loss_function(
        cfg.DEVICE, ce_weights, dice_weights, include_background=True
    )
    loss_fn_domain = nn.CrossEntropyLoss()
    optimizer, scheduler = get_scratch_optimizer_scheduler(model, cfg)
    scaler = GradScaler(device_type=str(cfg.DEVICE))

    # --- Training Loop ---
    best_metric = -1.0
    stale_epochs = 0
    model_path = f'./dann_model_{num_mri}_mri.pth'
    
    print(f"Starting DANN training with {num_mri} MRI samples.")
    print(f"Model checkpoints will be saved to {model_path}")
    
    for epoch in range(1, cfg.DANN_MAX_EPOCHS + 1):
        lambda_p = get_lambda_p(epoch, cfg.DANN_MAX_EPOCHS)
        seg_loss, dom_loss = train_epoch_dann(
            model, train_loader, optimizer, loss_fn_seg, loss_fn_domain, 
            scaler, cfg.DEVICE, lambda_p
        )
        scheduler.step()
        
        print(f"Epoch {epoch}/{cfg.DANN_MAX_EPOCHS} | SegLoss: {seg_loss:.4f} | DomLoss: {dom_loss:.4f}", end="")
        
        if epoch % cfg.VAL_INTERVAL == 0:
            val_loss, ct_dice, mri_dice = validate_dann(
                model, val_loader, loss_fn_seg, cfg.DEVICE, cfg.NUM_CLASSES
            )
            # Use average validation Dice across domains as the metric for checkpointing
            val_metric = (ct_dice + mri_dice) / 2
            
            print(f" | ValLoss: {val_loss:.4f} | Val CT Dice: {ct_dice:.4f} | Val MRI Dice: {mri_dice:.4f}")

            if val_metric > best_metric + cfg.MIN_IMPROVEMENT_DELTA:
                print(f"New best validation metric. Saving model to {model_path}")
                best_metric = val_metric
                stale_epochs = 0
                torch.save(model.state_dict(), model_path)
            else:
                stale_epochs += 1
                if stale_epochs >= cfg.EARLY_STOP_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break
        else:
            print()
            
    print(f"\nTraining for {num_mri} MRI samples completed. Best model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DANN UNet for domain adaptation.")
    parser.add_argument(
        "num_mri_samples", type=str,
        help="Number of MRI samples for the target domain (e.g., 5, 15, 30) or 'all'."
    )
    args = parser.parse_args()
    main(args)