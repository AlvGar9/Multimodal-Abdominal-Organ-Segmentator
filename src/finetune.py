# src/finetune.py
import argparse
import torch
from torch.cuda.amp import GradScaler

from training import config as cfg
from training.utils import set_seed
from training.data import get_dataloaders
from training.model import build_model, load_pretrained_weights
from training.loss import get_loss_function
from training.optimizer import get_finetune_optimizer_scheduler
from training.engine import train_epoch, validate

def main(args):
    """
    Main script for fine-tuning a pre-trained model.
    """
    set_seed(cfg.SEED)
    
    cfg.NUM_TRAIN_SAMPLES = args.num_samples
    print(f"Initializing fine-tuning for MR data with {cfg.NUM_TRAIN_SAMPLES} samples.")

    # Configure paths for this experiment
    cfg.MODALITY = 'MR'
    cfg.BASE_DATA_DIR = f'{cfg.PREPROCESSED_DATA_DIR}/{cfg.MODALITY}'
    cfg.IMAGE_DIR = f'{cfg.BASE_DATA_DIR}/images'
    cfg.LABEL_DIR = f'{cfg.BASE_DATA_DIR}/labels'
    
    regime_name = "full" if cfg.NUM_TRAIN_SAMPLES == 'all' else cfg.NUM_TRAIN_SAMPLES
    model_filename = f'finetuned_model_mr_{regime_name}_samples.pth'
    cfg.BEST_MODEL_PATH = f'./{model_filename}'
    
    # Initialize components
    train_loader, val_loader, _ = get_dataloaders(cfg)
    model = build_model(cfg).to(cfg.DEVICE)
    load_pretrained_weights(model, cfg.PRETRAINED_CT_PATH, cfg.DEVICE)
    
    loss_fn = get_loss_function(device=cfg.DEVICE, include_background=False)
    optimizer, scheduler = get_finetune_optimizer_scheduler(model, cfg)
    scaler = GradScaler(device_type=str(cfg.DEVICE))

    # Main training loop
    best_metric = -1.0
    stale_epochs = 0

    print(f"Starting fine-tuning for {cfg.FINETUNE_MAX_EPOCHS} epochs.")
    print(f"Best model will be saved to: {cfg.BEST_MODEL_PATH}")
    
    for epoch in range(1, cfg.FINETUNE_MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, cfg.DEVICE)
        scheduler.step()
        
        print(f"Epoch {epoch}/{cfg.FINETUNE_MAX_EPOCHS}, Train Loss: {train_loss:.4f}", end="")

        if epoch % cfg.VAL_INTERVAL == 0:
            val_loss, val_dice = validate(model, val_loader, loss_fn, cfg.NUM_CLASSES, cfg.DEVICE)
            print(f", Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

            if val_dice > best_metric + cfg.MIN_IMPROVEMENT_DELTA:
                print(f"New best metric. Saving model to {cfg.BEST_MODEL_PATH}")
                best_metric = val_dice
                stale_epochs = 0
                torch.save(model.state_dict(), cfg.BEST_MODEL_PATH)
            else:
                stale_epochs += 1
                if stale_epochs >= cfg.EARLY_STOP_PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break
        else:
            print()

    print(f"\nFine-tuning for {regime_name} samples completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained UNet on MRI data.")
    parser.add_argument(
        "num_samples", type=str,
        help="Number of training samples (e.g., 5, 15, 30) or 'all' to use the full training pool."
    )
    args = parser.parse_args()
    main(args)