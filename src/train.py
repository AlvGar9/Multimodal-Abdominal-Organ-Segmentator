# src/train.py
import argparse
import torch
from torch.optim.lr_scheduler import PolynomialLR
from torch.cuda.amp import GradScaler

# Import from our training module
from training import config as cfg
from training.utils import set_seed
from training.data import get_dataloaders
from training.model import build_model
from training.loss import calculate_loss_weights, get_loss_function
from training.engine import train_epoch, validate

def main(args):
    """
    Main training script.
    """
    # -- Setup --
    set_seed(cfg.SEED)
    
    # Store CLI args in the config object for easy access
    cfg.MODALITY = (args.modality or cfg.MODALITY).upper()
    cfg.NUM_TRAIN_SAMPLES = args.num_samples
    
    print(f"Modality set to: {cfg.MODALITY}")
    if cfg.NUM_TRAIN_SAMPLES:
        print(f"Number of training samples set to: {cfg.NUM_TRAIN_SAMPLES}")

    # We update the data paths based on modality
    cfg.BASE_DATA_DIR = f'./preprocessed_data/AMOS/{cfg.MODALITY}'
    cfg.IMAGE_DIR = f'{cfg.BASE_DATA_DIR}/images'
    cfg.LABEL_DIR = f'{cfg.BASE_DATA_DIR}/labels'
    
    # Create a unique model filename for this specific experiment
    if cfg.NUM_TRAIN_SAMPLES:
        model_filename = f'best_model_{cfg.MODALITY.lower()}_{cfg.NUM_TRAIN_SAMPLES}_samples.pth'
    else:
        model_filename = f'best_model_{cfg.MODALITY.lower()}_full.pth'
    cfg.BEST_MODEL_PATH = f'./{model_filename}'

    # -- Data --
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # -- Model, Loss, Optimizer --
    model = build_model(cfg).to(cfg.DEVICE)
    ce_weights, dice_weights = calculate_loss_weights(train_loader, cfg.NUM_CLASSES)
    loss_fn = get_loss_function(ce_weights, dice_weights, cfg.DEVICE)
    
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.INITIAL_LR, momentum=0.99,
        weight_decay=cfg.WEIGHT_DECAY, nesterov=True
    )
    scheduler = PolynomialLR(optimizer, total_iters=cfg.MAX_EPOCHS, power=0.9)
    scaler = GradScaler(device_type=str(cfg.DEVICE))

    # -- Training Loop --
    best_metric = -1.0
    stale_epochs = 0

    print(f"Starting training for {cfg.MAX_EPOCHS} epochs on device: {cfg.DEVICE}")
    print(f"Best model will be saved to: {cfg.BEST_MODEL_PATH}")
    print("="*60)

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, cfg.DEVICE)
        scheduler.step()
        
        print(f"Epoch {epoch}/{cfg.MAX_EPOCHS} | Train Loss: {train_loss:.4f}", end='')

        if epoch % cfg.VAL_INTERVAL == 0:
            val_loss, val_dice = validate(model, val_loader, loss_fn, cfg.NUM_CLASSES, cfg.DEVICE)
            print(f" | Val Loss: {val_loss:.4f} | Mean FG Dice: {val_dice:.4f}")

            # Checkpointing and Early Stopping
            if val_dice > best_metric + cfg.MIN_IMPROVEMENT_DELTA:
                print(f"  🎉 New best metric! Saving model to {cfg.BEST_MODEL_PATH}")
                best_metric = val_dice
                stale_epochs = 0
                torch.save(model.state_dict(), cfg.BEST_MODEL_PATH)
            else:
                stale_epochs += 1
                print(f"  ⚠️ No improvement ({stale_epochs}/{cfg.EARLY_STOP_PATIENCE})")

            if stale_epochs >= cfg.EARLY_STOP_PATIENCE:
                print(f"\n🏁 Early stopping triggered at epoch {epoch}.")
                break
        else:
            print() # Newline for epochs without validation, cleaner look to be honest

    print("\nTraining completed!")
    print(f"Best Mean Foreground Dice: {best_metric:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a 3D UNet for organ segmentation.")
    parser.add_argument(
        "--modality", type=str, choices=['MR', 'CT'],
        help="Specify the imaging modality to train on (MR or CT)."
    )
    parser.add_argument(
        "--num_samples", type=str,
        help="Specify number of training samples (e.g., 5, 15, 30) or 'all' to use train+val."
    )
    args = parser.parse_args()
    main(args)