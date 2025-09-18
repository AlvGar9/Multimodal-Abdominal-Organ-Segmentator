# src/train_uda.py
import argparse
import torch
from torch.cuda.amp import GradScaler
from monai.networks.nets import UNet

from training import config as cfg
from training.utils import set_seed
from training.loss import get_loss_function
from uda.data import get_uda_dataloaders
from uda.model import MultiLevelUDANet
from uda.loss import AmplifiedCosineLoss
from uda.engine import train_epoch_uda
from visualize import plot_tsne_features

def main(args):
    """Main script for UDA experiments."""
    set_seed(cfg.SEED)
    num_mri = args.num_mri_samples
    
    # --- Data ---
    train_loader, ct_test_loader, mri_test_loader = get_uda_dataloaders(cfg, num_mri)
    
    # --- Model ---
    base_unet = UNet(
        spatial_dims=3, in_channels=1, out_channels=cfg.NUM_CLASSES,
        channels=cfg.MODEL_CHANNELS, strides=cfg.MODEL_STRIDES, num_res_units=2
    ).to(cfg.DEVICE)
    
    checkpoint = torch.load(cfg.PRETRAINED_CT_PATH, map_location=cfg.DEVICE)
    base_unet.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    
    model = MultiLevelUDANet(base_unet, feature_dims=32).to(cfg.DEVICE)
    
    # --- Loss, Optimizer, Scheduler ---
    loss_fn_seg = get_loss_function(cfg.DEVICE, include_background=True) # Standard seg loss
    loss_fn_align = AmplifiedCosineLoss(amplification_power=0.25)
    
    optimizer = torch.optim.AdamW([
        {'params': model.unet.parameters(), 'lr': 1e-6},
        {'params': model.projectors.parameters(), 'lr': 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
    scaler = GradScaler()
    
    # --- Visualization (Before Training) ---
    plot_tsne_features(
        model, ct_test_loader, mri_test_loader,
        title=f"Features Before Training ({num_mri} MRI Samples)",
        output_path=f'./tsne_before_uda_{num_mri}_mri.png'
    )
    
    # --- Training Loop ---
    max_epochs = 50
    print(f"Starting UDA training for {max_epochs} epochs.")
    for epoch in range(1, max_epochs + 1):
        train_epoch_uda(
            model, train_loader, loss_fn_seg, loss_fn_align,
            optimizer, scaler, epoch, max_epochs
        )
        scheduler.step()
        
    # --- Save Final Model ---
    model_path = f'./uda_model_{num_mri}_mri.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining complete. Final model saved to {model_path}")
    
    # --- Visualization (After Training) ---
    plot_tsne_features(
        model, ct_test_loader, mri_test_loader,
        title=f"Features After Training ({num_mri} MRI Samples)",
        output_path=f'./tsne_after_uda_{num_mri}_mri.png'
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a UDA model with multi-level feature alignment.")
    parser.add_argument(
        "num_mri_samples", type=str,
        help="Number of MRI samples for the target domain (e.g., 5, 15, 30, 54)."
    )
    args = parser.parse_args()
    main(args)