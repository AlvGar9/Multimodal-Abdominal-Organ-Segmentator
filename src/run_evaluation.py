# src/run_evaluation.py

import os
import glob
import argparse
import pandas as pd
import numpy as np
import torch
from typing import Optional, Dict, Tuple, List
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Lambdad, \
    NormalizeIntensityd, ScaleIntensityRanged, ToTensord, CopyItemsd
from monai.networks.nets import UNet

# Project-specific modules
from training import config as cfg
from dann.model import DANNUNet
from uda.model import MultiLevelUDANet


def get_model_instance(model_name: str, config) -> torch.nn.Module:
    """
    Instantiates the appropriate model architecture based on naming convention.
    
    Model types:
    - Standard UNet: Baseline 3D U-Net architecture
    - DANN UNet: Domain Adversarial Neural Network for domain adaptation
    - UDA Net: Unsupervised Domain Adaptation with multi-level feature alignment
    
    Args:
        model_name (str): Model identifier from config.MODELS_TO_EVALUATE
            Examples: "baseline_UNet", "DANN_CT_to_MR", "UDA_MR_to_CT"
        config: Configuration object containing:
            - NUM_CLASSES: Number of segmentation classes (organs + background)
            - MODEL_CHANNELS: Channel dimensions for each UNet level (e.g., [32, 64, 128, 256])
            - MODEL_STRIDES: Downsampling strides between levels (e.g., [2, 2, 2])
            - DEVICE: torch.device for model placement (cuda/cpu)
    
    Returns:
        torch.nn.Module: Initialized model moved to specified device
    
    Note:
        All models share the same base UNet architecture but differ in
        training methodology and auxiliary components (discriminators, etc.)
    """
    # Initialize base UNet architecture (shared across all model types)
    base_unet = UNet(
        spatial_dims=3,  # 3D volumetric segmentation
        in_channels=1,   # Single channel input (grayscale medical images)
        out_channels=config.NUM_CLASSES,  # Multi-class segmentation
        channels=config.MODEL_CHANNELS,   # Feature channels per level
        strides=config.MODEL_STRIDES,     # Downsampling factors
        num_res_units=2  # Residual units per level for better gradient flow
    )
    
    # Instantiate specialized architectures based on model name
    if "UDA" in model_name:
        # Multi-level UDA: Uses feature alignment at multiple decoder levels
        # Wraps base UNet with domain adaptation components
        return MultiLevelUDANet(pretrained_unet=base_unet).to(config.DEVICE)
    
    if "DANN" in model_name:
        # DANN: Uses gradient reversal for adversarial domain adaptation
        # Note: This appears to use its own internal architecture
        return DANNUNet().to(config.DEVICE)
    
    # Default: Standard UNet without domain adaptation
    return base_unet.to(config.DEVICE)


def get_test_dataloaders(config) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Creates data loaders for three test datasets with standardized preprocessing.
    
    Preprocessing pipeline:
    1. Load NIfTI volumes (image and label)
    2. Ensure channel-first format for PyTorch
    3. Reorient to RAS+ coordinate system (standard medical imaging)
    4. Store raw image copy for visualization
    5. Remap organ labels for consistency
    6. Normalize intensities using non-zero voxels
    7. Clip and scale to [0, 1] range
    8. Convert to PyTorch tensors
    
    Args:
        config: Configuration object containing:
            - PREPROCESSED_DATA_DIR: Root directory for preprocessed AMOS data
            - CHAOS_MRI_IMAGES_DIR: Directory for CHAOS MRI images
            - CHAOS_MRI_LABELS_DIR: Directory for CHAOS MRI labels
            - NUM_WORKERS: Number of parallel data loading workers
    
    Returns:
        Tuple of three DataLoaders:
            - ct_loader: AMOS CT test set (may be None if not found)
            - mri_loader: AMOS MRI test set (may be None if not found)
            - chaos_loader: CHAOS MRI test set (may be None if not found)
    
    Note:
        Label remapping (6→4) handles inconsistencies between datasets
        where organ class indices may differ.
    """
    # Define preprocessing transformations
    transforms = Compose([
        # I/O operations
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        
        # Spatial standardization
        # RAS+ orientation: Right-Anterior-Superior coordinate system
        # Ensures consistent anatomical orientation across datasets
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        
        # Preserve original image for visualization
        # Creates a copy before intensity normalization
        CopyItemsd(keys=['image'], times=1, names=['image_raw_for_vis']),
        
        # Label harmonization
        # Remap class 6 to class 4 to handle dataset-specific label inconsistencies
        # This ensures organ classes align across different annotation protocols
        Lambdad(keys=['label'], func=lambda x: np.where(x == 6, 4, x)),
        
        # Intensity normalization
        # nonzero=True: Calculate statistics only from non-background voxels
        # channel_wise=True: Normalize each channel independently (though we have 1 channel)
        NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
        
        # Intensity clipping and scaling
        # Clips to [-3, 3] standard deviations (removes outliers)
        # Then scales to [0, 1] for neural network input
        ScaleIntensityRanged(
            keys=['image'], 
            a_min=-3,  # Lower bound (in standard deviations)
            a_max=3,   # Upper bound (in standard deviations)
            b_min=0.0, # Target minimum
            b_max=1.0, # Target maximum
            clip=True  # Clip values outside range
        ),
        
        # Convert to PyTorch tensors
        ToTensord(keys=['image', 'label', 'image_raw_for_vis'])
    ])
    
    def create_loader(img_dir: str, lbl_dir: str) -> Optional[DataLoader]:
        """
        Helper function to create a DataLoader from image and label directories.
        
        Args:
            img_dir: Directory containing .nii.gz image files
            lbl_dir: Directory containing corresponding .nii.gz label files
            
        Returns:
            DataLoader or None if directories are empty/missing
        """
        # Find all NIfTI files (both .nii.gz and .nii)
        image_files = sorted(glob.glob(os.path.join(img_dir, '*.nii.gz*')))
        label_files = sorted(glob.glob(os.path.join(lbl_dir, '*.nii.gz*')))
        
        # Create paired image-label dictionaries
        files = [
            {'image': img_path, 'label': lbl_path} 
            for img_path, lbl_path in zip(image_files, label_files)
        ]
        
        # Return None if no files found (handles missing datasets gracefully)
        if not files:
            print(f"Warning: No files found in {img_dir}")
            return None
        
        # Create MONAI Dataset with transforms
        ds = Dataset(data=files, transform=transforms)
        
        # Create DataLoader
        # batch_size=1: Process one volume at a time (memory efficiency)
        # num_workers: Parallel data loading for speed
        return DataLoader(
            ds, 
            batch_size=1, 
            num_workers=config.NUM_WORKERS,
            pin_memory=True if config.DEVICE.type == 'cuda' else False  # GPU optimization
        )

    # Create loaders for each test dataset
    ct_loader = create_loader(
        f"{config.PREPROCESSED_DATA_DIR}/CT/images",
        f"{config.PREPROCESSED_DATA_DIR}/CT/labels"
    )
    
    mri_loader = create_loader(
        f"{config.PREPROCESSED_DATA_DIR}/MR/images",
        f"{config.PREPROCESSED_DATA_DIR}/MR/labels"
    )
    
    chaos_loader = create_loader(
        config.CHAOS_MRI_IMAGES_DIR,
        config.CHAOS_MRI_LABELS_DIR
    )
    
    return ct_loader, mri_loader, chaos_loader


def main(args: argparse.Namespace) -> None:
    """
    Main orchestration function for multi-model evaluation pipeline.
    
    Workflow:
    1. Set up output directories
    2. Initialize test data loaders
    3. Load and evaluate each configured model
    4. Generate per-model and aggregate reports
    5. Display final summary table
    
    Args:
        args: Command-line arguments containing:
            - model (str, optional): Specific model name to evaluate
              If not provided, evaluates all models in config
    
    Outputs:
        Creates the following file structure:
        - evaluation_output/
            - evaluation_summary.csv: Aggregate Dice scores across all models
            - [ModelName]/
                - results_raw_*.csv: Per-patient detailed metrics
                - results_summary_*.csv: Statistical summaries
                - prediction_*.png: Visualization samples
    
    Raises:
        ValueError: If specified model name not found in configuration
    """
    # Delayed import to avoid circular dependencies
    # (evaluate module may import from this module)
    from evaluate import run_inference
    
    # Create main output directory
    os.makedirs(cfg.EVALUATION_OUTPUT_DIR, exist_ok=True)
    print(f"Evaluation output directory: {cfg.EVALUATION_OUTPUT_DIR}")
    
    # Initialize data loaders for all test sets
    print("\nLoading test datasets...")
    ct_loader, mri_loader, chaos_loader = get_test_dataloaders(cfg)
    
    # Determine which models to evaluate
    if args.model:
        # Single model evaluation mode
        models_to_run = {
            k: v for k, v in cfg.MODELS_TO_EVALUATE.items() 
            if k == args.model
        }
        if not models_to_run:
            raise ValueError(
                f"Model '{args.model}' not found in config.MODELS_TO_EVALUATE.\n"
                f"Available models: {list(cfg.MODELS_TO_EVALUATE.keys())}"
            )
    else:
        # Evaluate all configured models
        models_to_run = cfg.MODELS_TO_EVALUATE
    
    print(f"\nModels to evaluate: {list(models_to_run.keys())}")
    
    # Store aggregate results for final comparison
    all_results = []
    
    # ============ Main Evaluation Loop ============
    for model_name, model_path in models_to_run.items():
        print(f"\n{'='*80}")
        print(f"EVALUATING MODEL: {model_name}")
        print(f"{'='*80}")
        
        # Verify model checkpoint exists
        if not os.path.exists(model_path):
            print(f" Warning: Model checkpoint not found: {model_path}")
            print(f"  Skipping {model_name}")
            continue
        
        # Create model-specific output directory
        model_output_dir = os.path.join(cfg.EVALUATION_OUTPUT_DIR, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Load model architecture and weights
        print(f"Loading model from: {model_path}")
        model = get_model_instance(model_name, cfg)
        
        # Load trained weights
        # map_location ensures weights are loaded to correct device
        checkpoint = torch.load(model_path, map_location=cfg.DEVICE, weights_only=False)
        # Check if the checkpoint is a dictionary containing the state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded successfully")
        
        # ============ Run Evaluation on Each Dataset ============
        # CT AMOS Dataset
        if ct_loader:
            print(f"\nEvaluating on CT AMOS dataset...")
            summary_ct = run_inference(
                model, ct_loader, cfg, "CT_AMOS", model_output_dir
            )
        else:
            print("CT AMOS dataset not available")
            summary_ct = None
        
        # MRI AMOS Dataset
        if mri_loader:
            print(f"\nEvaluating on MR AMOS dataset...")
            summary_mri = run_inference(
                model, mri_loader, cfg, "MR_AMOS", model_output_dir
            )
        else:
            print("MR AMOS dataset not available")
            summary_mri = None
        
        # CHAOS MRI Dataset (optional external validation)
        if chaos_loader:
            print(f"\nEvaluating on MR CHAOS dataset...")
            summary_chaos = run_inference(
                model, chaos_loader, cfg, "MR_CHAOS", model_output_dir
            )
        else:
            print("MR CHAOS dataset not available (optional)")
            summary_chaos = None
        
        # ============ Aggregate Metrics ============
        # Calculate mean Dice scores across all organs for each dataset
        mean_dice = {
            'Model': model_name,
            'CT_Dice': np.nan,
            'MR_Dice': np.nan,
            'CHAOS_Dice': np.nan,
        }
        
        # Extract mean Dice scores from summaries
        # Filter rows containing 'Dice' and compute their mean
        if summary_ct is not None:
            dice_rows = summary_ct.index.str.contains('Dice')
            mean_dice['CT_Dice'] = summary_ct.loc[dice_rows, 'mean'].mean()
        
        if summary_mri is not None:
            dice_rows = summary_mri.index.str.contains('Dice')
            mean_dice['MR_Dice'] = summary_mri.loc[dice_rows, 'mean'].mean()
        
        if summary_chaos is not None:
            dice_rows = summary_chaos.index.str.contains('Dice')
            mean_dice['CHAOS_Dice'] = summary_chaos.loc[dice_rows, 'mean'].mean()
        
        all_results.append(mean_dice)

    # ============ Generate Final Summary Report ============
    if all_results:
        # Create summary DataFrame with all models' performance
        summary_df = pd.DataFrame(all_results)
        
        # Save to CSV with 4 decimal precision
        summary_path = os.path.join(cfg.EVALUATION_OUTPUT_DIR, "evaluation_summary.csv")
        summary_df.to_csv(
            summary_path, 
            index=False, 
            float_format='%.4f'  # 4 decimal places for precision
        )
        
        # Display final results table
        print("\n" + "="*80)
        print("FINAL EVALUATION SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print(f"\n Summary saved to: {summary_path}")
    else:
        print("\n No models were successfully evaluated")


if __name__ == '__main__':
    """
    Command-line interface for the evaluation pipeline.
    
    Examples:
        # Evaluate all models
        python run_evaluation.py
        
        # Evaluate specific model
        python run_evaluation.py --model "UDA_CT_to_MR"
    """
    parser = argparse.ArgumentParser(
        description="Multi-model medical image segmentation evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py                    # Evaluate all configured models
  python run_evaluation.py --model UDA_Net    # Evaluate specific model only
        """
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        help="Specific model name to evaluate (from config.MODELS_TO_EVALUATE). "
             "If not provided, all configured models will be evaluated.",
        default=None
    )
    
    args = parser.parse_args()
    main(args)