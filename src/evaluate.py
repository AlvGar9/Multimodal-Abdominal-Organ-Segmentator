# src/evaluate.py

import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff


def find_best_slice(label_volume: np.ndarray) -> int:
    """
    Identifies the most informative axial slice for visualization purposes.
    
    Selection criteria prioritizes slices with:
    1. Maximum number of distinct organs (diversity)
    2. Largest total segmentation area (content richness)
    
    Args:
        label_volume (np.ndarray): 3D label volume with shape (H, W, D) where:
            - H, W: spatial dimensions (height, width)
            - D: depth/number of axial slices
            - Values: 0 for background, 1-N for organ classes
    
    Returns:
        int: Index of the selected slice (0 to D-1)
    
    Example:
        >>> label_vol = np.random.randint(0, 5, size=(256, 256, 100))
        >>> best_idx = find_best_slice(label_vol)
        >>> print(f"Best slice for visualization: {best_idx}")
    """
    num_slices = label_volume.shape[2]
    slice_scores = []
    
    # Evaluate each axial slice
    for i in range(num_slices):
        slice_2d = label_volume[:, :, i]
        
        # Count non-background voxels (total segmentation area)
        voxel_count = np.sum(slice_2d > 0)
        
        # Count unique organ labels (excluding background)
        num_organs = len(np.unique(slice_2d[slice_2d > 0]))
        
        slice_scores.append({
            'slice_idx': i, 
            'num_organs': num_organs, 
            'voxel_count': voxel_count
        })
    
    # Handle edge case: no segmentation in entire volume
    if not any(s['voxel_count'] > 0 for s in slice_scores):
        # Return middle slice as fallback
        return num_slices // 2
    
    # Sort by: 1) number of organs (primary), 2) voxel count (secondary)
    # This ensures we get diverse, content-rich slices for visualization
    best_slice = sorted(
        slice_scores, 
        key=lambda x: (x['num_organs'], x['voxel_count']), 
        reverse=True
    )[0]['slice_idx']
    
    return best_slice


def visualize_prediction(batch, prediction, sample_idx, dataset_name, output_dir, config):
    """
    Creates a comprehensive 3-panel visualization comparing ground truth with predictions.
    
    Generates a figure with:
    - Panel 1: Ground truth segmentation overlay
    - Panel 2: Model prediction overlay
    - Panel 3: Error map showing TP/FP/FN regions
    
    Args:
        batch (dict): Data batch containing:
            - 'image_raw_for_vis': Original image tensor [B, C, H, W, D]
            - 'label': Ground truth segmentation [B, C, H, W, D]
        prediction (torch.Tensor): Model output segmentation [B, C, H, W, D]
        sample_idx (int): Index of current sample (for filename)
        dataset_name (str): Name of evaluation dataset
        output_dir (str): Directory to save visualization
        config: Configuration object with:
            - NUM_CLASSES: Total number of segmentation classes
            - ORGAN_NAMES: Mapping of class indices to organ names
    
    Note:
        Images are rotated and flipped for radiological viewing convention
        (as if viewing patient from feet upward)
    """
    # Extract numpy arrays from tensors (first batch, first channel)
    img_np = batch['image_raw_for_vis'][0, 0].cpu().numpy()
    lab_np = batch['label'][0, 0].cpu().numpy()
    pred_np = prediction[0, 0].cpu().numpy()

    # Select most informative slice for visualization
    slice_idx = find_best_slice(lab_np)
    img_slice = img_np[:, :, slice_idx]
    lab_slice = lab_np[:, :, slice_idx]
    pred_slice = pred_np[:, :, slice_idx]

    # ============ Color Mapping Setup ============
    # Create colormap for organ classes using viridis
    cmap = plt.cm.get_cmap('viridis', config.NUM_CLASSES)
    colors = cmap(np.linspace(0, 1, config.NUM_CLASSES))
    
    # Make background (class 0) transparent for overlay visualization
    colors[0, -1] = 0  # Set alpha channel to 0 for background
    custom_cmap = mcolors.ListedColormap(colors)

    # ============ Error Map Calculation ============
    # Create error map to highlight segmentation mistakes
    # Values: 0=background, 1=TP (correct), 2=FN (missed), 3=FP (false alarm)
    error_map = np.zeros_like(lab_slice)
    
    # True Positive: Both ground truth and prediction have segmentation
    error_map[(lab_slice > 0) & (pred_slice > 0)] = 1
    
    # False Negative: Ground truth has segmentation but prediction missed it
    error_map[(lab_slice > 0) & (pred_slice == 0)] = 2
    
    # False Positive: Prediction has segmentation but ground truth doesn't
    error_map[(lab_slice == 0) & (pred_slice > 0)] = 3
    
    # Define error visualization colors
    error_colors = [
        'black',    # 0: Background (not shown)
        '#2ca02c',  # 1: True Positive (green)
        '#1f77b4',  # 2: False Negative (blue) 
        '#d62728'   # 3: False Positive (red)
    ]
    error_cmap = mcolors.ListedColormap(error_colors)
    error_norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4], error_cmap.N)
    
    # ============ Figure Creation ============
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        f"Qualitative Result on {dataset_name} - Sample {sample_idx+1}", 
        fontsize=18
    )

    def format_ax(ax, title, image):
        """Helper function to standardize axis formatting."""
        # Apply radiological viewing convention: rotate 90° and flip horizontally
        # This shows the image as if viewing patient from feet upward
        ax.imshow(np.fliplr(np.rot90(image, k=1)), cmap="gray")
        ax.set_title(title, fontsize=16)
        ax.axis('off')

    # Panel 1: Ground Truth Segmentation
    format_ax(axes[0], "Ground Truth", img_slice)
    # Overlay segmentation with transparency, masking background pixels
    axes[0].imshow(
        np.fliplr(np.rot90(np.ma.masked_where(lab_slice == 0, lab_slice), k=1)),
        cmap=custom_cmap, 
        alpha=0.7,  # 70% opacity for overlay
        vmin=0, 
        vmax=config.NUM_CLASSES-1
    )

    # Panel 2: Model Prediction
    format_ax(axes[1], "Model Prediction", img_slice)
    axes[1].imshow(
        np.fliplr(np.rot90(np.ma.masked_where(pred_slice == 0, pred_slice), k=1)),
        cmap=custom_cmap, 
        alpha=0.7,
        vmin=0, 
        vmax=config.NUM_CLASSES-1
    )
    
    # Panel 3: Error Map Visualization
    format_ax(axes[2], "Segmentation Error Map", img_slice)
    axes[2].imshow(
        np.fliplr(np.rot90(error_map, k=1)), 
        cmap=error_cmap, 
        norm=error_norm, 
        alpha=0.8  # 80% opacity to see underlying anatomy
    )

    # Add legend for error map interpretation
    legend_elements = [
        Patch(facecolor=c, label=l) for c, l in 
        zip(error_colors[1:], ['True Positive', 'False Negative', 'False Positive'])
    ]
    fig.legend(
        handles=legend_elements, 
        loc='lower center', 
        ncol=3, 
        bbox_to_anchor=(0.5, 0.02)
    )
    
    # Adjust layout to prevent overlap and save high-resolution image
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Leave space for legend
    output_path = os.path.join(output_dir, f"prediction_{dataset_name}_{sample_idx+1}.png")
    plt.savefig(output_path, dpi=300)  # High DPI for publication quality
    plt.close(fig)  # Free memory


@torch.no_grad()
def run_inference(model, loader, config, dataset_name, output_dir):
    """
    Performs comprehensive evaluation of a segmentation model on a dataset.
    
    Computes per-organ metrics:
    - Dice Similarity Coefficient (DSC): Measures overlap (2*TP/(2*TP+FP+FN))
    - Intersection over Union (IoU): Jaccard index (TP/(TP+FP+FN))
    - Hausdorff Distance 95th percentile (HD95): Surface distance metric
    
    Args:
        model (torch.nn.Module): Trained segmentation model
        loader (DataLoader): PyTorch dataloader with evaluation data
        config: Configuration object containing:
            - DEVICE: torch device (cuda/cpu)
            - NUM_CLASSES: Number of segmentation classes (including background)
            - ORGAN_NAMES: List mapping class indices to organ names
        dataset_name (str): Name for results files
        output_dir (str): Directory to save results and visualizations
    
    Returns:
        pd.DataFrame: Summary statistics (mean, std) for all metrics
    
    Outputs:
        - CSV file with per-patient metrics
        - CSV file with summary statistics
        - Visualization images for first 3 samples
    
    Example:
        >>> model = load_model('checkpoint.pth')
        >>> loader = get_dataloader('test_data/')
        >>> summary = run_inference(model, loader, config, 'TestSet', './results/')
    """
    model.eval()  # Set model to evaluation mode (disables dropout, batch norm updates)
    results = []
    
    # Process each patient/volume in the dataset
    for i, batch in enumerate(tqdm(loader, desc=f"Evaluating on {dataset_name}")):
        # Move data to computation device (GPU/CPU)
        img = batch['image'].to(config.DEVICE)
        lab = batch['label'].to(config.DEVICE)
        
        # ============ Time Inference ============
        start_time = time.time()
        
        # Forward pass: get model predictions
        # Model outputs logits [B, C, H, W, D], argmax to get class predictions
        pred = torch.argmax(model(img), dim=1, keepdim=True)
        
        # Ensure GPU operations complete before timing (for accurate measurement)
        if config.DEVICE.type == 'cuda':
            torch.cuda.synchronize()
            
        inference_time = time.time() - start_time
        
        # Extract patient identifier from metadata
        patient_id = os.path.basename(batch['image_meta_dict']['filename_or_obj'][0])
        patient_metrics = {
            'id': patient_id, 
            'inference_time': inference_time
        }
        
        # Convert to numpy for metric computation
        pred_np = pred.cpu().numpy()
        lab_np = lab.cpu().numpy()

        # ============ Compute Per-Organ Metrics ============
        for c in range(1, config.NUM_CLASSES):  # Skip background (class 0)
            # Create binary masks for current organ class
            pred_c = (pred == c)  # Binary mask for predictions
            lab_c = (lab == c)    # Binary mask for ground truth
            
            # Dice Similarity Coefficient (DSC)
            # Formula: 2 * |A ∩ B| / (|A| + |B|)
            # Add epsilon (1e-8) to prevent division by zero
            intersection = (pred_c * lab_c).sum()
            dice = (2. * intersection) / (pred_c.sum() + lab_c.sum() + 1e-8)
            
            # Intersection over Union (IoU / Jaccard Index)
            # Formula: |A ∩ B| / |A ∪ B|
            union = (pred_c | lab_c).sum()
            iou = intersection / (union + 1e-8)
            
            # Hausdorff Distance (HD95)
            # Measures maximum surface distance between prediction and ground truth
            # Find all voxel coordinates for this organ class
            pts_pred = np.argwhere(pred_np == c)
            pts_true = np.argwhere(lab_np == c)
            
            # Handle empty segmentations (no organ present)
            if pts_pred.shape[0] == 0 or pts_true.shape[0] == 0:
                hd = np.nan  # Cannot compute distance if one set is empty
            else:
                # Compute bidirectional Hausdorff distance (symmetric)
                # Takes maximum of distances in both directions
                hd_forward = directed_hausdorff(pts_pred, pts_true)[0]
                hd_backward = directed_hausdorff(pts_true, pts_pred)[0]
                hd = max(hd_forward, hd_backward)

            # Store metrics for this organ
            organ_name = config.ORGAN_NAMES[c]
            patient_metrics[f'Dice_{organ_name}'] = dice.item()
            patient_metrics[f'IoU_{organ_name}'] = iou.item()
            patient_metrics[f'HD95_{organ_name}'] = hd

        results.append(patient_metrics)
        
        # Generate visualizations for first 3 samples
        # Limit to 3 to balance insight with computational cost
        if i < 3:
            visualize_prediction(batch, pred, i, dataset_name, output_dir, config)
    
    # ============ Compile and Export Results ============
    # Create DataFrame with all patient results
    df = pd.DataFrame(results)
    
    # Save raw per-patient results
    raw_results_path = os.path.join(output_dir, f"results_raw_{dataset_name}.csv")
    df.to_csv(raw_results_path, index=False)
    
    # Compute summary statistics (mean, std) across all patients
    # Exclude 'id' column from statistical analysis
    summary = df.drop(columns='id').agg(['mean', 'std']).T
    
    # Save summary statistics
    summary_path = os.path.join(output_dir, f"results_summary_{dataset_name}.csv")
    summary.to_csv(summary_path)
    
    print(f"✓ Results for {dataset_name} saved to {output_dir}")
    print(f"  - Raw results: {raw_results_path}")
    print(f"  - Summary stats: {summary_path}")
    
    return summary