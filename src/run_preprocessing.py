#!/usr/bin/env python3

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
from PIL import Image
import nibabel as nib

from preprocessing.utils import save_nifti
from preprocessing.resample import resample_image
from preprocessing.resize import resize_image

# ============ Configuration Constants ============

# Target volume dimensions after preprocessing
# 192³ provides balance between computational efficiency and spatial detail
# Powers of 2 (192 = 3 * 64) work well with U-Net architectures
TARGET_SIZE = (192, 192, 192)

# Target organ labels to retain (AMOS annotation scheme)
# Other organs in original data are filtered out to focus training
TARGET_LABELS = {
    1,  # Spleen
    2,  # Right Kidney
    3,  # Left Kidney  
    6,  # Liver (largest abdominal organ, critical for many applications)
}

# CHAOS dataset uses different label values for the same organs
# This mapping converts CHAOS labels to AMOS-compatible labels
CHAOS_LABEL_MAP = {
    63: 6,   # Liver (CHAOS) -> Liver (AMOS)
    126: 2,  # Right Kidney (CHAOS) -> Right Kidney (AMOS)
    189: 3,  # Left Kidney (CHAOS) -> Left Kidney (AMOS)
    252: 1,  # Spleen (CHAOS) -> Spleen (AMOS)
}


def safe_read_image(path: str) -> sitk.Image:
    """
    Robustly read NIfTI images with fallback for non-standard headers.
    
    Some medical images have non-standard or corrupted headers that cause
    SimpleITK to fail. This function provides a fallback using nibabel
    which is more tolerant of header issues.
    
    Args:
        path (str): Path to NIfTI file (.nii or .nii.gz)
    
    Returns:
        sitk.Image: Loaded image with proper spacing and orientation metadata
    
    Note:
        The fallback method reconstructs basic metadata (spacing, origin, direction)
        from nibabel's header information when SimpleITK fails.
    """
    try:
        # Primary method: SimpleITK (faster, preserves all metadata)
        return sitk.ReadImage(path)
    except RuntimeError:
        # Fallback method: nibabel (more robust for problematic files)
        print(f"Warning: Using fallback loader for {path}")
        
        # Load with nibabel
        nb_img = nib.load(path)
        
        # Extract image array and convert to float32 for processing
        arr = nb_img.get_fdata().astype(np.float32)
        
        # Create SimpleITK image from array
        img = sitk.GetImageFromArray(arr)
        
        # Reconstruct spatial metadata from nibabel header
        # get_zooms() returns voxel dimensions in mm
        zooms = nb_img.header.get_zooms()
        img.SetSpacing([float(z) for z in zooms[:3]])  # Use only spatial dimensions
        
        # Set default origin at (0,0,0) - corner of image volume
        img.SetOrigin([0.0, 0.0, 0.0])
        
        # Set default direction to identity matrix (no rotation)
        # This assumes standard radiological orientation
        img.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        
        return img


def filter_mask(mask_img: sitk.Image) -> sitk.Image:
    """
    Filter segmentation mask to retain only target organ labels.
    
    Removes annotations for organs not in our target set, setting their
    voxels to background (0). This focuses the model on learning specific
    organs relevant to the clinical application.
    
    Args:
        mask_img (sitk.Image): Original segmentation mask with all organ labels
    
    Returns:
        sitk.Image: Filtered mask containing only target organ labels
    
    Example:
        Input mask may contain: [0, 1, 2, 3, 4, 5, 6, 7, 8] (9 organs)
        Output mask will contain: [0, 1, 2, 3, 6] (4 target organs + background)
    """
    # Convert to numpy for efficient array operations
    arr = sitk.GetArrayFromImage(mask_img)
    
    # Create filtered array: keep target labels, set others to 0 (background)
    # np.isin creates boolean mask for values in TARGET_LABELS
    # np.where applies conditional: if in target set, keep value; else set to 0
    filtered = np.where(
        np.isin(arr, list(TARGET_LABELS)), 
        arr, 
        0
    ).astype(np.uint8)  # uint8 sufficient for label indices
    
    # Convert back to SimpleITK image
    out_img = sitk.GetImageFromArray(filtered)
    
    # Preserve spatial metadata (spacing, origin, direction)
    out_img.CopyInformation(mask_img)
    
    return out_img


def load_dicom_series(dicom_dir: Path) -> sitk.Image:
    """
    Load a 3D volume from a directory of DICOM slices.
    
    DICOM is the standard format for medical imaging. Each slice is stored
    as a separate file, and this function reconstructs the 3D volume.
    
    Args:
        dicom_dir (Path): Directory containing DICOM files (.dcm)
    
    Returns:
        sitk.Image: Reconstructed 3D volume with proper spatial information
    
    Note:
        Uses GDCM (Grassroots DICOM) library for robust DICOM handling.
        Automatically sorts slices by position to ensure correct ordering.
    """
    reader = sitk.ImageSeriesReader()
    
    # Get sorted list of DICOM files in the directory
    # GDCM handles various DICOM naming conventions and sorts by slice position
    series_ids = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    
    reader.SetFileNames(series_ids)
    
    # Execute reading and volume reconstruction
    return reader.Execute()


def load_mask_png_series(png_dir: Path, reference: sitk.Image) -> sitk.Image:
    """
    Load segmentation mask from a series of PNG images and align with reference volume.
    
    Some datasets (like CHAOS) provide segmentation masks as 2D PNG slices rather
    than 3D volumes. This function reconstructs the 3D mask and ensures dimensional
    alignment with the corresponding image volume.
    
    Args:
        png_dir (Path): Directory containing PNG mask slices
        reference (sitk.Image): Reference image volume for spatial alignment
    
    Returns:
        sitk.Image: 3D mask volume aligned with reference image
    
    Note:
        Handles dimension mismatches through padding or cropping to ensure
        mask and image volumes have identical dimensions.
    """
    # Load all PNG files in sorted order (ensures correct slice ordering)
    files = sorted(png_dir.glob('*.png'))
    slices = []
    
    for png_path in files:
        # Load as grayscale ('L' mode) since masks are single-channel
        arr = np.array(Image.open(png_path).convert('L'))
        slices.append(arr.astype(np.uint8))
    
    # Stack 2D slices into 3D volume
    # axis=0 stacks along the depth dimension
    vol = np.stack(slices, axis=0)
    
    # ============ Dimension Alignment ============
    # Ensure mask dimensions match reference image exactly
    # This handles cases where mask and image were created with different protocols
    
    # Get reference dimensions (SimpleITK uses x,y,z order)
    ref_size = reference.GetSize()
    ref_width, ref_height, ref_depth = ref_size[0], ref_size[1], ref_size[2]
    
    # Get mask dimensions (numpy uses z,y,x order)
    mask_depth, mask_height, mask_width = vol.shape
    
    # Adjust depth (z-axis) - number of slices
    if mask_depth < ref_depth:
        # Pad with zeros (background) at the end
        pad_amount = ref_depth - mask_depth
        vol = np.pad(vol, ((0, pad_amount), (0, 0), (0, 0)), 
                     mode='constant', constant_values=0)
    elif mask_depth > ref_depth:
        # Crop excess slices
        vol = vol[:ref_depth, :, :]
    
    # Adjust height (y-axis) - anterior-posterior dimension
    if mask_height < ref_height:
        pad_amount = ref_height - mask_height
        vol = np.pad(vol, ((0, 0), (0, pad_amount), (0, 0)), 
                     mode='constant', constant_values=0)
    elif mask_height > ref_height:
        vol = vol[:, :ref_height, :]
    
    # Adjust width (x-axis) - left-right dimension
    if mask_width < ref_width:
        pad_amount = ref_width - mask_width
        vol = np.pad(vol, ((0, 0), (0, 0), (0, pad_amount)), 
                     mode='constant', constant_values=0)
    elif mask_width > ref_width:
        vol = vol[:, :, :ref_width]
    
    # Convert to SimpleITK image
    mask = sitk.GetImageFromArray(vol)
    
    # Copy spatial metadata from reference to ensure alignment
    mask.CopyInformation(reference)
    
    return mask


def preprocess_image(img: sitk.Image, modality: str = "CT") -> sitk.Image:
    """
    Apply standard preprocessing pipeline to medical images.
    
    Standardizes images to consistent resolution and size for neural network input.
    The 1mm³ spacing ensures anatomical structures have consistent physical dimensions
    across different scanners and protocols.
    
    Args:
        img (sitk.Image): Original medical image
        modality (str): Imaging modality ('CT' or 'MR') for modality-specific processing
    
    Returns:
        sitk.Image: Preprocessed image with 1mm³ spacing and 192³ dimensions
    
    Note:
        Future versions may include modality-specific intensity normalization
        (e.g., Hounsfield unit windowing for CT, bias field correction for MR)
    """
    # Step 1: Resample to isotropic 1mm³ voxel spacing
    # This ensures consistent physical dimensions across different scanners
    # 1mm provides good balance between detail and computational cost
    img = resample_image(img, (1.0, 1.0, 1.0), is_mask=False)
    
    # Step 2: Resize to fixed dimensions for batch processing
    # 192³ works well with U-Net architectures (divisible by 2^n for n encoder levels)
    img = resize_image(img, TARGET_SIZE, is_mask=False)
    
    return img


def preprocess_mask(mask: sitk.Image) -> sitk.Image:
    """
    Apply preprocessing pipeline to segmentation masks.
    
    Uses nearest-neighbor interpolation to preserve discrete label values
    during resampling and resizing operations.
    
    Args:
        mask (sitk.Image): Original segmentation mask
    
    Returns:
        sitk.Image: Preprocessed mask with 1mm³ spacing and 192³ dimensions
    
    Note:
        is_mask=True ensures nearest-neighbor interpolation to prevent
        label smoothing that would create invalid intermediate values
    """
    # Resample with nearest-neighbor to preserve discrete labels
    mask = resample_image(mask, (1.0, 1.0, 1.0), is_mask=True)
    
    # Resize with nearest-neighbor to maintain label integrity
    mask = resize_image(mask, TARGET_SIZE, is_mask=True)
    
    return mask


def process_amos_dataset(root_dir: Path, output_dir: Path) -> None:
    """
    Process AMOS dataset with its specific directory structure and conventions.
    
    AMOS provides separate training and validation splits with paired
    image-label files. Images are numbered, with IDs <500 being CT
    and >=500 being MR scans.
    
    Args:
        root_dir (Path): AMOS dataset root containing imagesTr/labelsTr folders
        output_dir (Path): Output directory for preprocessed data
    """
    # Define dataset splits to process
    splits = [
        ('imagesTr', 'labelsTr'),  # Training set
        ('imagesVa', 'labelsVa')   # Validation set
    ]
    
    for split_img, split_lbl in tqdm(splits, desc='AMOS splits'):
        img_dir = root_dir / split_img
        lbl_dir = root_dir / split_lbl
        
        # Skip if split doesn't exist (some datasets may not have validation)
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"Skipping missing split: {split_img}")
            continue
        
        # Process each image-label pair
        for img_path in tqdm(sorted(img_dir.glob('*.nii*')), 
                            desc=f'  {split_img}', unit='file'):
            fname = img_path.name
            
            # Extract base name without extension
            base = fname.replace('.nii.gz', '').replace('.nii', '')
            
            # Extract patient ID to determine modality
            # AMOS convention: amos_XXXX where XXXX is patient ID
            try:
                idx = int(base.split('_')[-1])
            except ValueError:
                print(f"Skipping unexpected file: {fname}")
                continue
            
            # Determine modality based on ID range
            # AMOS convention: ID < 500 = CT, ID >= 500 = MR
            modality = 'CT' if idx < 500 else 'MR'
            
            # ============ Load Data ============
            img = safe_read_image(str(img_path))
            original_mask = safe_read_image(str(lbl_dir / fname))
            
            # ============ Process Masks ============
            # Filter to keep only target organs
            eval_mask = filter_mask(original_mask)
            
            # ============ Apply Preprocessing ============
            processed_img = preprocess_image(img, modality)
            processed_mask = preprocess_mask(eval_mask)
            
            # ============ Save Outputs ============
            # Create consistent naming scheme
            prefix = f'amos_{modality.lower()}_'
            out_name = f"{prefix}{base}.nii.gz"
            
            # Organize by modality
            mod_dir = output_dir / 'AMOS' / modality
            
            # Save preprocessed data for training (192³)
            save_nifti(processed_img, str(mod_dir / 'images' / out_name))
            save_nifti(processed_mask, str(mod_dir / 'labels' / out_name))
            
            # Save original-resolution mask for evaluation metrics
            # (Evaluation at original resolution is more clinically meaningful)
            save_nifti(eval_mask, str(mod_dir / 'labels_orig' / out_name))


def process_chaos_dataset(root_dir: Path, output_dir: Path) -> None:
    """
    Process CHAOS dataset with its unique DICOM/PNG structure.
    
    CHAOS provides images as DICOM series and masks as PNG slices,
    requiring special handling for format conversion and label mapping.
    
    Args:
        root_dir (Path): CHAOS dataset root directory
        output_dir (Path): Output directory for preprocessed data
    
    Note:
        CT cases only have liver annotations, while MR cases have
        multi-organ annotations requiring label remapping.
    """
    base = root_dir / 'CHAOS_Train_Sets' / 'Train_Sets'
    
    for modality in ['CT', 'MR']:
        # Process each patient case
        for case in tqdm(sorted((base / modality).iterdir()), 
                        desc=f'CHAOS {modality}', unit='case'):
            
            # Skip non-directory entries
            if not case.is_dir():
                continue
            
            # ============ Load Image Volume ============
            if modality == 'CT':
                # CT images directly under case directory
                dcm_dir = case / 'DICOM_anon'
            else:
                # MR images under T2SPIR subfolder (T2-weighted sequence)
                dcm_dir = case / 'T2SPIR' / 'DICOM_anon'
            
            img = load_dicom_series(dcm_dir)
            
            # ============ Load Mask Volume ============
            if modality == 'CT':
                mask_dir = case / 'Ground'
            else:
                # Try T2SPIR-specific masks first, fall back to general
                mask_dir = case / 'T2SPIR' / 'Ground'
                if not mask_dir.exists():
                    mask_dir = case / 'Ground'
            
            if not mask_dir.exists():
                raise FileNotFoundError(f"Cannot find mask directory for case: {case}")
            
            # Load PNG slices and align with image volume
            raw_mask = load_mask_png_series(mask_dir, img)
            arr_raw = sitk.GetArrayFromImage(raw_mask)
            
            # ============ Remap Labels to AMOS Convention ============
            if modality == "CT":
                # CHAOS CT only annotates liver
                # Convert any non-zero value to liver label (6)
                remap = (arr_raw > 0).astype(np.uint8) * 6
            else:
                # CHAOS MR uses different label values
                # Apply mapping to match AMOS label convention
                remap = np.zeros_like(arr_raw, dtype=np.uint8)
                for orig_val, tgt_label in CHAOS_LABEL_MAP.items():
                    remap[arr_raw == orig_val] = tgt_label
            
            # Create new mask with remapped labels
            mask = sitk.GetImageFromArray(remap)
            mask.CopyInformation(img)
            
            # ============ Filter and Preprocess ============
            # For MR, filter to target organs; for CT, keep as-is (only liver)
            eval_mask = filter_mask(mask) if modality == "MR" else mask
            
            # Apply standard preprocessing pipeline
            processed_img = preprocess_image(img, modality)
            processed_mask = preprocess_mask(eval_mask)
            
            # ============ Save Outputs ============
            # Create consistent naming scheme
            prefix = f"chaos_{modality.lower()}_"
            name = prefix + case.name + '.nii.gz'
            
            # Organize by modality
            mod_dir = output_dir / 'CHAOS' / modality
            
            # Save all outputs
            save_nifti(processed_img, str(mod_dir / 'images' / name))
            save_nifti(processed_mask, str(mod_dir / 'labels' / name))
            save_nifti(eval_mask, str(mod_dir / 'labels_orig' / name))


def setup_output_dirs(output_dir: Path, datasets: List[str]) -> None:
    """
    Create complete output directory structure for preprocessed data.
    
    Creates a hierarchical structure organized by dataset, modality,
    and data type for easy access during training and evaluation.
    
    Args:
        output_dir (Path): Root output directory
        datasets (List[str]): List of dataset names to process ('AMOS', 'CHAOS')
    """
    for dataset in datasets:
        for modality in ['CT', 'MR']:
            for subdir in ['images', 'labels', 'labels_orig']:
                dir_path = output_dir / dataset / modality / subdir
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"Created: {dir_path}")


def main():
    """
    Main entry point for the preprocessing pipeline.
    
    Parses command-line arguments and orchestrates preprocessing
    for specified datasets. Supports processing AMOS and CHAOS
    datasets independently or together.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess AMOS and CHAOS datasets for multi-organ segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process only AMOS dataset
  python preprocess.py --amos_dir /data/AMOS --output_dir /data/preprocessed
  
  # Process only CHAOS dataset  
  python preprocess.py --chaos_dir /data/CHAOS --output_dir /data/preprocessed
  
  # Process both datasets
  python preprocess.py --amos_dir /data/AMOS --chaos_dir /data/CHAOS --output_dir /data/preprocessed
        """
    )
    
    parser.add_argument(
        '--amos_dir', 
        type=Path,
        help='Path to AMOS dataset root directory (contains imagesTr, labelsTr, etc.)'
    )
    parser.add_argument(
        '--chaos_dir', 
        type=Path,
        help='Path to CHAOS dataset root directory (contains CHAOS_Train_Sets folder)'
    )
    parser.add_argument(
        '--output_dir', 
        type=Path, 
        required=True,
        help='Output directory for preprocessed data'
    )
    
    args = parser.parse_args()
    
    # Determine which datasets to process
    datasets_to_process = []
    if args.amos_dir:
        if not args.amos_dir.exists():
            print(f"Error: AMOS directory not found: {args.amos_dir}")
            return
        datasets_to_process.append('AMOS')
    
    if args.chaos_dir:
        if not args.chaos_dir.exists():
            print(f"Error: CHAOS directory not found: {args.chaos_dir}")
            return
        datasets_to_process.append('CHAOS')
    
    if not datasets_to_process:
        print("Error: No datasets specified. Use --amos_dir and/or --chaos_dir")
        parser.print_help()
        return
    
    print(f"Datasets to process: {datasets_to_process}")
    print(f"Output directory: {args.output_dir}")
    
    # Setup output directory structure
    print("\nCreating output directories...")
    setup_output_dirs(args.output_dir, datasets_to_process)
    
    # Process each dataset
    if 'AMOS' in datasets_to_process:
        print("\n" + "="*60)
        print("Processing AMOS dataset...")
        print("="*60)
        process_amos_dataset(args.amos_dir, args.output_dir)
    
    if 'CHAOS' in datasets_to_process:
        print("\n" + "="*60)
        print("Processing CHAOS dataset...")
        print("="*60)
        process_chaos_dataset(args.chaos_dir, args.output_dir)
    
    print("\n" + "="*60)
    print(f"Preprocessing complete!")
    print(f"Output saved to: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()