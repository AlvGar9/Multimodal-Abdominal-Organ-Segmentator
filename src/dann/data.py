# src/dann/data.py
import os
import glob
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from monai.data import Dataset, DataLoader

# We can reuse the transforms from the original training module
from training.data import get_transforms

def get_domain_files(base_dir: str, modality: str, domain_id: int) -> List[Dict]:
    """Helper function to load file paths and assign a domain ID."""
    image_dir = os.path.join(base_dir, modality, 'images')
    label_dir = os.path.join(base_dir, modality, 'labels')
    
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz*')))
    label_paths = sorted(glob.glob(os.path.join(label_dir, '*.nii.gz*')))
    
    if not image_paths:
        raise FileNotFoundError(f"No image files found for {modality} in {image_dir}")

    return [{'image': img, 'label': lbl, 'domain': domain_id} 
            for img, lbl in zip(image_paths, label_paths)]

def get_dann_dataloaders(
    config: object,
    num_mri_samples: str
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Creates dataloaders for DANN training.

    The training set consists of all CT data and a subset of MRI data.
    The validation set consists of a mix of CT and MRI data.
    Two separate test sets are created for final evaluation.
    """
    ct_files = get_domain_files(config.PREPROCESSED_DATA_DIR, 'CT', 0)
    mri_files = get_domain_files(config.PREPROCESSED_DATA_DIR, 'MR', 1)

    # Split CT data
    ct_train_files, ct_test_files = train_test_split(ct_files, test_size=0.1, random_state=config.SEED)
    ct_train_files, ct_val_files = train_test_split(ct_train_files, test_size=0.22, random_state=config.SEED)

    # Split MRI data
    mri_train_files, mri_test_files = train_test_split(mri_files, test_size=0.2, random_state=config.SEED)
    mri_train_files, mri_val_files = train_test_split(mri_train_files, test_size=0.25, random_state=config.SEED)
    
    # Create the training dataset for the current experiment
    if num_mri_samples == 'all':
        mri_subset = mri_train_files
    else:
        num_samples = int(num_mri_samples)
        if num_samples > len(mri_train_files):
            raise ValueError(f"Requested {num_samples} MRI samples, but only {len(mri_train_files)} available.")
        mri_subset = mri_train_files[:num_samples] # Simple slice for reproducibility
        
    combined_train_files = ct_train_files + mri_subset
    combined_val_files = ct_val_files + mri_val_files
    
    print(f"Training set: {len(combined_train_files)} ({len(ct_train_files)} CT, {len(mri_subset)} MRI)")
    print(f"Validation set: {len(combined_val_files)} ({len(ct_val_files)} CT, {len(mri_val_files)} MRI)")

    # Get transforms and create datasets/dataloaders
    train_ts, val_ts = get_transforms()
    
    train_ds = Dataset(data=combined_train_files, transform=train_ts)
    val_ds = Dataset(data=combined_val_files, transform=val_ts)
    ct_test_ds = Dataset(data=ct_test_files, transform=val_ts)
    mri_test_ds = Dataset(data=mri_test_files, transform=val_ts)
    
    train_loader = DataLoader(train_ds, batch_size=config.SCRATCH_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.SCRATCH_BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    ct_test_loader = DataLoader(ct_test_ds, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)
    mri_test_loader = DataLoader(mri_test_ds, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)
    
    return train_loader, val_loader, ct_test_loader, mri_test_loader