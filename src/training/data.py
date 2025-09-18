# src/training/data.py
import os
import glob
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Lambdad,
    NormalizeIntensityd, ScaleIntensityRanged, ToTensord, RandRotate90d,
    RandFlipd, RandAdjustContrastd, RandGaussianNoised,
    RandScaleIntensityd, RandBiasFieldd
)

def get_data_files(image_dir: str, label_dir: str) -> List[Dict[str, str]]:
    """
    Scans data directories, creates image-label pairs, and returns them as a list.
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz*')))
    label_paths = sorted(glob.glob(os.path.join(label_dir, '*.nii.gz*')))
    
    if not image_paths or not label_paths:
        raise FileNotFoundError(f"No NIfTI files found in {image_dir} or {label_dir}")
    
    assert len(image_paths) == len(label_paths), "Mismatch in number of images and labels"
    
    files = [{'image': img, 'label': lbl} for img, lbl in zip(image_paths, label_paths)]
    return files

def split_data(
    files: List[Dict[str, str]],
    seed: int,
    num_train_samples: int | str | None = None
) -> Tuple[List, List, List]:
    """
    Splits file list into training, validation, and test sets.
    Optionally subsamples the training set based on `num_train_samples`.

    - If `num_train_samples` is an integer (e.g., 5), it subsamples the training set.
    - If `num_train_samples` is 'all', it uses the train+val sets for training.
    - If `num_train_samples` is None, it performs a standard split.
    """
    indices = list(range(len(files)))
    
    # First, split off the test set (10%) to keep it truly held-out
    train_val_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=seed)
    test_files = [files[i] for i in test_indices]

    # Now, split the remaining data into a default training and validation set
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.22, random_state=seed)
    
    if num_train_samples:
        if str(num_train_samples).lower() == 'all':
            # Use combined train+val sets for training and the test set for validation
            train_files = [files[i] for i in train_val_indices]
            val_files = test_files
            print(f"Using all {len(train_files)} train+val samples for training, and test set for validation.")
        else:
            # Subsample the training set to the specified number
            num_samples = int(num_train_samples)
            if num_samples > len(train_indices):
                raise ValueError(f"Requested {num_samples} samples, but only {len(train_indices)} available.")
            
            # Use a separate RandomState for subsampling to ensure consistency
            subsampled_train_indices = np.random.RandomState(seed).choice(
                train_indices, num_samples, replace=False
            )
            train_files = [files[i] for i in subsampled_train_indices]
            val_files = [files[i] for i in val_indices] # Keep the original validation set
            print(f"Subsampling training data to {len(train_files)} samples.")
    else:
        # Default behavior: use the standard 70/20/10 split
        train_files = [files[i] for i in train_indices]
        val_files = [files[i] for i in val_indices]

    print(f"Final data split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")
    return train_files, val_files, test_files
    
def get_transforms() -> Tuple[Compose, Compose]:
    """
    Defines and returns the MONAI transforms for training and validation.
    """
    # Remap class 6 (gallbladder), if present, to class 4 (liver)
    remap_label = Lambdad(keys=['label'], func=lambda x: np.where(x == 6, 4, x))

    train_transforms = Compose([
        LoadImaged(['image', 'label']),
        EnsureChannelFirstd(['image', 'label']),
        Orientationd(['image', 'label'], axcodes='RAS'),
        remap_label,
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        ScaleIntensityRanged(keys='image', a_min=-3, a_max=3, b_min=0, b_max=1, clip=True),
        RandRotate90d(keys=['image','label'], prob=0.5, spatial_axes=(1, 2), max_k=3),
        RandFlipd(keys=['image','label'], prob=0.5, spatial_axis=2),
        RandAdjustContrastd(keys=['image'], prob=0.3, gamma=(0.7, 1.5)),
        RandGaussianNoised(keys=['image'], prob=0.3, mean=0.0, std=0.01),
        RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.3),
        RandBiasFieldd(keys=['image'], prob=0.3),
        ToTensord(['image','label'])
    ])

    val_transforms = Compose([
        LoadImaged(['image','label']),
        EnsureChannelFirstd(['image','label']),
        Orientationd(['image', 'label'], axcodes='RAS'),
        remap_label,
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        ScaleIntensityRanged(keys='image', a_min=-3, a_max=3, b_min=0, b_max=1, clip=True),
        ToTensord(['image','label'])
    ])
    
    return train_transforms, val_transforms

def get_dataloaders(
    config: object
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns the training, validation, and test dataloaders
    based on the provided configuration object.
    """
    files = get_data_files(config.IMAGE_DIR, config.LABEL_DIR)
    
    # Safely get NUM_TRAIN_SAMPLES from config, defaulting to None if not present
    num_samples = getattr(config, 'NUM_TRAIN_SAMPLES', None)
    train_files, val_files, test_files = split_data(files, config.SEED, num_samples)
    
    train_ts, val_ts = get_transforms()
    
    train_ds = Dataset(data=train_files, transform=train_ts)
    val_ds = Dataset(data=val_files, transform=val_ts)
    test_ds = Dataset(data=test_files, transform=val_ts)
    
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=1
    )
    
    return train_loader, val_loader, test_loader