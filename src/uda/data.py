# src/uda/data.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split
from skimage.exposure import match_histograms
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Lambdad, \
    Orientationd, ToTensord, RandRotate90d, RandFlipd, RandGaussianNoised, \
    RandScaleIntensityd, RandBiasFieldd, ScaleIntensityRangePercentilesd
from monai.data import Dataset, DataLoader

class UDACombinedDataset(TorchDataset):
    """
    A dataset that pairs a source item (CT) with a randomly selected target
    item (MRI) for each training step.
    """
    def __init__(self, source_files, target_files, source_transform, target_transform):
        self.source_files = source_files
        self.target_files = target_files
        self.source_transform = source_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.source_files) # Length is determined by the labeled source domain

    def __getitem__(self, idx):
        source_item = self.source_transform(self.source_files[idx])
        
        # Select a random target item for pairing
        target_idx = np.random.randint(0, len(self.target_files))
        target_item = self.target_transform({'image': self.target_files[target_idx]['image']})
        
        return {
            'ct_image': source_item['image'],
            'ct_label': source_item['label'],
            'mri_image': target_item['image']
        }

def get_uda_transforms(ct_train_files: list) -> tuple:
    """Creates and returns the MONAI transforms for UDA, including histogram matching."""
    # Create a reference image for histogram matching from a subset of CT scans
    ref_indices = np.random.choice(len(ct_train_files), 10, replace=False)
    ref_paths = [ct_train_files[i]['image'] for i in ref_indices]
    loader = Compose([LoadImaged(keys=['image']), EnsureChannelFirstd(keys=['image'])])
    ref_images = [loader({'image': p})['image'] for p in ref_paths]
    ref_np = torch.mean(torch.cat(ref_images, dim=0), dim=0).squeeze(0).numpy()

    hist_match = Lambdad(keys="image", func=lambda img: torch.as_tensor(
        np.expand_dims(match_histograms(img.squeeze(0).cpu().numpy(), ref_np, channel_axis=None), axis=0)
    ))

    # Define base and augmentation transforms
    loading = Compose([
        LoadImaged(keys=['image', 'label'], allow_missing_keys=True),
        EnsureChannelFirstd(keys=['image', 'label'], allow_missing_keys=True),
        Lambdad(keys='label', func=lambda x: np.where(x == 6, 4, x), allow_missing_keys=True),
        Orientationd(keys=['image', 'label'], axcodes='RAS', allow_missing_keys=True),
    ])
    augment = Compose([
        RandRotate90d(keys=['image','label'], prob=0.5, allow_missing_keys=True),
        RandFlipd(keys=['image','label'], prob=0.5, allow_missing_keys=True),
        RandGaussianNoised(keys=['image'], prob=0.3, std=0.01),
        RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.3),
        RandBiasFieldd(keys=['image'], prob=0.3),
    ])
    intensity = ScaleIntensityRangePercentilesd(keys='image', lower=1, upper=99, b_min=0.0, b_max=1.0)
    to_tensor = ToTensord(keys=['image', 'label'], allow_missing_keys=True)

    ct_train_ts = Compose([loading, intensity, augment, to_tensor])
    mri_train_ts = Compose([loading, hist_match, intensity, augment, to_tensor])
    ct_val_ts = Compose([loading, intensity, to_tensor])
    mri_val_ts = Compose([loading, hist_match, intensity, to_tensor])
    
    return ct_train_ts, mri_train_ts, ct_val_ts, mri_val_ts

def get_uda_dataloaders(config, num_mri_samples):
    """Prepares datasets and dataloaders for the UDA experiment."""
    # Get file lists
    ct_files = [{'image': p, 'label': l} for p, l in zip(sorted(glob.glob(os.path.join(config.PREPROCESSED_DATA_DIR, 'CT/images/*.nii.gz*'))), sorted(glob.glob(os.path.join(config.PREPROCESSED_DATA_DIR, 'CT/labels/*.nii.gz*'))))]
    mri_files = [{'image': p, 'label': l} for p, l in zip(sorted(glob.glob(os.path.join(config.PREPROCESSED_DATA_DIR, 'MR/images/*.nii.gz*'))), sorted(glob.glob(os.path.join(config.PREPROCESSED_DATA_DIR, 'MR/labels/*.nii.gz*'))))]

    # Split data
    ct_train_files, ct_test_files = train_test_split(ct_files, test_size=0.2, random_state=config.SEED)
    
    mri_subset, _ = train_test_split(mri_files, train_size=int(num_mri_samples), random_state=config.SEED)
    mri_train_files, mri_test_files = train_test_split(mri_subset, test_size=0.3, random_state=config.SEED)

    # Get transforms
    ct_train_ts, mri_train_ts, ct_val_ts, mri_val_ts = get_uda_transforms(ct_train_files)
    
    # Create datasets
    train_ds = UDACombinedDataset(ct_train_files, mri_train_files, ct_train_ts, mri_train_ts)
    ct_test_ds = Dataset(data=ct_test_files, transform=ct_val_ts)
    mri_test_ds = Dataset(data=mri_test_files, transform=mri_val_ts)
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=config.SCRATCH_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    ct_test_loader = DataLoader(ct_test_ds, batch_size=1, shuffle=False)
    mri_test_loader = DataLoader(mri_test_ds, batch_size=1, shuffle=False)
    
    return train_loader, ct_test_loader, mri_test_loader