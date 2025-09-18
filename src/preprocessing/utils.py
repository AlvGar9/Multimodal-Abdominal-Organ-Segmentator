"""Utility functions for medical image I/O operations."""

import os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import cv2


def load_dicom_series(dicom_dir: str) -> sitk.Image:
    """Load DICOM series from directory."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    
    if not series_ids:
        raise ValueError(f"No DICOM series found in {dicom_dir}")
    
    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(file_names)
    return reader.Execute()


def load_png_masks(mask_dir: str) -> sitk.Image:
    """Load and stack PNG slice masks into 3D volume."""
    png_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
    
    if not png_files:
        raise ValueError(f"No PNG masks found in {mask_dir}")
    
    slices = []
    for fname in png_files:
        path = os.path.join(mask_dir, fname)
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        slices.append(mask)
    
    # Stack into (Z, Y, X) for SimpleITK
    arr = np.stack(slices, axis=0)
    return sitk.GetImageFromArray(arr)


def save_nifti(image: sitk.Image, output_path: str) -> None:
    """Save SimpleITK image as NIfTI file."""
    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, output_path)


def copy_metadata(source: sitk.Image, target: sitk.Image) -> sitk.Image:
    """Copy spatial metadata from source to target image."""
    target.SetSpacing(source.GetSpacing())
    target.SetOrigin(source.GetOrigin())
    target.SetDirection(source.GetDirection())
    return target