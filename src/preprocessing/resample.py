"""Image resampling functions for medical images."""

import os
import argparse
import SimpleITK as sitk


def resample_image(image: sitk.Image,
                   new_spacing: tuple,
                   is_mask: bool = False) -> sitk.Image:
    """
    Resample image to target spacing.
    
    Args:
        image: Input SimpleITK image
        new_spacing: Target voxel spacing (x, y, z) in mm
        is_mask: Whether image is a segmentation mask
    
    Returns:
        Resampled SimpleITK image
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # Calculate new size based on target spacing
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]
    
    # Setup resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    
    # Set interpolation method
    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
    
    return resampler.Execute(image)


def main():
    """Command-line interface for resampling."""
    parser = argparse.ArgumentParser(
        description="Resample medical image to isotropic voxels")
    parser.add_argument('--input', required=True,
                       help='Input file path or DICOM directory')
    parser.add_argument('--output', required=True,
                       help='Output NIfTI file path')
    parser.add_argument('--spacing', nargs=3, type=float, 
                       default=[1.0, 1.0, 1.0],
                       help='New voxel spacing in mm (default: 1 1 1)')
    parser.add_argument('--mask', action='store_true',
                       help='Flag if input is a segmentation mask')
    args = parser.parse_args()
    
    # Load input
    if os.path.isdir(args.input):
        from .utils import load_dicom_series
        img = load_dicom_series(args.input)
    else:
        img = sitk.ReadImage(args.input)
    
    # Resample
    resampled = resample_image(img, tuple(args.spacing), is_mask=args.mask)
    
    # Save output
    from .utils import save_nifti
    save_nifti(resampled, args.output)
    print(f"Resampled image saved to: {args.output}")


if __name__ == '__main__':
    main()