"""Image resizing functions for fixed output dimensions."""

import argparse
import SimpleITK as sitk


def resize_image(image: sitk.Image,
                 output_size: tuple = (256, 256, 256),
                 is_mask: bool = False) -> sitk.Image:
    """
    Resize image to fixed target dimensions.
    
    Args:
        image: Input SimpleITK image
        output_size: Target size (x, y, z) in voxels
        is_mask: Whether image is a segmentation mask
    
    Returns:
        Resized SimpleITK image
    """
    orig_size = image.GetSize()
    orig_spacing = image.GetSpacing()
    
    # Calculate new spacing to achieve target size
    new_spacing = [
        ospc * osz / nsz 
        for ospc, osz, nsz in zip(orig_spacing, orig_size, output_size)
    ]
    
    # Setup resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(output_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    
    # Set interpolation method
    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
    
    return resampler.Execute(image)


def main():
    """Command-line interface for resizing."""
    parser = argparse.ArgumentParser(
        description="Resize medical volume to fixed dimensions")
    parser.add_argument('--input', required=True, 
                       help='Input NIfTI file')
    parser.add_argument('--output', required=True, 
                       help='Output NIfTI file')
    parser.add_argument('--size', nargs=3, type=int, 
                       default=[256, 256, 256],
                       help='Target volume dimensions (default: 256 256 256)')
    parser.add_argument('--mask', action='store_true', 
                       help='Flag if input is a segmentation mask')
    args = parser.parse_args()
    
    # Load, resize, and save
    img = sitk.ReadImage(args.input)
    resized = resize_image(img, tuple(args.size), is_mask=args.mask)
    sitk.WriteImage(resized, args.output)
    print(f"Resized image saved to: {args.output}")


if __name__ == '__main__':
    main()