import os
import argparse
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from glob import glob
import shutil
import sys

def validate_nifti_file(filepath):
    """
    Validate that a NIfTI file can be loaded and its data accessed
    
    Args:
        filepath: Path to NIfTI file
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Try loading with nibabel
        nib_img = nib.load(filepath)
        # Access the data to ensure it's valid
        data = nib_img.get_fdata()
        
        # Check for invalid values
        if np.isnan(data).any():
            return False, "Data contains NaN values"
        
        if np.isinf(data).any():
            return False, "Data contains infinite values"
        
        # Try loading with SimpleITK
        sitk_img = sitk.ReadImage(filepath)
        
        return True, "File is valid"
    except Exception as e:
        return False, f"Error loading file: {str(e)}"

def visualize_nifti(filepath, output_dir=None):
    """
    Visualize a NIfTI file by showing center slices in each plane
    
    Args:
        filepath: Path to NIfTI file
        output_dir: Directory to save visualization (if None, display instead)
        
    Returns:
        bool: True if successful
    """
    try:
        # Load the image
        img = nib.load(filepath)
        data = img.get_fdata()
        
        # Get center slices
        x, y, z = np.array(data.shape) // 2
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot axial view (top-down)
        axes[0].imshow(data[:, :, z], cmap='gray')
        axes[0].set_title('Axial')
        axes[0].axis('off')
        
        # Plot coronal view (front-back)
        axes[1].imshow(data[:, y, :], cmap='gray')
        axes[1].set_title('Coronal')
        axes[1].axis('off')
        
        # Plot sagittal view (left-right)
        axes[2].imshow(data[x, :, :], cmap='gray')
        axes[2].set_title('Sagittal')
        axes[2].axis('off')
        
        # Set title
        plt.suptitle(os.path.basename(filepath))
        
        # Save or display
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, os.path.basename(filepath).replace('.nii.gz', '.png').replace('.nii', '.png'))
            plt.savefig(output_file)
            plt.close()
            print(f"Saved visualization to {output_file}")
        else:
            plt.tight_layout()
            plt.show()
        
        return True
    except Exception as e:
        print(f"Error visualizing {filepath}: {e}")
        return False

def save_metadata(filepath, output_dir):
    """
    Save metadata about a NIfTI file
    
    Args:
        filepath: Path to NIfTI file
        output_dir: Directory to save metadata
        
    Returns:
        bool: True if successful
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the image
        img = nib.load(filepath)
        
        # Get metadata
        metadata = {
            'filename': os.path.basename(filepath),
            'shape': img.shape,
            'datatype': img.get_data_dtype(),
            'header': dict(img.header),
            'affine': img.affine.tolist()
        }
        
        # Save metadata
        output_file = os.path.join(output_dir, os.path.basename(filepath) + '.metadata.txt')
        with open(output_file, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        # Also try with SimpleITK
        try:
            sitk_img = sitk.ReadImage(filepath)
            sitk_metadata = {
                'Size': sitk_img.GetSize(),
                'Spacing': sitk_img.GetSpacing(),
                'Origin': sitk_img.GetOrigin(),
                'Direction': sitk_img.GetDirection()
            }
            
            with open(output_file, 'a') as f:
                f.write("\nSimpleITK Metadata:\n")
                for key, value in sitk_metadata.items():
                    f.write(f"{key}: {value}\n")
        except Exception as e:
            with open(output_file, 'a') as f:
                f.write(f"\nError getting SimpleITK metadata: {e}\n")
        
        return True
    except Exception as e:
        print(f"Error saving metadata for {filepath}: {e}")
        return False

def get_file_extension(filepath):
    """
    Get the file extension of a NIfTI file (.nii or .nii.gz)
    
    Args:
        filepath: Path to NIfTI file
        
    Returns:
        str: File extension
    """
    filename = os.path.basename(filepath)
    if filename.endswith('.nii.gz'):
        return '.nii.gz'
    elif filename.endswith('.nii'):
        return '.nii'
    else:
        return ''

def test_preprocessing_steps(input_file, output_dir):
    """
    Test each preprocessing step individually
    
    Args:
        input_file: Path to input NIfTI file
        output_dir: Directory to save output files
        
    Returns:
        bool: True if all steps were successful
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the file extension of the input file
        ext = get_file_extension(input_file)
        
        # Step 1: Copy the input file
        step1_file = os.path.join(output_dir, f'1_copied{ext}')
        shutil.copy(input_file, step1_file)
        
        # Validate the copied file
        valid, message = validate_nifti_file(step1_file)
        if not valid:
            print(f"Step 1 (copy) failed: {message}")
            return False
        else:
            print("Step 1 (copy) succeeded")
        
        # Step 2: Test loading and saving with nibabel
        step2_file = os.path.join(output_dir, f'2_nibabel{ext}')
        img = nib.load(step1_file)
        data = img.get_fdata()
        new_img = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(new_img, step2_file)
        
        # Validate the nibabel file
        valid, message = validate_nifti_file(step2_file)
        if not valid:
            print(f"Step 2 (nibabel) failed: {message}")
            return False
        else:
            print("Step 2 (nibabel) succeeded")
        
        # Step 3: Test loading and saving with SimpleITK
        step3_file = os.path.join(output_dir, f'3_sitk{ext}')
        sitk_img = sitk.ReadImage(step2_file)
        sitk.WriteImage(sitk_img, step3_file)
        
        # Validate the SimpleITK file
        valid, message = validate_nifti_file(step3_file)
        if not valid:
            print(f"Step 3 (SimpleITK) failed: {message}")
            return False
        else:
            print("Step 3 (SimpleITK) succeeded")
        
        # Step 4: Test resampling with SimpleITK
        step4_file = os.path.join(output_dir, f'4_resampled{ext}')
        original_spacing = sitk_img.GetSpacing()
        original_size = sitk_img.GetSize()
        new_spacing = [1.0, 1.0, 1.0]  # 1mm isotropic
        
        # Calculate new size
        new_size = [
            int(round(original_size[0] * original_spacing[0] / new_spacing[0])),
            int(round(original_size[1] * original_spacing[1] / new_spacing[1])),
            int(round(original_size[2] * original_spacing[2] / new_spacing[2]))
        ]
        
        # Create resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(sitk_img.GetDirection())
        resampler.SetOutputOrigin(sitk_img.GetOrigin())
        resampler.SetDefaultPixelValue(0)
        
        # Resample the image
        resampled_img = resampler.Execute(sitk_img)
        sitk.WriteImage(resampled_img, step4_file)
        
        # Validate the resampled file
        valid, message = validate_nifti_file(step4_file)
        if not valid:
            print(f"Step 4 (resampling) failed: {message}")
            return False
        else:
            print("Step 4 (resampling) succeeded")
        
        # Step 5: Test intensity normalization
        step5_file = os.path.join(output_dir, f'5_normalized{ext}')
        img = nib.load(step4_file)
        data = img.get_fdata()
        
        # Create a mask (non-zero voxels)
        mask = data > 0
        
        # Calculate mean and std within the mask
        mean_val = np.mean(data[mask])
        std_val = np.std(data[mask])
        
        if std_val == 0:
            std_val = 1.0
        
        # Z-score normalization
        normalized_data = np.zeros_like(data)
        normalized_data[mask] = (data[mask] - mean_val) / std_val
        
        # Save the normalized image
        normalized_img = nib.Nifti1Image(normalized_data, img.affine, img.header)
        nib.save(normalized_img, step5_file)
        
        # Validate the normalized file
        valid, message = validate_nifti_file(step5_file)
        if not valid:
            print(f"Step 5 (normalization) failed: {message}")
            return False
        else:
            print("Step 5 (normalization) succeeded")
        
        # Visualize each step
        for step_file in [step1_file, step2_file, step3_file, step4_file, step5_file]:
            visualize_nifti(step_file, os.path.join(output_dir, 'visualizations'))
            save_metadata(step_file, os.path.join(output_dir, 'metadata'))
        
        return True
    except Exception as e:
        print(f"Error in preprocessing steps: {e}")
        import traceback
        traceback.print_exc()
        return False

def find_sample_file(data_dir, max_files=5):
    """
    Find a sample NIfTI file in the data directory
    
    Args:
        data_dir: Directory to search for NIfTI files
        max_files: Maximum number of files to validate
        
    Returns:
        list: List of validated file paths
    """
    # Find all NIfTI files
    try:
        # Windows compatibility: use Path.glob instead of glob with recursive=True
        data_dir_path = Path(os.path.abspath(data_dir))
        nii_files = list(data_dir_path.glob('**/*.nii'))
        nii_files += list(data_dir_path.glob('**/*.nii.gz'))
        nii_files = [str(f) for f in nii_files]
    except Exception as e:
        print(f"Error finding NIfTI files: {e}")
        # Fallback to regular glob
        nii_files = glob(os.path.join(data_dir, '**', '*.nii'), recursive=True)
        nii_files += glob(os.path.join(data_dir, '**', '*.nii.gz'), recursive=True)
    
    # Limit the number of files
    nii_files = nii_files[:max_files]
    
    # Validate each file
    valid_files = []
    for filepath in nii_files:
        print(f"Validating {filepath}...")
        valid, message = validate_nifti_file(filepath)
        print(f"  {message}")
        if valid:
            valid_files.append(filepath)
    
    return valid_files

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Validate preprocessing pipeline')
    
    parser.add_argument('--data_dir', type=str, default='./MPR/ADNI',
                       help='Directory containing input NIfTI files')
    parser.add_argument('--output_dir', type=str, default='./preprocessing_test',
                       help='Directory to save output files')
    parser.add_argument('--max_files', type=int, default=5,
                       help='Maximum number of files to validate')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    print(f"Searching for NIfTI files in {args.data_dir}...")
    valid_files = find_sample_file(args.data_dir, args.max_files)
    
    if not valid_files:
        print("No valid NIfTI files found.")
        sys.exit(1)
    
    print(f"Found {len(valid_files)} valid NIfTI files.")
    
    # Test preprocessing on each valid file
    for i, filepath in enumerate(valid_files):
        print(f"\nTesting preprocessing on file {i+1}/{len(valid_files)}: {filepath}")
        output_dir = os.path.join(args.output_dir, f"test_{i+1}")
        success = test_preprocessing_steps(filepath, output_dir)
        if success:
            print(f"Preprocessing test {i+1} succeeded.")
        else:
            print(f"Preprocessing test {i+1} failed.")

if __name__ == "__main__":
    main() 