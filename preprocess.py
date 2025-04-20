import os
import argparse
import subprocess
import shutil
import glob
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import SimpleITK as sitk


def run_command(cmd, verbose=True):
    """Execute shell command"""
    if verbose:
        print(f"Running: {cmd}")
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error executing command: {cmd}")
        print(f"STDERR: {stderr.decode('utf-8')}")
        return False
    
    if verbose:
        print(f"STDOUT: {stdout.decode('utf-8')}")
    
    return True


def skull_strip_using_hd_bet(input_file, output_file, device=0):
    """
    Perform skull stripping using HD-BET (assumes HD-BET is installed)
    
    Args:
        input_file: Path to input NIfTI file
        output_file: Path to output skull-stripped NIfTI file
        device: GPU device ID to use (default: 0, use -1 for CPU)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Convert to forward slashes for external command
    input_file_fwd = str(input_file).replace('\\', '/')
    output_file_fwd = str(output_file).replace('\\', '/')
    cmd = f"hd-bet -i {input_file_fwd} -o {output_file_fwd} -device {device} -mode fast -tta 0"
    return run_command(cmd)


def register_to_mni(input_file, output_file, reference_file, use_flirt=True):
    """
    Register image to MNI152 template using FSL or ANTs
    
    Args:
        input_file: Path to input NIfTI file
        output_file: Path to output registered NIfTI file
        reference_file: Path to reference template (MNI152)
        use_flirt: Whether to use FSL FLIRT (True) or ANTs (False)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Convert to forward slashes for external command
    input_file_fwd = str(input_file).replace('\\', '/')
    output_file_fwd = str(output_file).replace('\\', '/')
    reference_file_fwd = str(reference_file).replace('\\', '/')
    
    if use_flirt:
        # Use FSL FLIRT (assumes FSL is installed)
        cmd = f"flirt -in {input_file_fwd} -ref {reference_file_fwd} -out {output_file_fwd} -dof 12 -interp trilinear"
        return run_command(cmd)
    else:
        # Use ANTs (assumes ANTs is installed)
        output_dir = os.path.dirname(output_file)
        output_prefix = os.path.join(output_dir, "ants_")
        output_prefix_fwd = str(output_prefix).replace('\\', '/')
        
        cmd = (f"antsRegistrationSyN.sh -d 3 -f {reference_file_fwd} -m {input_file_fwd} "
               f"-o {output_prefix_fwd} -t a")
        
        success = run_command(cmd)
        
        if success:
            # Rename the warped image to match the expected output filename
            warped_image = f"{output_prefix}Warped.nii.gz"
            shutil.move(warped_image, output_file)
            
            # Optionally clean up other ANTs output files
            for f in glob.glob(f"{output_prefix}*"):
                if f != output_file:
                    os.remove(f)
        
        return success


def normalize_intensity(input_file, output_file, mask_file=None):
    """
    Normalize intensity values to zero mean and unit variance
    
    Args:
        input_file: Path to input NIfTI file
        output_file: Path to output normalized NIfTI file
        mask_file: Optional path to brain mask file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert paths to absolute paths with proper format
        input_file = os.path.abspath(input_file)
        output_file = os.path.abspath(output_file)
        
        # Load the image
        img = nib.load(input_file)
        data = img.get_fdata()
        
        # Load the mask if provided
        if mask_file and os.path.exists(mask_file):
            mask_file = os.path.abspath(mask_file)
            mask = nib.load(mask_file).get_fdata() > 0
        else:
            # If no mask, create a simple mask (non-zero voxels)
            mask = data > 0
        
        # Calculate mean and std within the mask
        mean_val = np.mean(data[mask])
        std_val = np.std(data[mask])
        
        if std_val == 0:
            print(f"Warning: Standard deviation is zero for {input_file}")
            std_val = 1.0
        
        # Z-score normalization
        normalized_data = np.zeros_like(data)
        normalized_data[mask] = (data[mask] - mean_val) / std_val
        
        # Save the normalized image
        normalized_img = nib.Nifti1Image(normalized_data, img.affine, img.header)
        nib.save(normalized_img, output_file)
        return True
    
    except Exception as e:
        print(f"Error normalizing {input_file}: {e}")
        return False


def resample_to_isotropic(input_file, output_file, new_spacing=1.0):
    """
    Resample the image to isotropic resolution
    
    Args:
        input_file: Path to input NIfTI file
        output_file: Path to output resampled NIfTI file
        new_spacing: Target isotropic spacing in mm
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert paths to absolute paths with proper format
        input_file = os.path.abspath(input_file)
        output_file = os.path.abspath(output_file)
        
        # Load the image using SimpleITK
        img = sitk.ReadImage(input_file)
        
        # Get original spacing
        original_spacing = img.GetSpacing()
        original_size = img.GetSize()
        
        # Calculate new size
        new_size = [
            int(round(original_size[0] * original_spacing[0] / new_spacing)),
            int(round(original_size[1] * original_spacing[1] / new_spacing)),
            int(round(original_size[2] * original_spacing[2] / new_spacing))
        ]
        
        # Create resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputSpacing([new_spacing, new_spacing, new_spacing])
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetDefaultPixelValue(0)
        
        # Resample the image
        resampled_img = resampler.Execute(img)
        
        # Save the resampled image
        sitk.WriteImage(resampled_img, output_file)
        return True
    
    except Exception as e:
        print(f"Error resampling {input_file}: {e}")
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


def process_single_subject(args):
    """
    Process a single subject's MRI scan
    
    Args:
        args: Dictionary of arguments including:
            - input_file: Path to input NIfTI file
            - output_dir: Directory to save processed files
            - reference_file: Path to reference template
            - use_hd_bet: Whether to use HD-BET for skull stripping
            - use_flirt: Whether to use FSL FLIRT for registration
            - gpu_id: GPU device ID for HD-BET
    
    Returns:
        bool: True if successful, False otherwise
    """
    input_file = args['input_file']
    output_dir = args['output_dir']
    reference_file = args['reference_file']
    use_hd_bet = args['use_hd_bet']
    use_flirt = args['use_flirt']
    gpu_id = args['gpu_id']
    
    try:
        # Get the file extension
        ext = get_file_extension(input_file)
        
        # Create subject-specific output directory
        # Extract subject ID from filename (adapted for ADNI format)
        subject_id_parts = Path(input_file).stem.split('_')
        if len(subject_id_parts) >= 3:
            subject_id = '_'.join(subject_id_parts[1:3])
        else:
            # Fallback to using original stem if the expected pattern is not found
            subject_id = Path(input_file).stem
        
        subject_output_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        # Define output filenames with the same extension as the input file
        skull_stripped_file = os.path.join(subject_output_dir, f"{subject_id}_brain{ext}")
        registered_file = os.path.join(subject_output_dir, f"{subject_id}_registered{ext}")
        normalized_file = os.path.join(subject_output_dir, f"{subject_id}_normalized{ext}")
        resampled_file = os.path.join(subject_output_dir, f"{subject_id}_resampled{ext}")
        final_file = os.path.join(subject_output_dir, f"{subject_id}_preprocessed{ext}")
        
        # Step 1: Skull stripping
        if use_hd_bet:
            success = skull_strip_using_hd_bet(input_file, skull_stripped_file, device=gpu_id)
        else:
            # Placeholder for alternative skull stripping method
            # For now, just copy the input file
            shutil.copy(input_file, skull_stripped_file)
            success = True
        
        if not success:
            return False
        
        # Step 2: Registration to MNI space
        if reference_file:
            success = register_to_mni(skull_stripped_file, registered_file, reference_file, use_flirt=use_flirt)
            if not success:
                return False
        else:
            # Skip registration if no reference file
            shutil.copy(skull_stripped_file, registered_file)
        
        # Step 3: Resample to isotropic resolution
        success = resample_to_isotropic(registered_file, resampled_file)
        if not success:
            return False
        
        # Step 4: Intensity normalization
        success = normalize_intensity(resampled_file, final_file)
        if not success:
            return False
        
        return True
    
    except Exception as e:
        print(f"Error processing subject from {input_file}: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_nifti_files(input_dir):
    """Find all .nii and .nii.gz files in the input directory (recursively)"""
    try:
        # Windows compatibility: use Path.glob instead of glob with recursive=True
        data_dir_path = Path(os.path.abspath(input_dir))
        nii_files = list(data_dir_path.glob('**/*.nii'))
        nii_files += list(data_dir_path.glob('**/*.nii.gz'))
        nii_files = [str(f) for f in nii_files]
    except Exception as e:
        print(f"Error finding NIfTI files: {e}")
        # Fallback to regular glob
        nii_files = list(Path(input_dir).glob('**/*.nii')) + list(Path(input_dir).glob('**/*.nii.gz'))
        nii_files = [str(f) for f in nii_files]
    
    return nii_files


def process_dataset(input_dir, output_dir, reference_file=None, use_hd_bet=False, 
                   use_flirt=True, num_processes=1, gpu_ids=None):
    """
    Process all MRI scans in the input directory
    
    Args:
        input_dir: Directory containing input NIfTI files
        output_dir: Directory to save processed files
        reference_file: Path to reference template
        use_hd_bet: Whether to use HD-BET for skull stripping
        use_flirt: Whether to use FSL FLIRT for registration
        num_processes: Number of parallel processes to use
        gpu_ids: List of GPU device IDs to use
    """
    # Create output directory with absolute path
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all NIfTI files
    input_files = find_nifti_files(input_dir)
    print(f"Found {len(input_files)} NIfTI files")
    
    if not input_files:
        print(f"No NIfTI files found in {input_dir}")
        return
    
    # Convert reference file to absolute path if provided
    if reference_file:
        reference_file = os.path.abspath(reference_file)
    
    # Prepare arguments for parallel processing
    process_args = []
    for i, input_file in enumerate(input_files):
        gpu_id = gpu_ids[i % len(gpu_ids)] if gpu_ids else -1
        
        args = {
            'input_file': os.path.abspath(input_file),
            'output_dir': output_dir,
            'reference_file': reference_file,
            'use_hd_bet': use_hd_bet,
            'use_flirt': use_flirt,
            'gpu_id': gpu_id
        }
        process_args.append(args)
    
    # Process files in parallel
    successful = 0
    failed = 0
    
    # Use a smaller number of processes to avoid memory issues
    if num_processes > 1:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(tqdm(executor.map(process_single_subject, process_args), 
                               total=len(process_args), 
                               desc="Processing subjects"))
        
        for success in results:
            if success:
                successful += 1
            else:
                failed += 1
    else:
        # Process sequentially for debugging
        results = []
        for args in tqdm(process_args, desc="Processing subjects"):
            result = process_single_subject(args)
            results.append(result)
            if result:
                successful += 1
            else:
                failed += 1
    
    print(f"Preprocessing complete: {successful} successful, {failed} failed")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Preprocess MRI scans')
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input NIfTI files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save processed files')
    parser.add_argument('--reference_file', type=str, default=None,
                       help='Path to reference template (MNI152) for registration')
    parser.add_argument('--use_hd_bet', action='store_true',
                       help='Use HD-BET for skull stripping (requires HD-BET installation)')
    parser.add_argument('--use_flirt', action='store_true',
                       help='Use FSL FLIRT for registration (requires FSL installation)')
    parser.add_argument('--num_processes', type=int, default=1,
                       help='Number of parallel processes to use (default: 1)')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[-1],
                       help='GPU device IDs to use')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        reference_file=args.reference_file,
        use_hd_bet=args.use_hd_bet,
        use_flirt=args.use_flirt,
        num_processes=args.num_processes,
        gpu_ids=args.gpu_ids
    )


if __name__ == "__main__":
    main() 