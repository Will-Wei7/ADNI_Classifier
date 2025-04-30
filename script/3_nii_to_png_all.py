import os
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm # Optional: for progress bar
import uuid  # For generating shorter unique IDs
import math

# --- Configuration ---
# Set the path to your main dataset directory containing train/validation/test
input_base_dir = 'ADNI_dataset'
# Set the path where you want to save the PNG slices
output_base_dir = 'ADNI_dataset_png_axial'
# Use shorter folder names to reduce path length (True = shorter paths, False = original structure)
use_short_paths = True
# --- End Configuration ---

def normalize_slice(slice_data):
    """Normalizes a 2D slice to 0-255 range."""
    max_val = np.max(slice_data)
    min_val = np.min(slice_data)
    if max_val > min_val:
        # Normalize to 0-1 range first
        normalized = (slice_data - min_val) / (max_val - min_val)
        # Scale to 0-255 and convert to unsigned 8-bit integer
        scaled = (normalized * 255.0).astype(np.uint8)
        return scaled
    elif max_val > 0: # Handle case where slice might be uniform non-zero
         # Scale based on max value, avoids division by zero if min==max
         scaled = (slice_data / max_val * 255.0).astype(np.uint8)
         return scaled
    else: # Handle case where slice is all zeros
        return slice_data.astype(np.uint8) # Return as is (all zeros)

def convert_nii_to_png_slices(nifti_file_path, output_folder_path):
    """Loads a NIfTI file, extracts axial slices from 30% to 50% depth, and saves them as PNGs."""
    try:
        # First check if the file exists
        if not os.path.exists(nifti_file_path):
            print(f"Error: File does not exist: {nifti_file_path}")
            return False

        # Load the NIfTI file
        img = nib.load(nifti_file_path)
        data = np.asanyarray(img.dataobj) # Get data as numpy array

        # Assuming the first dimension corresponds to axial slices
        num_slices = data.shape[0]

        # --- Calculate indices for the 30% to 50% slice range ---
        # Start index is 40% of the way down (e.g., slice 30 for 100 slices)
        start_index = math.floor(0.40 * num_slices)
        # End index is 50% of the way down (e.g., slice 50 for 100 slices)
        # The range function excludes the end index, so range(start, end) includes slices up to end-1.
        end_index = math.floor(0.50 * num_slices)

        # Handle edge case where start and end might be the same if num_slices is very small
        if start_index >= end_index:
             # Optionally, save just the single slice at start_index if they overlap,
             # or save nothing. Let's save nothing in this edge case.
             print(f"Warning: No slices found in 30%-50% range for {os.path.basename(nifti_file_path)} (num_slices={num_slices}). Skipping.")
             return True # Return True as there was no error, just no slices to save.
        # --- End calculation ---

        # Ensure the output directory exists
        os.makedirs(output_folder_path, exist_ok=True)

        slices_saved_count = 0
        # Extract, normalize, and save only the selected slices in the 30%-50% range
        for i in range(start_index, end_index):
            slice_2d = data[i, :, :]
            slice_2d_rotated = np.rot90(slice_2d) # Rotate 90 degrees
            normalized_slice = normalize_slice(slice_2d_rotated)
            pil_image = Image.fromarray(normalized_slice).convert('L')

            base_filename = os.path.basename(nifti_file_path)
            filename_without_ext = base_filename.replace('.nii.gz', '').replace('.nii', '')

            # Use a shorter output filename (logic from your provided script)
            if len(filename_without_ext) > 20:
                parts = filename_without_ext.split('_')
                if len(parts) >= 3:
                    short_name = '_'.join(parts[:3])
                else:
                    short_name = f"{filename_without_ext[:10]}_{filename_without_ext[-5:]}"
            else:
                short_name = filename_without_ext

            # Include original slice index 'i' in the filename
            output_png_filename = f"{short_name}_slice_{i:03d}.png"
            output_png_path = os.path.join(output_folder_path, output_png_filename)
            pil_image.save(output_png_path)
            slices_saved_count += 1

        # Optional: Print how many slices were saved in the range
        # print(f"Saved {slices_saved_count} slices (from index {start_index} to {end_index-1}) for {os.path.basename(nifti_file_path)}")

        return True
    except Exception as e:
        print(f"Error processing {nifti_file_path}: {e}")
        return False

# --- Main Execution ---
if __name__ == '__main__':
    print(f"Starting NIfTI to PNG conversion...")
    print(f"Input directory: {os.path.abspath(input_base_dir)}")
    print(f"Output directory: {os.path.abspath(output_base_dir)}")

    nifti_files_to_process = []

    # Walk through the input directory structure
    for root, dirs, files in os.walk(input_base_dir):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                nifti_file_path = os.path.join(root, file)
                # Determine corresponding output path, mirroring the structure
                relative_path = os.path.relpath(root, input_base_dir)
                
                # Create shorter output paths to avoid Windows path length limitations
                if use_short_paths:
                    # Extract meaningful parts of the path structure (category and group)
                    path_parts = relative_path.split(os.sep)
                    if len(path_parts) >= 2:
                        # Keep only the last two parts of the path (typically train/val/test and diagnosis)
                        short_relative_path = os.path.join(path_parts[-2], path_parts[-1])
                    else:
                        short_relative_path = relative_path
                        
                    # Create a shorter filename for the subfolder
                    filename_without_ext = file.replace('.nii.gz', '').replace('.nii', '')
                    # Extract just the patient ID or create a short hash
                    parts = filename_without_ext.split('_')
                    if len(parts) >= 3:
                        folder_name = '_'.join(parts[:3])  # e.g., 099_S_0533
                    else:
                        # Use first 15 chars to maintain some recognizability
                        folder_name = filename_without_ext[:15]
                        
                    output_folder_for_nii = os.path.join(output_base_dir, short_relative_path, folder_name)
                else:
                    # Original path strategy (which may be too long for Windows)
                    filename_without_ext = file.replace('.nii.gz', '').replace('.nii', '')
                    output_folder_for_nii = os.path.join(output_base_dir, relative_path, filename_without_ext)

                nifti_files_to_process.append((nifti_file_path, output_folder_for_nii))

    print(f"Found {len(nifti_files_to_process)} NIfTI files to process.")

    # Process all found NIfTI files with a progress bar (optional)
    success_count = 0
    fail_count = 0
    for nifti_path, output_path in tqdm(nifti_files_to_process, desc="Converting NIfTI files"):
        if convert_nii_to_png_slices(nifti_path, output_path):
            success_count += 1
        else:
            fail_count += 1

    print(f"\nConversion finished.")
    print(f"Successfully converted: {success_count} NIfTI files.")
    print(f"Failed conversions: {fail_count}.")