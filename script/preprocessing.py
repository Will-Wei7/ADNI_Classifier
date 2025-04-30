import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import nibabel as nib
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import shutil

# --- Configuration ---
# Root directory containing the train/validation/test splits
# Ensure this path is exactly correct for your mounted Google Drive
NIFTI_ROOT_DIR = '/ADNI_dataset/'
OUTPUT_ROOT_DIR = '/ADNI_dataset_processed_pytorch/' # Desired output directory

DTYPES = ["AD", "CN", "MCI"] # Labels match user's structure
SPLITS = ["train", "validation", "test"]

TARGET_SIZE = (192, 160) # Target Height, Width for PyTorch (PIL uses W, H)

# Slice selection parameters (from original script 5_select_sequences_ttv.py)
SLICE_SELECTION_RANGES = {
    192: (95, 140),
    240: (100, 180),
    256: (120, 200),
    # Add more shapes if needed based on your specific data
}
PERFORM_SLICE_SELECTION = True # Set to False to process all slices


def normalize_slice_numpy(slice_data):
    """Normalizes a single slice numpy array like in original script 3"""
    max_val = np.max(slice_data)
    if max_val > 0:
        normalized_slice = (slice_data / max_val) * 255.0
    else:
        normalized_slice = slice_data # Avoid division by zero
    return normalized_slice.astype(np.uint8) # Convert to uint8 for saving/PIL

def process_nifti_file(nifti_path, output_label_dir):
    """Loads NIfTI, extracts/normalizes/selects/resizes slices, saves as PNG."""
    try:
        nifti_filename = os.path.basename(nifti_path)
        nifti_filename_stem = nifti_filename.split('.nii')[0] # Get filename without extension

        # --- More robust loading ---
        try:
            img = nib.load(nifti_path)
        except FileNotFoundError:
            print(f"  Error: Nibabel could not find file at {nifti_path}. Check path and permissions.")
            return
        except nib.filebasedimages.ImageFileError as e:
            print(f"  Error: Nibabel cannot work out file type for {nifti_path}. File might be corrupted or wrong format. Details: {e}")
            return
        except Exception as e:
             print(f"  Error: Unexpected error loading {nifti_path} with nibabel: {e}")
             return

        data = np.asanyarray(img.dataobj)

        if data.ndim != 3:
             print(f"  Skipping {nifti_path}: Expected 3D data, got {data.ndim}D.")
             return

        # Assuming axial slices are along the first axis (like original script)
        AXIAL_AXIS = 0 # Change if needed (0: Sagittal, 1: Coronal, 2: Axial in nibabel standard?) - CHECK YOUR DATA ORIENTATION
        if data.shape[AXIAL_AXIS] < 10:
             print(f"  Warning: Low slice count ({data.shape[AXIAL_AXIS]}) on axis {AXIAL_AXIS} for {nifti_path}. Check slice axis.")
        n_slices = data.shape[AXIAL_AXIS]


        # Determine slice range for selection
        if PERFORM_SLICE_SELECTION:
            selected_range = SLICE_SELECTION_RANGES.get(n_slices)
            if selected_range is None:
                # Using a default percentile range if exact shape not found
                start_slice = int(n_slices * 0.45) # ~45th percentile
                end_slice = int(n_slices * 0.80)   # ~80th percentile
                # print(f"  Info: No exact slice range for shape {data.shape} (axis {AXIAL_AXIS}: {n_slices}). Using default: {start_slice}-{end_slice}")
            else:
                start_slice, end_slice = selected_range[0], selected_range[1]
        else:
             start_slice, end_slice = 0, n_slices - 1

        # Ensure output directory for this label exists
        os.makedirs(output_label_dir, exist_ok=True)

        # Define the resize transform using Pillow (matching original resize script)
        target_size_pil = (TARGET_SIZE[1], TARGET_SIZE[0]) # PIL resize takes (width, height)

        slice_count = 0
        for i in range(n_slices):
            # Apply slice selection (using original script's reverse index logic)
            original_index = n_slices - 1 - i
            if PERFORM_SLICE_SELECTION and not (start_slice <= original_index <= end_slice):
                continue

            # Extract slice based on the chosen axis
            if AXIAL_AXIS == 0:
                 slice_data = data[i, :, :]
            elif AXIAL_AXIS == 1:
                 slice_data = data[:, i, :]
            elif AXIAL_AXIS == 2:
                 slice_data = data[:, :, i]
            else:
                 print(f"  Error: Invalid AXIAL_AXIS {AXIAL_AXIS}")
                 return

            # Rotate if necessary (sometimes needed depending on NIfTI orientation)
            # slice_data = np.rot90(slice_data)

            # Normalize like the original script
            normalized_slice_np = normalize_slice_numpy(slice_data)

            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(normalized_slice_np, mode='L')

            # Resize using PIL with Lanczos resampling
            resized_pil_img = pil_img.resize(target_size_pil, resample=Image.LANCZOS)

            # Save the resized slice as PNG
            output_filename = f"{nifti_filename_stem}_slice_{original_index}.png"
            output_path = os.path.join(output_label_dir, output_filename)
            resized_pil_img.save(output_path)
            slice_count += 1

        if slice_count == 0 and PERFORM_SLICE_SELECTION and selected_range is not None:
              print(f"  Info: No slices selected for {nifti_path} (Total: {n_slices}, Range: {start_slice}-{end_slice}) based on defined ranges.")


    except Exception as e:
        print(f"--- Unhandled Error processing {nifti_path}: {e} ---")


if __name__ == '__main__':
    if not os.path.isdir(NIFTI_ROOT_DIR):
        print(f"Error: Input directory '{NIFTI_ROOT_DIR}' not found.")
        exit()

    # --- Use os.walk to find .nii/.nii.gz files ---
    for split in SPLITS:
        for dtype in DTYPES:
            input_label_dir = os.path.join(NIFTI_ROOT_DIR, split, dtype)
            output_label_dir = os.path.join(OUTPUT_ROOT_DIR, split, dtype)

            if not os.path.isdir(input_label_dir):
                print(f"Input directory not found, skipping: {input_label_dir}")
                continue

            print(f"\nProcessing NIfTI files in: {input_label_dir}")
            print(f"Outputting PNG slices to: {output_label_dir}")

            # Create the base output directory for this split/label
            os.makedirs(output_label_dir, exist_ok=True)

            file_count = 0
            # Walk through the directory to find NIFTI files
            for root, dirs, files in os.walk(input_label_dir):
                nifti_files_in_dir = [f for f in files if f.endswith('.nii') or f.endswith('.nii.gz')]
                if not nifti_files_in_dir:
                    continue

                # Process files found in this specific directory (root)
                # Wrap the list with tqdm for progress bar per directory
                for nifti_file in tqdm(nifti_files_in_dir, desc=f"Processing {os.path.relpath(root, input_label_dir)}", leave=False):
                    nifti_full_path = os.path.join(root, nifti_file)
                    # Process the file, saving slices directly into the main output label dir
                    process_nifti_file(nifti_full_path, output_label_dir)
                    file_count += 1

            if file_count == 0:
                print(f"  No .nii/.nii.gz files found recursively within {input_label_dir}")


    print("\nPreprocessing finished.")
    print(f"Processed images saved under: {OUTPUT_ROOT_DIR}")