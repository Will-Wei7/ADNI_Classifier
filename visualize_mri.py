import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import random

def find_nifti_files(data_dir, max_samples=5):
    """Find NIfTI files in the data directory."""
    nifti_files = []
    data_dir = Path(data_dir)
    
    for class_name in ["AD", "CN", "MCI"]:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Class directory {class_dir} does not exist. Skipping.")
            continue
            
        for subject_dir in class_dir.iterdir():
            if not subject_dir.is_dir():
                continue
                
            # Find the NIfTI file
            for root, _, files in os.walk(subject_dir):
                for file in files:
                    if file.endswith('.nii'):
                        nifti_files.append({
                            "path": Path(root) / file,
                            "class": class_name
                        })
                        break
                if len(nifti_files) >= max_samples:
                    break
            if len(nifti_files) >= max_samples:
                break
        if len(nifti_files) >= max_samples:
            break
    
    return nifti_files

def visualize_mri(nifti_file, class_name, save_dir="mri_visualizations"):
    """Visualize a 3D MRI scan."""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the NIfTI image
    img = nib.load(nifti_file).get_fdata()
    
    # Get the middle slice for each dimension
    mid_x = img.shape[0] // 2
    mid_y = img.shape[1] // 2
    mid_z = img.shape[2] // 2
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the middle slice for each dimension
    axes[0].imshow(img[mid_x, :, :], cmap='gray')
    axes[0].set_title(f'Sagittal (X={mid_x})')
    axes[0].axis('off')
    
    axes[1].imshow(img[:, mid_y, :], cmap='gray')
    axes[1].set_title(f'Coronal (Y={mid_y})')
    axes[1].axis('off')
    
    axes[2].imshow(img[:, :, mid_z], cmap='gray')
    axes[2].set_title(f'Axial (Z={mid_z})')
    axes[2].axis('off')
    
    # Add a title to the figure
    fig.suptitle(f'MRI Scan - {class_name}', fontsize=16)
    
    # Save the figure
    save_path = Path(save_dir) / f"{class_name}_{Path(nifti_file).stem}.png"
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved visualization to {save_path}")
    
    return save_path

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Find NIfTI files
    train_dir = "E:/EECS_PROJECT/ADNI_split/train"
    nifti_files = find_nifti_files(train_dir, max_samples=3)
    
    if not nifti_files:
        print("No NIfTI files found.")
        return
    
    # Visualize each NIfTI file
    for file_info in nifti_files:
        visualize_mri(file_info["path"], file_info["class"])
    
    print("Visualization complete.")

if __name__ == "__main__":
    main() 