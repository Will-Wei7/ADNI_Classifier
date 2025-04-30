import os
import csv
import random
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd

# Define paths
source_folder = '648_MPR_n3_scaled'
csv_file = os.path.join('csv', 'filtered_data_mpr_gradwarp.csv')
output_folder = 'ADNI_dataset'

# Create output directory structure
os.makedirs(output_folder, exist_ok=True)
for split in ['train', 'validation', 'test']:
    for class_name in ['AD', 'CN', 'MCI']:
        os.makedirs(os.path.join(output_folder, split, class_name), exist_ok=True)

# Read the CSV file to get subject ID to class mapping
subject_class_map = {}
image_id_subject_map = {}

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if len(row) >= 3:
            image_id = row[0]
            subject_id = row[1]
            class_label = row[2]
            subject_class_map[subject_id] = class_label
            image_id_subject_map[image_id] = subject_id

print(f"Found {len(subject_class_map)} unique subjects with class labels")

# Find all .nii files in the source directory
nii_files = []
for root, _, files in os.walk(os.path.join(source_folder, 'ADNI')):
    for file in files:
        if file.endswith('.nii'):
            nii_files.append(os.path.join(root, file))

print(f"Found {len(nii_files)} .nii files")

# Extract subject IDs from file paths
file_subject_map = {}
file_image_id_map = {}
for nii_file in nii_files:
    # Extract subject ID from path
    # Path format: 648_MPR_n3_scaled/ADNI/006_S_0681/MPR__GradWarp__B1_Correction__N3__Scaled/2006-08-31_13_42_29.0/I92305/ADNI_006_S_0681_MR_...
    parts = nii_file.split(os.sep)
    if len(parts) >= 3:
        subject_id = parts[2]  # Should be in format like "006_S_0681"
        file_subject_map[nii_file] = subject_id
    
    # Extract image ID from filename (e.g., "I92305")
    if len(parts) >= 6:
        image_id = parts[5]  # The image ID folder name
        file_image_id_map[nii_file] = image_id

print(f"Matched {len(file_subject_map)} files with subject IDs")

# Group files by class
class_files = {'AD': [], 'CN': [], 'MCI': []}
for nii_file, subject_id in file_subject_map.items():
    if subject_id in subject_class_map:
        class_label = subject_class_map[subject_id]
        if class_label in class_files:
            class_files[class_label].append(nii_file)

# Print statistics
for class_label, files in class_files.items():
    print(f"Class {class_label}: {len(files)} files")

# Set random seed for reproducibility
random.seed(42)

# Split each class into train/validation/test sets (83/7/10)
splits = {}
for class_label, files in class_files.items():
    # Shuffle files to randomize
    random.shuffle(files)
    
    # Calculate split sizes
    num_files = len(files)
    train_size = int(0.83 * num_files)
    val_size = int(0.07 * num_files)
    # test_size will be the remainder
    
    # Split files
    train_files = files[:train_size]
    val_files = files[train_size:train_size + val_size]
    test_files = files[train_size + val_size:]
    
    splits[class_label] = {
        'train': train_files,
        'validation': val_files,
        'test': test_files
    }
    
    print(f"Class {class_label} - Train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")

# Add an option to use symbolic links instead of copying files (saves disk space)
use_symlinks = False  # Set to True to use symbolic links instead of copying

# Copy files to the new directory structure
for class_label, split_files in splits.items():
    for split, files in split_files.items():
        dest_dir = os.path.join(output_folder, split, class_label)
        
        if use_symlinks:
            print(f"Creating symbolic links for {len(files)} files in {dest_dir}")
        else:
            print(f"Copying {len(files)} files to {dest_dir}")
            
        for src_file in tqdm(files):
            # Get the subject ID and image ID for a more informative filename
            subject_id = file_subject_map.get(src_file, "unknown")
            image_id = os.path.basename(src_file)
            
            # Create a simplified filename with subject and image info
            dest_filename = f"{subject_id}_{os.path.basename(src_file)}"
            dest_file = os.path.join(dest_dir, dest_filename)
            
            # Copy the file or create a symbolic link
            if use_symlinks:
                try:
                    os.symlink(os.path.abspath(src_file), dest_file)
                except OSError:
                    # If symbolic link fails, fallback to copy
                    shutil.copy2(src_file, dest_file)
            else:
                shutil.copy2(src_file, dest_file)

print("Dataset split and organization complete!")

# Print final dataset statistics
for split in ['train', 'validation', 'test']:
    print(f"\n{split.capitalize()} set:")
    for class_name in ['AD', 'CN', 'MCI']:
        class_dir = os.path.join(output_folder, split, class_name)
        num_files = len([f for f in os.listdir(class_dir) if f.endswith('.nii')])
        print(f"  {class_name}: {num_files} files") 