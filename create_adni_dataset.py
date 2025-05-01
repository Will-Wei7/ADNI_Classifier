import os
import shutil
import random
import tqdm
import numpy as np
from pathlib import Path

# Define source and destination paths
src_root = Path("AugmentedAlzheimerDataset")
dst_root = Path("ADNI_Dataset")

# Define split ratios
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

# Create destination directories
for split in ["train", "valid", "test"]:
    for class_name in ["AD", "CN", "EMCI", "LMCI"]:
        os.makedirs(dst_root / split / class_name, exist_ok=True)

# Process each class
class_counts = {}
for class_name in ["AD", "CN", "EMCI", "LMCI"]:
    # Get list of all files in this class
    class_files = list((src_root / class_name).glob("*.jpg"))
    class_counts[class_name] = len(class_files)
    
    # Shuffle the files
    random.shuffle(class_files)
    
    # Calculate split sizes
    train_size = int(len(class_files) * train_ratio)
    valid_size = int(len(class_files) * valid_ratio)
    
    # Split data
    train_files = class_files[:train_size]
    valid_files = class_files[train_size:train_size + valid_size]
    test_files = class_files[train_size + valid_size:]
    
    # Copy files to their respective directories
    print(f"Processing {class_name} class...")
    
    for files, split in zip([train_files, valid_files, test_files], ["train", "valid", "test"]):
        for src_file in tqdm.tqdm(files, desc=f"{split.capitalize()} split"):
            dst_file = dst_root / split / class_name / src_file.name
            shutil.copy2(src_file, dst_file)

# Print summary
print("\nDataset Summary:")
print("-" * 50)
for class_name, count in class_counts.items():
    train_count = len(list((dst_root / "train" / class_name).glob("*.jpg")))
    valid_count = len(list((dst_root / "valid" / class_name).glob("*.jpg")))
    test_count = len(list((dst_root / "test" / class_name).glob("*.jpg")))
    
    print(f"{class_name}:")
    print(f"  Total: {count}")
    print(f"  Train: {train_count} ({train_count/count:.1%})")
    print(f"  Valid: {valid_count} ({valid_count/count:.1%})")
    print(f"  Test: {test_count} ({test_count/count:.1%})")

print("\nDataset created successfully at:", dst_root.absolute()) 