# Action-1: Tool Code Generation.
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# --- Configuration ---

# 1. Path to the CSV file listing the selected subjects and their classes
csv_file_path = 'download (1).csv' # Using the file user confirmed has 645 subjects

# 2. Base path where the original subject folders (e.g., '002_S_0295') are located
#    IMPORTANT: Use raw string (r"...") or forward slashes for Windows paths
#    VERIFY THIS PATH IS STILL CORRECT
base_data_path = Path(r"E:\EECS_PROJECT\MPR\ADNI")

# 3. Path where the new 'train' and 'test' folders will be created
#    A new folder 'ADNI_split' will be created here containing 'train' and 'test'
#    VERIFY THIS PATH IS STILL CORRECT
output_base_path = Path(r"E:\EECS_PROJECT\MPR\ADNI_split")
output_split_folder_name = "ADNI_split" # Name for the folder holding train/test sets

# 4. Test set size (e.g., 0.2 for 20%)
test_set_fraction = 0.2

# 5. Set to True to move folders, False to copy folders (Copying is safer!)
move_files = False # Keep as False (copy) unless you are sure

# --- End Configuration ---

print(f"--- Starting Data Split based on '{csv_file_path}' (shutil.copytree - Robust Skip) ---")
print(f"Using subject list from: {csv_file_path}")
print(f"Base data path: {base_data_path}")
print(f"Output path: {output_base_path / output_split_folder_name}")
print(f"Test set fraction: {test_set_fraction}")
print(f"Move files instead of copy: {move_files}")
print("-" * 25)

# --- Load Subject List ---
try:
    # Load the CSV file specified by the user
    subject_df = pd.read_csv(csv_file_path)
    print(f"Loaded {len(subject_df)} subjects from {csv_file_path}")
except FileNotFoundError:
    print(f"ERROR: Cannot find CSV file '{csv_file_path}'. Please check the path.")
    sys.exit(1) # Stop execution if file not found
except Exception as e:
    print(f"ERROR: Could not read CSV file '{csv_file_path}'. Error: {e}")
    sys.exit(1)

# Use column names confirmed during inspection
subject_col = 'Subject ID'
group_col = 'Research Group'

if subject_col not in subject_df.columns or group_col not in subject_df.columns:
    print(f"ERROR: Required columns '{subject_col}' or '{group_col}' not found in '{csv_file_path}'.")
    print(f"Available columns: {subject_df.columns.tolist()}")
    sys.exit(1)

# Check if it's one row per subject (as inspection suggested)
if subject_df[subject_col].nunique() != len(subject_df):
     print(f"Warning: The file '{csv_file_path}' contains multiple rows per Subject ID.")
     print("This script assumes one row per subject. Please ensure this CSV contains your final unique subject list.")
     # For now, proceed assuming the user provided the final list.

subjects = subject_df[subject_col].tolist()
labels = subject_df[group_col].tolist()
unique_labels = sorted(subject_df[group_col].unique())

# --- Perform Stratified Split ---
try:
    train_subjects, test_subjects, train_labels, test_labels = train_test_split(
        subjects,
        labels,
        test_size=test_set_fraction,
        random_state=42,  # for reproducibility
        stratify=labels    # Ensures class proportions are maintained
    )
    print(f"\nSplitting into {len(train_subjects)} training subjects and {len(test_subjects)} testing subjects.")
    print("Expected class distribution in training set:")
    print(pd.Series(train_labels).value_counts())
    print("Expected class distribution in test set:")
    print(pd.Series(test_labels).value_counts())

except ValueError as e:
     print(f"ERROR: Failed to split data. Check class distribution. Error: {e}")
     print("\nPlease check the class distribution loaded from the CSV:")
     print(subject_df[group_col].value_counts())
     sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to split data. Error: {e}")
    sys.exit(1)

# --- Create Output Directories ---
output_split_path = output_base_path / output_split_folder_name
output_train_path = output_split_path / "train"
output_test_path = output_split_path / "test"

print(f"\nCreating output directories at: {output_split_path}")
try:
    for label in unique_labels:
        os.makedirs(output_train_path / label, exist_ok=True)
        os.makedirs(output_test_path / label, exist_ok=True)
    print("Output directories created successfully.")
except OSError as e:
     print(f"ERROR: Could not create output directories. Check permissions or path. Error: {e}")
     sys.exit(1)


# --- Define Action (Copy or Move) ---
action_func = shutil.copytree if not move_files else shutil.move
action_name = "Copying" if not move_files else "Moving"

# --- Process Train Set ---
print(f"\nProcessing training set folders...")
train_errors = 0
train_skipped_exist = 0
train_skipped_missing = 0
for subject, label in zip(train_subjects, train_labels):
    source_path = base_data_path / subject
    dest_path = output_train_path / label / subject

    # Check if source exists first
    if not source_path.exists() or not source_path.is_dir():
        print(f"  Warning: Source folder not found, skipping: {source_path}")
        train_skipped_missing += 1
        continue # Skip to next subject

    # Check if destination already exists
    if dest_path.exists():
        # print(f"  Info: Destination path already exists, skipping {action_name}: {dest_path}") # Optional info message
        train_skipped_exist += 1
        continue # Skip if destination already exists

    # If source exists and destination does not, attempt copy/move
    try:
        action_func(source_path, dest_path)
        # print(f"  {action_name} {source_path} -> {dest_path}") # Uncomment for verbose output
    except Exception as e:
        print(f"  ERROR {action_name} {subject} to {dest_path}: {e}")
        # Attempt to clean up potentially partially created destination folder on error
        if dest_path.exists():
             try:
                 shutil.rmtree(dest_path)
                 print(f"  INFO: Cleaned up partially created destination folder on error: {dest_path}")
             except Exception as cleanup_e:
                 print(f"  WARNING: Failed to cleanup destination folder after error: {cleanup_e}")
        train_errors += 1
print(f"Processing training set complete. Errors: {train_errors}, Skipped (Destination Exists): {train_skipped_exist}, Skipped (Source Missing): {train_skipped_missing}")


# --- Process Test Set ---
print(f"\nProcessing test set folders...")
test_errors = 0
test_skipped_exist = 0
test_skipped_missing = 0
for subject, label in zip(test_subjects, test_labels):
    source_path = base_data_path / subject
    dest_path = output_test_path / label / subject

    # Check if source exists first
    if not source_path.exists() or not source_path.is_dir():
        print(f"  Warning: Source folder not found, skipping: {source_path}")
        test_skipped_missing += 1
        continue # Skip to next subject

    # Check if destination already exists
    if dest_path.exists():
        # print(f"  Info: Destination path already exists, skipping {action_name}: {dest_path}") # Optional info message
        test_skipped_exist += 1
        continue # Skip if destination already exists

    # If source exists and destination does not, attempt copy/move
    try:
        action_func(source_path, dest_path)
        # print(f"  {action_name} {source_path} -> {dest_path}") # Uncomment for verbose output
    except Exception as e:
        print(f"  ERROR {action_name} {subject} to {dest_path}: {e}")
        # Attempt to clean up potentially partially created destination folder on error
        if dest_path.exists():
             try:
                 shutil.rmtree(dest_path)
                 print(f"  INFO: Cleaned up partially created destination folder on error: {dest_path}")
             except Exception as cleanup_e:
                 print(f"  WARNING: Failed to cleanup destination folder after error: {cleanup_e}")
        test_errors += 1
print(f"Processing test set complete. Errors: {test_errors}, Skipped (Destination Exists): {test_skipped_exist}, Skipped (Source Missing): {test_skipped_missing}")

print("\n--- Data Split Finished ---")
total_skipped_missing = train_skipped_missing + test_skipped_missing
if total_skipped_missing > 0:
    print(f"Warning: {total_skipped_missing} subjects were skipped because their source folders were not found in '{base_data_path}'.")
    print("         Please verify your downloaded data matches the list in '{csv_file_path}'.")
if train_errors > 0 or test_errors > 0:
    print(f"Warning: Encountered {train_errors + test_errors} errors during file operations. Please check logs above for details.")