import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
# Import the scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
# Import PEFT libraries
from peft import LoraConfig, get_peft_model # Removed TaskType import as it's not used now
from sklearn.model_selection import train_test_split # Used here for demonstration if needed, but assumes pre-split folders
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm # For progress bars
import matplotlib.pyplot as plt
import time # Import time for basic timing
# Required for Windows multiprocessing support
from multiprocessing import freeze_support

# --- Configuration ---
# Paths (!! IMPORTANT: Update these paths to match your environment !!)
DATA_DIR = 'ADNI_dataset/' # Main directory containing train/validation/test folders
PEFT_MODEL_SAVE_PATH = 'adni_vit_lora_adapters_binary' # Directory to save LoRA adapters (Binary Task)

# Model & Preprocessing Parameters
MODEL_NAME = "google/vit-base-patch16-224" # Pre-trained ViT model
IMAGE_SIZE = 224 # ViT standard input size
NUM_SLICES_PER_SCAN = 16 # Number of axial slices to extract from the middle of each scan
AXIS_TO_SLICE = 2 # 0: Sagittal, 1: Coronal, 2: Axial

# Training Parameters
# !! Changed for Binary Classification !!
NUM_CLASSES = 2 # AD vs CN
BATCH_SIZE = 8 # Keep reduced batch size
EPOCHS = 20 # Increase max epochs, rely on early stopping
LEARNING_RATE = 1e-4 # LoRA can often use a slightly higher LR than full fine-tuning
NUM_WORKERS = 2 # Number of worker processes for DataLoader (Set to 0 if issues)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LoRA Parameters
LORA_R = 16 # Rank of the update matrices (common values: 8, 16, 32)
LORA_ALPHA = 32 # LoRA scaling factor (often r or 2*r)
LORA_DROPOUT = 0.1
# Target modules for ViT (can vary slightly based on implementation)
# Common targets are query and value layers in attention blocks
LORA_TARGET_MODULES = ["query", "value"]

# Early Stopping Parameters
EARLY_STOPPING_PATIENCE = 5 # Stop after N epochs with no improvement in val F1
MIN_DELTA = 0.001 # Minimum change in validation F1 to be considered an improvement

# Class mapping - !! Updated for Binary Classification !!
CLASS_MAP = {'AD': 0, 'CN': 1} # Ignoring MCI
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

# --- Helper Functions ---

def normalize_slice(slice_data):
    """Normalize slice intensity to [0, 1]"""
    slice_data = slice_data.astype(np.float32)
    min_val = np.min(slice_data)
    max_val = np.max(slice_data)
    if max_val > min_val:
        slice_data = (slice_data - min_val) / (max_val - min_val)
    else:
        # Handle cases where the slice is constant (min == max)
        slice_data = np.zeros_like(slice_data) # Or set to a constant value like 0.5
    return slice_data

def extract_middle_slices(nifti_data, num_slices, axis):
    """Extracts a specified number of slices from the middle of a 3D volume along a given axis."""
    depth = nifti_data.shape[axis]
    if depth == 0:
        print(f"Warning: Scan depth is 0 along axis {axis}. Cannot extract slices.")
        # Return an empty array or handle as appropriate
        # Returning an array of zeros with expected shape for robustness downstream
        other_dims = [d for i, d in enumerate(nifti_data.shape) if i != axis]
        # Expected output shape after moveaxis: (num_slices, dim1, dim2)
        final_shape = (num_slices,) + tuple(other_dims)
        return np.zeros(final_shape, dtype=nifti_data.dtype)


    if depth < num_slices:
        # Handle cases where the scan has fewer slices than requested
        start_idx = 0
        end_idx = depth
        extracted = np.take(nifti_data, indices=range(start_idx, end_idx), axis=axis)
        # Padding logic
        num_pad = num_slices - depth
        if num_pad > 0:
            # Pad by repeating the last slice
            pad_slice_index = depth - 1
            pad_slice = np.expand_dims(np.take(nifti_data, indices=pad_slice_index, axis=axis), axis=axis)
            pad_array = np.repeat(pad_slice, num_pad, axis=axis)
            # Ensure shapes match for concatenation
            if extracted.shape[1:] != pad_array.shape[1:] or extracted.shape[0] != depth:
                 print(f"Shape mismatch during padding: extracted {extracted.shape}, pad_array {pad_array.shape}")
                 # Fallback: return zeros or raise error
                 final_shape = (num_slices,) + tuple(d for i, d in enumerate(nifti_data.shape) if i != axis)
                 return np.zeros(final_shape, dtype=nifti_data.dtype)

            extracted = np.concatenate((extracted, pad_array), axis=axis)
            # print(f"Warning: Scan depth ({depth}) less than num_slices ({num_slices}). Padded with last slice.")

    else:
        start_idx = (depth - num_slices) // 2
        end_idx = start_idx + num_slices
        extracted = np.take(nifti_data, indices=range(start_idx, end_idx), axis=axis)

    # Ensure the sliced axis is the first dimension for easy iteration
    if axis != 0:
         # Move the slicing axis to the front
         extracted = np.moveaxis(extracted, source=axis, destination=0)

    # Final sanity check on the number of slices
    if extracted.shape[0] != num_slices:
        print(f"Error: Final extracted slice count ({extracted.shape[0]}) does not match requested ({num_slices}).")
        # Fallback: return zeros or raise error
        other_dims = [d for i, d in enumerate(nifti_data.shape) if i != axis]
        final_shape = (num_slices,) + tuple(other_dims)
        return np.zeros(final_shape, dtype=nifti_data.dtype)


    return extracted

# --- Custom PyTorch Dataset ---

class ADNIDataset(Dataset):
    """
    Custom PyTorch Dataset for loading ADNI NIfTI files, extracting 2D slices,
    and applying transformations. Each item returned is a single 2D slice.
    Loads only classes specified in class_map.
    """
    def __init__(self, data_dir, class_map, num_slices_per_scan, axis_to_slice, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing class subfolders (e.g., 'ADNI_dataset/train/').
            class_map (dict): Dictionary mapping class names to load (e.g., {'AD':0, 'CN':1}) to integers.
            num_slices_per_scan (int): Number of slices to extract from each scan.
            axis_to_slice (int): Axis along which to slice the 3D volume (0, 1, or 2).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.class_map = class_map
        self.num_slices_per_scan = num_slices_per_scan
        self.axis_to_slice = axis_to_slice
        self.transform = transform
        self.file_paths = []
        self.labels = []
        self.slice_indices = [] # To track which slice of which file

        print(f"Loading data from: {self.data_dir}")
        print(f"Target classes: {list(self.class_map.keys())}")
        # Find all .nii and .nii.gz files and assign labels
        file_index_counter = 0
        # !! Only iterate through classes specified in the updated class_map !!
        for class_name, label in self.class_map.items():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory not found for class '{class_name}': {class_dir}")
                continue

            file_pattern = os.path.join(class_dir, '*.nii*')
            found_files = glob.glob(file_pattern)

            if not found_files:
                print(f"Warning: No NIfTI files found in {class_dir}")
                continue

            print(f"Found {len(found_files)} files for class '{class_name}'")
            for file_path in found_files:
                # Store file path and label once per file
                self.file_paths.append(file_path)
                self.labels.append(label)
                # Store indices for the slices belonging to this file
                for slice_idx in range(self.num_slices_per_scan):
                    self.slice_indices.append((file_index_counter, slice_idx)) # Tuple: (file_index, slice_index_within_file)
                file_index_counter += 1 # Increment file index

        if not self.file_paths:
             raise ValueError(f"No NIfTI files found for the specified classes {list(self.class_map.keys())} under {data_dir}")

        print(f"Total NIfTI files found for target classes: {len(self.file_paths)}")
        print(f"Total slices to be processed: {len(self.slice_indices)}")

    def __len__(self):
        """Returns the total number of slices across all scans."""
        return len(self.slice_indices)

    def __getitem__(self, idx):
        """
        Loads a NIfTI file, extracts one specific slice, preprocesses it,
        and returns the slice tensor and its corresponding label.
        """
        if idx >= len(self.slice_indices):
            raise IndexError("Index out of bounds")

        file_idx, slice_idx_in_file = self.slice_indices[idx]

        # Bounds check for file_idx
        if file_idx >= len(self.file_paths):
             print(f"Error: file_idx {file_idx} out of bounds for file_paths length {len(self.file_paths)}")
             return None, None # Indicate error

        file_path = self.file_paths[file_idx]
        label = self.labels[file_idx]

        try:
            # Load NIfTI file
            nifti_img = nib.load(file_path)
            # It's often safer to work with a copy if modifying data
            nifti_data = nifti_img.get_fdata(caching='unchanged').copy() # Load data as numpy array

            # Check if data is empty or invalid
            if nifti_data is None or nifti_data.size == 0:
                print(f"Warning: Loaded empty or invalid data from {file_path}")
                return None, None

            # Extract the specific slice needed
            # We extract all middle slices first, then select the one for this index
            middle_slices_data = extract_middle_slices(nifti_data, self.num_slices_per_scan, self.axis_to_slice)

            # Check if middle_slices_data is valid
            if middle_slices_data is None or middle_slices_data.size == 0:
                 print(f"Warning: Failed to extract middle slices for {file_path}")
                 return None, None

            if slice_idx_in_file >= middle_slices_data.shape[0]:
                 # This might happen if extract_middle_slices returned fewer slices due to small scan depth
                 # print(f"Warning: slice_idx_in_file {slice_idx_in_file} >= extracted slices {middle_slices_data.shape[0]} for {file_path}. Clamping index.")
                 slice_idx_in_file = middle_slices_data.shape[0] - 1 # Use the last available slice

            # Ensure slice_idx_in_file is valid after potential clamping
            if slice_idx_in_file < 0:
                print(f"Warning: Invalid slice_idx_in_file {slice_idx_in_file} after clamping for {file_path}")
                return None, None

            single_slice_data = middle_slices_data[slice_idx_in_file, :, :]

            # Normalize slice
            normalized_slice = normalize_slice(single_slice_data)

            # Convert to 3 channels (required by ViT) by repeating the channel
            # Ensure input is HxW before stacking
            if normalized_slice.ndim == 2:
                 three_channel_slice = np.stack([normalized_slice] * 3, axis=-1) # Creates HxWx3
            else:
                 print(f"Warning: Unexpected slice dimension {normalized_slice.ndim} for {file_path}, slice {slice_idx_in_file}")
                 return None, None


            # Apply transformations (expects PIL image or Tensor)
            # Convert numpy array (HxWx3, float 0-1) to PIL Image
            # Need to scale back to 0-255 uint8 for ToPILImage if input is float
            slice_image = transforms.ToPILImage()((three_channel_slice * 255).astype(np.uint8))


            if self.transform:
                slice_tensor = self.transform(slice_image)
            else:
                # Basic conversion to tensor if no other transforms
                slice_tensor = transforms.ToTensor()(slice_image) # Converts PIL (0-255) to Tensor (0.0-1.0)


            return slice_tensor, label

        except FileNotFoundError:
            print(f"Error: File not found {file_path}")
            return None, None # Indicate error
        except nib.filebasedimages.ImageFileError as e:
             print(f"Error loading NIfTI file {file_path}: {e}")
             return None, None # Indicate error
        except Exception as e:
            print(f"Error processing file {file_path}, slice {slice_idx_in_file}: {e}")
            # Potentially skip this sample or return dummy data
            return None, None # Indicate error


# --- Handle Data Loading Errors ---
# Custom collate function to filter out None samples caused by loading errors
def collate_fn(batch):
    # Filter out samples where __getitem__ returned None
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch: # If all samples in batch failed or the batch is empty
        return None, None
    try:
        # Use default collate to stack tensors
        return torch.utils.data.dataloader.default_collate(batch)
    except RuntimeError as e:
        print(f"RuntimeError during collation: {e}")
        print("This often happens if tensors in the batch have different shapes.")
        # Optionally print shapes for debugging:
        # for i, (tensor, label) in enumerate(batch):
        #     print(f" Sample {i} shape: {tensor.shape}")
        return None, None # Skip batch if collation fails
    except Exception as e:
        print(f"Error during collation: {e}")
        return None, None # Skip batch if collation fails


# --- Training and Validation Functions ---

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    batches_processed = 0
    batches_skipped = 0

    progress_bar = tqdm(loader, desc="Training", leave=False, mininterval=1.0) # Update progress bar less often
    for inputs, labels in progress_bar:
        # Skip batch if collate_fn returned None
        if inputs is None or labels is None:
            batches_skipped += 1
            # print("Skipping a batch due to loading/collation errors.")
            continue

        batches_processed += 1
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        logits = outputs.logits
        loss = criterion(logits, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(logits.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Update progress bar less frequently to avoid slowdown
        if batches_processed % 10 == 0 or batches_processed == len(loader):
             current_loss = running_loss / total_samples if total_samples > 0 else 0
             current_acc = correct_predictions / total_samples if total_samples > 0 else 0
             progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}", skip=batches_skipped)

    if batches_skipped > 0:
        print(f"Skipped {batches_skipped} batches during training epoch due to errors.")

    if total_samples == 0:
        print("Warning: No valid samples processed during training epoch.")
        return 0.0, 0.0 # Return zeros or handle appropriately

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    all_preds = []
    all_labels = []
    batches_skipped = 0

    with torch.no_grad(): # No need to track gradients during validation
        progress_bar = tqdm(loader, desc="Validation", leave=False, mininterval=1.0)
        for inputs, labels in progress_bar:
            if inputs is None or labels is None:
                batches_skipped += 1
                # print("Skipping a validation batch due to loading/collation errors.")
                continue

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(logits.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if batches_skipped > 0:
        print(f"Skipped {batches_skipped} batches during validation due to errors.")

    total_samples = len(all_labels)
    if total_samples == 0:
        print("Warning: No valid samples found during validation.")
        return 0.0, 0.0, 0.0, 0.0, 0.0, None # Return zeros or handle appropriately

    val_loss = running_loss / total_samples

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    # Use average='weighted' for overall performance, but also consider 'binary' or per-class later
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    # Ensure labels parameter includes all possible classes for a complete matrix
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(CLASS_MAP.values()))

    return val_loss, accuracy, precision, recall, f1, conf_matrix

# --- Helper function to print trainable parameters ---
def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

# --- Main Execution Guard ---
if __name__ == '__main__':
    # freeze_support() # Call this early for Windows executable freezing (optional otherwise)
    print(f"Script execution started. Using device: {DEVICE}")

    # --- Data Augmentation and Transforms ---
    print("Defining image processor and transforms...")
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    image_mean = processor.image_mean
    image_std = processor.image_std
    print(f"Using ImageNet mean: {image_mean}, std: {image_std}")

    # Add RandomAffine for more augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15), # Slightly increased rotation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Added translation and scaling
        transforms.ToTensor(), # Converts PIL Image (0-255) to Tensor (0.0-1.0)
        transforms.Normalize(mean=image_mean, std=image_std) # Normalize AFTER ToTensor
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ])
    print("Transforms defined.")

    # --- Create Datasets ---
    print("Creating Datasets...")
    start_time = time.time()
    try:
        train_dataset = ADNIDataset(
            data_dir=os.path.join(DATA_DIR, 'train'),
            class_map=CLASS_MAP, # Uses updated map {'AD': 0, 'CN': 1}
            num_slices_per_scan=NUM_SLICES_PER_SCAN,
            axis_to_slice=AXIS_TO_SLICE,
            transform=train_transform
        )

        val_dataset = ADNIDataset(
            data_dir=os.path.join(DATA_DIR, 'validation'),
            class_map=CLASS_MAP, # Uses updated map
            num_slices_per_scan=NUM_SLICES_PER_SCAN,
            axis_to_slice=AXIS_TO_SLICE,
            transform=val_test_transform
        )

        test_dataset = ADNIDataset(
            data_dir=os.path.join(DATA_DIR, 'test'),
            class_map=CLASS_MAP, # Uses updated map
            num_slices_per_scan=NUM_SLICES_PER_SCAN,
            axis_to_slice=AXIS_TO_SLICE,
            transform=val_test_transform
        )
        print(f"Datasets created in {time.time() - start_time:.2f} seconds.")
    except ValueError as e:
        print(f"Error creating datasets: {e}")
        exit() # Exit if datasets cannot be created
    except Exception as e:
        print(f"An unexpected error occurred during dataset creation: {e}")
        exit()


    # --- Calculate Class Weights for Imbalanced Data ---
    print("Calculating class weights...")
    start_time = time.time()
    # Need to iterate through the dataset to get labels, handle potential None items
    train_labels_list = []
    num_processed = 0
    # Iterate carefully, skipping None items returned by __getitem__
    for i in range(len(train_dataset)):
        try:
            # Accessing internal lists directly (less safe but faster for this step)
            file_idx, _ = train_dataset.slice_indices[i]
            if file_idx < len(train_dataset.labels):
                 train_labels_list.append(train_dataset.labels[file_idx])
            else:
                 print(f"Warning: file_idx {file_idx} out of bounds when collecting labels.")
            num_processed += 1
        except IndexError:
             print(f"Warning: IndexError accessing slice_indices at index {i}")
        except Exception as e:
            print(f"Error getting label for index {i} during weight calculation: {e}")

    if not train_labels_list:
         print("Error: Could not retrieve any labels from the training dataset to calculate class weights.")
         exit() # Exit if weights cannot be calculated

    try:
        # Ensure classes passed to compute_class_weight match the actual labels (0, 1)
        unique_labels = np.unique(train_labels_list)
        if not np.all(np.isin(unique_labels, list(CLASS_MAP.values()))):
            print(f"Warning: Unique labels found ({unique_labels}) don't match CLASS_MAP values ({list(CLASS_MAP.values())}). Check dataset loading.")

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_labels, # Use actual unique labels found
            y=train_labels_list
        )
        # Map weights back to the 0, 1 indices if necessary (should align if unique_labels is [0, 1])
        class_weights_tensor = torch.zeros(NUM_CLASSES, dtype=torch.float)
        for i, label in enumerate(unique_labels):
             if label in range(NUM_CLASSES): # Check if label is 0 or 1
                 class_weights_tensor[label] = class_weights[i]

        class_weights_tensor = class_weights_tensor.to(DEVICE)
        print(f"Calculated class weights (for classes {unique_labels}): {class_weights}")
        print(f"Class weights tensor (for classes 0, 1): {class_weights_tensor}")
        print(f"Class weight calculation took {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error calculating class weights: {e}")
        exit()


    # --- Create DataLoaders ---
    print("Creating DataLoaders...")
    start_time = time.time()
    # Use num_workers > 0 for faster loading, but can cause issues on Windows/Jupyter
    # Set pin_memory=True if using GPU
    # IMPORTANT: Set num_workers=0 if you still encounter multiprocessing issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn # Use custom collate function
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    print(f"DataLoaders created in {time.time() - start_time:.2f} seconds.")


    # --- Model Definition ---
    print("Loading pre-trained ViT model...")
    start_time = time.time()
    try:
        # Load the base model - !! Ensure NUM_CLASSES is 2 !!
        model = ViTForImageClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True # Necessary to load pre-trained weights with a new head
        )
        print(f"Base model loaded in {time.time() - start_time:.2f} seconds.")

        # --- Apply LoRA ---
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none", # or 'all' or 'lora_only'
            # task_type=TaskType.IMAGE_CLASSIFICATION # Removed
        )

        # Wrap the base model with PEFT adapter
        model = get_peft_model(model, lora_config)
        print("LoRA applied.")
        print_trainable_parameters(model) # Print parameter count after applying LoRA

        print("Moving PEFT model to device...")
        start_time = time.time()
        model.to(DEVICE)
        print(f"PEFT Model moved to {DEVICE} in {time.time() - start_time:.2f} seconds.")

    except Exception as e:
        print(f"Error loading or applying LoRA to the model: {e}")
        exit()


    # --- Loss Function and Optimizer ---
    print("Setting up loss function and optimizer...")
    # Use weighted CrossEntropyLoss for imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # AdamW is often recommended for Transformers
    # Optimizer should be created AFTER applying get_peft_model
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    # Add Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, min_lr=1e-7) # Monitor val F1 ('max' mode)
    print("Loss, optimizer, and scheduler ready.")


    # --- Training Loop ---
    print("\nStarting Training...")
    best_val_f1 = 0.0
    epochs_no_improve = 0 # Counter for early stopping
    best_epoch = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    val_f1s = []


    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"Epoch {epoch+1} Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc, val_prec, val_rec, val_f1, conf_matrix = validate(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch+1} Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        if conf_matrix is not None:
            print("Validation Confusion Matrix:")
            # Pretty print confusion matrix
            print("Labels:", [INV_CLASS_MAP[i] for i in range(NUM_CLASSES)]) # Uses updated INV_CLASS_MAP
            print(conf_matrix)

        # --- Learning Rate Scheduler Step ---
        # Step the scheduler based on validation F1 score
        scheduler.step(val_f1)
        # Optional: Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.1e}")


        # --- Early Stopping Check ---
        # Check if validation F1 improved
        if val_f1 > best_val_f1 + MIN_DELTA:
            print(f"Validation F1 improved ({best_val_f1:.4f} --> {val_f1:.4f}). Saving LoRA adapters...")
            try:
                # Save only the LoRA adapters
                model.save_pretrained(PEFT_MODEL_SAVE_PATH)
                best_val_f1 = val_f1
                epochs_no_improve = 0 # Reset counter
                best_epoch = epoch + 1
            except Exception as e:
                 print(f"Error saving LoRA adapters: {e}")
        else:
            epochs_no_improve += 1
            print(f"Validation F1 did not improve for {epochs_no_improve} epoch(s). Best F1: {best_val_f1:.4f} at epoch {best_epoch}")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break # Exit the training loop

        print(f"Epoch completed in {time.time() - epoch_start_time:.2f} seconds.")


    print(f"\nTraining Finished after {epoch+1} epochs.") # Use final epoch number

    # --- Plotting Training History (Optional) ---
    print("Plotting training history...")
    actual_epochs = len(train_losses) # Number of epochs actually run
    if actual_epochs > 0:
        try:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(range(1, actual_epochs + 1), train_losses, label='Training Loss', marker='o')
            plt.plot(range(1, actual_epochs + 1), val_losses, label='Validation Loss', marker='o')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss over Epochs (Binary)') # Updated title
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(range(1, actual_epochs + 1), train_accs, label='Training Accuracy', marker='o')
            plt.plot(range(1, actual_epochs + 1), val_accs, label='Validation Accuracy', marker='o')
            plt.plot(range(1, actual_epochs + 1), val_f1s, label='Validation F1-Score', linestyle='--', marker='x')
            if best_epoch > 0: # Mark the best epoch if found
                 plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
            plt.xlabel('Epochs')
            plt.ylabel('Metric Value')
            plt.title('Accuracy & F1 over Epochs (Binary)') # Updated title
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig('training_history_lora_binary.png') # Changed filename
            print("Training history plot saved to training_history_lora_binary.png")
            # plt.show() # Uncomment to display plot directly
        except Exception as e:
            print(f"Error plotting training history: {e}")
    else:
        print("No training history to plot.")


    # --- Final Evaluation on Test Set ---
    print("\nEvaluating on Test Set...")
    # Load the base model again and apply the saved adapters
    if os.path.exists(PEFT_MODEL_SAVE_PATH):
        print(f"Loading best LoRA adapters from {PEFT_MODEL_SAVE_PATH} (Epoch {best_epoch})")
        try:
            # Load the base model architecture
            base_model = ViTForImageClassification.from_pretrained(
                MODEL_NAME,
                num_labels=NUM_CLASSES, # Should be 2
                ignore_mismatched_sizes=True
            )
            # Load the PEFT model with adapters
            # Re-create the config used during training
            lora_config_load = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                target_modules=LORA_TARGET_MODULES,
                lora_dropout=LORA_DROPOUT,
                bias="none",
                # task_type=TaskType.IMAGE_CLASSIFICATION # Removed
            )
            model = get_peft_model(base_model, lora_config_load) # Re-apply config structure
            # Load the trained adapter weights
            model.load_adapter(PEFT_MODEL_SAVE_PATH, adapter_name="default")

            model.to(DEVICE) # Move the final model to device
            print("Best LoRA model loaded successfully.")
        except Exception as e:
            print(f"Error loading saved LoRA adapters: {e}. Evaluation might use the last state.")
            # Ensure the model used for eval is still on the correct device
            model.to(DEVICE)
    else:
        print("Warning: No saved LoRA adapters found. Evaluating with the model from the last epoch.")


    test_loss, test_acc, test_prec, test_rec, test_f1, test_conf_matrix = validate(model, test_loader, criterion, DEVICE)

    print("\n--- Test Set Results (Best LoRA Model - Binary) ---") # Updated title
    if test_conf_matrix is not None:
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision (Weighted): {test_prec:.4f}")
        print(f"Test Recall (Weighted): {test_rec:.4f}")
        print(f"Test F1-Score (Weighted): {test_f1:.4f}")
        print("Test Confusion Matrix:")
        print("Labels:", [INV_CLASS_MAP[i] for i in range(NUM_CLASSES)]) # Uses updated INV_CLASS_MAP
        print(test_conf_matrix)

        # Calculate and print per-class metrics for test set
        test_preds = []
        test_labels_list = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
                if inputs is None or labels is None: continue
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.logits.data, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_labels_list.extend(labels.cpu().numpy())

        if test_labels_list: # Ensure there are samples
            try:
                # Calculate binary metrics directly if preferred, or use per-class from multiclass setup
                # Option 1: Use standard binary metrics (if labels are 0 and 1)
                # precision_bin, recall_bin, f1_bin, _ = precision_recall_fscore_support(test_labels_list, test_preds, average='binary', pos_label=1) # Assuming CN=1 is positive class
                # print(f"\nBinary Metrics (Positive Class: CN):")
                # print(f"  Precision: {precision_bin:.4f}")
                # print(f"  Recall:    {recall_bin:.4f}")
                # print(f"  F1-Score:  {f1_bin:.4f}")

                # Option 2: Stick with per-class from multiclass setup (already calculated)
                precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
                    test_labels_list, test_preds, average=None, labels=list(CLASS_MAP.values()), zero_division=0
                )
                print("\nPer-Class Test Metrics:")
                for i, class_name in INV_CLASS_MAP.items():
                     # Check if class index exists in calculated metrics
                     if i < len(precision_per):
                         print(f"  Class: {class_name} (Support: {support_per[i]})")
                         print(f"    Precision: {precision_per[i]:.4f}")
                         print(f"    Recall:    {recall_per[i]:.4f}")
                         print(f"    F1-Score:  {f1_per[i]:.4f}")
                     else:
                         print(f"  Class: {class_name} (Support: 0) - Metrics not available (likely no samples predicted or true)")

            except Exception as e:
                print(f"Error calculating test metrics: {e}")
        else:
            print("No valid samples in test set to calculate metrics.")
    else:
        print("Could not evaluate on test set (likely due to data loading issues).")


    print("\nScript finished.")
