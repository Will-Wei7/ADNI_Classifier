import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import os
import time
import copy # To save the best model
from sklearn.metrics import confusion_matrix, classification_report # For detailed evaluation

# --- Configuration ---
# Base directory where train/, validation/, test/ folders reside
# This should contain the AD/ CN/ MCI/ subfolders with patient ID folders inside
data_dir = 'ADNI_dataset_png_axial'
num_epochs = 25 # Adjust as needed
batch_size = 32 # Adjust based on your GPU memory
learning_rate = 0.001
# Specify the classes you want for binary classification
binary_classes = ['AD', 'CN'] # The folders to include

# --- Data Transformations ---
# Define transforms for the training, validation, and test sets
# Pre-trained models expect specific input sizes and normalization
# ResNet input size is typically 224x224
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(), # Data augmentation
        transforms.ToTensor(),
        # Normalization values for models pre-trained on ImageNet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
     'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def load_filtered_dataset(image_folder_path, target_classes, transform):
    """ Loads ImageFolder, filters for target classes, and returns dataset and class_to_idx mapping. """
    full_dataset = datasets.ImageFolder(image_folder_path, transform=transform)
    indices_to_keep = []
    original_class_to_idx = full_dataset.class_to_idx
    target_indices = {original_class_to_idx[cls] for cls in target_classes if cls in original_class_to_idx}

    if not target_indices:
         raise ValueError(f"None of the target classes {target_classes} found in {image_folder_path}")

    sorted_target_classes = sorted(list(target_classes))
    binary_class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted_target_classes)}
    # Removed print statement for binary mapping here

    for i, (path, target) in enumerate(full_dataset.samples):
        if target in target_indices:
            indices_to_keep.append(i)

    filtered_dataset = Subset(full_dataset, indices_to_keep)

    # Removed print statement for loaded samples here
    return filtered_dataset, binary_class_to_idx


# --- Load Datasets and Create Dataloaders ---
image_datasets = {}
dataloaders = {}
dataset_sizes = {}
binary_class_mappings = {} # Store the binary mapping for each split if needed

for x in ['train', 'validation', 'test']:
    image_folder_path = os.path.join(data_dir, x)
    if not os.path.isdir(image_folder_path):
        print(f"Warning: Directory not found - {image_folder_path}. Skipping this split.")
        continue
        
    try:
        dataset, mapping = load_filtered_dataset(image_folder_path, binary_classes, data_transforms[x])
        image_datasets[x] = dataset
        binary_class_mappings[x] = mapping # Store {'AD': 0, 'CN': 1} or similar
        dataloaders[x] = DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=4)
        dataset_sizes[x] = len(image_datasets[x])
    except ValueError as e:
         print(e)
    except Exception as e:
         print(f"Error loading data for {x}: {e}")


# Check if we loaded any data, especially training data
if 'train' not in dataloaders:
    print("Error: Training data could not be loaded. Please check data_dir and subfolder structure.")
    exit()

# Assuming the mapping is consistent across splits, get the definitive one from train
if 'train' in binary_class_mappings:
    class_to_idx = binary_class_mappings['train']
    idx_to_class = {v: k for k, v in class_to_idx.items()} # For interpreting output
    num_classes = len(class_to_idx)
    print(f"Final Binary Class Mapping: {class_to_idx}")
else:
    print("Error: Could not establish class mapping from training data.")
    exit()

# --- Model Definition ---
# Load a pre-trained ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# ResNet models have RGB input, PNGs are grayscale. We need to adapt.
# Option 1: Modify first layer to accept 1 channel (requires careful weight handling)
# Option 2: Convert grayscale images to pseudo-RGB by duplicating the channel
# Let's use Option 2 with a transform adjustment (modify ToTensor if needed or handle in dataset)
# The current ToTensor transform handles grayscale correctly for Normalize if source is L mode PNG.

# Freeze all layers except the final classification layer
for param in model.parameters():
    param.requires_grad = False

# Get the number of input features for the classifier layer
num_ftrs = model.fc.in_features

# Replace the final fully connected layer for binary classification (1 output unit)
model.fc = nn.Linear(num_ftrs, 1) # Output size is 1 for BCEWithLogitsLoss

# --- Setup Device, Loss, Optimizer ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Binary Cross-Entropy loss with logits (handles sigmoid internally)
criterion = nn.BCEWithLogitsLoss()

# Optimizer - only optimize the parameters of the final layer
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

# Learning rate scheduler (optional)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# --- Training Function (Cleaned) ---
def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Removed redundant dataset loading for mapping here
    # The global class_to_idx and idx_to_class are used in the loop below

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase not in dataloaders:
                 print(f"Skipping {phase} phase.")
                 continue

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels_original in dataloaders[phase]:
                inputs = inputs.to(device)
                # Map original labels to binary labels using globally defined mappings
                # Ensure the original label exists in idx_to_class before proceeding
                labels_binary_list = []
                valid_indices = [] # Track indices corresponding to valid labels in the batch
                for idx, l in enumerate(labels_original):
                    item_label = l.item()
                    if item_label in idx_to_class and idx_to_class[item_label] in class_to_idx:
                         labels_binary_list.append(class_to_idx[idx_to_class[item_label]])
                         valid_indices.append(idx)
                    else:
                         # This case should ideally not happen if Subset filtering worked correctly,
                         # but handle defensively. Skip this sample if label is unexpected.
                         print(f"Warning: Skipping sample with unexpected original label {item_label}")


                # Only proceed if there are valid labels in the batch
                if not valid_indices:
                     continue

                # Filter inputs to match valid labels
                inputs = inputs[valid_indices]
                labels_binary = torch.tensor(labels_binary_list, dtype=torch.float32).unsqueeze(1).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_binary)
                    preds = torch.round(torch.sigmoid(outputs))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0) # Use size of filtered inputs
                running_corrects += torch.sum(preds == labels_binary.data)

            # Adjust dataset size for loss/acc calculation if samples were skipped (unlikely but defensive)
            effective_dataset_size = dataset_sizes[phase] # Assume full size unless specific filtering needed

            epoch_loss = running_loss / effective_dataset_size
            epoch_acc = running_corrects.double() / effective_dataset_size

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

# --- Evaluation Function ---
def evaluate_model(model, dataloader, criterion, device, class_to_idx, idx_to_class):
    """Evaluates the model on a given dataloader."""
    since = time.time()
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    dataset_size = len(dataloader.dataset)

    print(f"\n--- Evaluating on {dataset_size} samples ---")

    # Iterate over data.
    for inputs, labels_original in dataloader:
        inputs = inputs.to(device)

        # Map original labels to binary labels using the provided mappings
        labels_binary_list = []
        valid_indices = []
        for idx, l in enumerate(labels_original):
             item_label = l.item()
             if item_label in idx_to_class and idx_to_class[item_label] in class_to_idx:
                 labels_binary_list.append(class_to_idx[idx_to_class[item_label]])
                 valid_indices.append(idx)

        if not valid_indices:
             continue # Skip batch if no valid labels

        inputs = inputs[valid_indices]
        labels_binary = torch.tensor(labels_binary_list, dtype=torch.float32).unsqueeze(1).to(device)

        # Disable gradient calculation
        with torch.no_grad():
            outputs = model(inputs) # Logits output
            loss = criterion(outputs, labels_binary)
            preds = torch.round(torch.sigmoid(outputs)) # Threshold at 0.5

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels_binary.data)

        # Store predictions and labels for detailed report
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_binary.cpu().numpy())


    eval_loss = running_loss / dataset_size
    eval_acc = running_corrects.double() / dataset_size

    time_elapsed = time.time() - since
    print(f'Evaluation complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Loss: {eval_loss:.4f} Acc: {eval_acc:.4f}')

    # Detailed Classification Report
    print("\nClassification Report:")
    # Ensure labels are integers for classification_report
    all_labels_int = [int(l[0]) for l in all_labels]
    all_preds_int = [int(p[0]) for p in all_preds]
    target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())] # Get class names 'AD', 'CN'
    try:
        print(classification_report(all_labels_int, all_preds_int, target_names=target_names))
    except ValueError as e:
         print(f"Could not generate classification report: {e}")
         print("Ensure both classes are present in the predictions and labels.")


    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels_int, all_preds_int)
    print(f"Labels: {target_names}")
    print(cm)

    return eval_loss, eval_acc

# --- Start Training ---
if __name__ == '__main__':
    # --- Load Datasets and Create Dataloaders ---
    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    binary_class_mappings = {}

    print("--- Loading and Filtering Datasets ---")
    for x in ['train', 'validation', 'test']:
        image_folder_path = os.path.join(data_dir, x)
        if not os.path.isdir(image_folder_path):
            print(f"Warning: Directory not found - {image_folder_path}. Skipping this split.")
            continue

        try:
            # Call the cleaned load_filtered_dataset function
            dataset, mapping = load_filtered_dataset(image_folder_path, binary_classes, data_transforms[x])
            image_datasets[x] = dataset
            binary_class_mappings[x] = mapping
            dataloaders[x] = DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=4)
            dataset_sizes[x] = len(image_datasets[x])
            print(f"Split '{x}': Loaded {dataset_sizes[x]} samples. Binary Mapping: {mapping}") # Print summary here
        except ValueError as e:
             print(f"Error loading {x} data: {e}")
        except Exception as e:
             print(f"Error loading data for {x}: {e}")

    if 'train' not in dataloaders:
        print("Error: Training data could not be loaded. Exiting.")
        exit()

    # Establish final mapping (should be consistent if loading worked)
    if 'train' in binary_class_mappings:
        class_to_idx = binary_class_mappings['train']
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)
        print(f"\nFinal Binary Class Mapping Used: {class_to_idx}")
    else:
        print("Error: Could not establish class mapping from training data. Exiting.")
        exit()

    # --- Model Definition ---
    print("\n--- Defining Model ---")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    # --- Setup Device, Loss, Optimizer ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") # Print device once
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    print("\n--- Starting Training ---")
    if 'train' in dataloaders and model and criterion and optimizer:
         model_ft = train_model(model, criterion, optimizer, num_epochs=num_epochs)

         # --- Save the best model ---
         save_path = 'ad_cn_binary_classifier_best.pth'
         torch.save(model_ft.state_dict(), save_path)
         print(f"\nBest model weights saved to {save_path}")

         # --- Evaluate on Test Set ---
         if 'test' in dataloaders:
             # Ensure the mappings are available
             if class_to_idx and idx_to_class:
                 evaluate_model(model_ft, dataloaders['test'], criterion, device, class_to_idx, idx_to_class)
             else:
                 print("Could not evaluate on test set: class mappings not available.")
         else:
             print("Test dataloader not found. Skipping test set evaluation.")

    else:
        print("Could not start training due to missing components.")