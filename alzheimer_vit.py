import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from PIL import Image
import random
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Check for MPS (Metal Performance Shaders) for Apple Silicon Macs
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Print PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Custom Dataset class for Alzheimer's MRI images
class AlzheimerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['AD', 'CN', 'EMCI', 'LMCI']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith('.jpg'):
                        self.samples.append((os.path.join(class_dir, filename), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # ViT requires 224x224 input
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load the dataset
dataset = AlzheimerDataset(root_dir='AugmentedAlzheimerDataset', 
                          transform=data_transforms['train'])

# Split the dataset into train and validation sets
train_size = 0.8
val_size = 0.2
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, 
    [int(len(dataset) * train_size), len(dataset) - int(len(dataset) * train_size)]
)

# Apply different transforms to validation set
val_dataset.dataset.transform = data_transforms['val']

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Print dataset information
# print(f"Total number of samples: {len(dataset)}")
# print(f"Number of training samples: {len(train_dataset)}")
# print(f"Number of validation samples: {len(val_dataset)}")
# print(f"Number of classes: {len(dataset.classes)}")
# print(f"Classes: {dataset.classes}")

# Visualize class distribution
class_counts = {}
for _, label in dataset.samples:
    class_name = dataset.classes[label]
    if class_name in class_counts:
        class_counts[class_name] += 1
    else:
        class_counts[class_name] = 1

plt.figure(figsize=(10, 6))
plt.bar(class_counts.keys(), class_counts.values())
plt.title('Class Distribution in Alzheimer Dataset')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=0)
for i, (key, value) in enumerate(class_counts.items()):
    plt.text(i, value + 100, str(value), ha='center')
plt.tight_layout()
plt.show()

# Visualize sample images from each class
def show_samples(dataset, num_samples=3):
    fig, axes = plt.subplots(len(dataset.classes), num_samples, figsize=(15, 12))
    
    for i, class_name in enumerate(dataset.classes):
        # Get indices of samples from this class
        indices = [idx for idx, (_, label) in enumerate(dataset.samples) if label == dataset.class_to_idx[class_name]]
        
        # Randomly select samples
        selected_indices = random.sample(indices, min(num_samples, len(indices)))
        
        for j, idx in enumerate(selected_indices):
            img, _ = dataset[idx]
            
            # Convert tensor to numpy for visualization
            img = img.numpy().transpose((1, 2, 0))
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            axes[i, j].imshow(img)
            axes[i, j].set_title(f"{class_name}")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

# Show sample images
show_samples(dataset)

# Function to load the pre-trained ViT model and modify it for our task
def get_vit_model(num_classes=4):
    # Load pre-trained ViT-B/16 model
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    
    # Modify the classifier head for our number of classes
    num_features = model.heads.head.in_features
    model.heads.head = nn.Linear(num_features, num_classes)
    
    return model

# Progressive unfreezing training function
def train_with_progressive_unfreezing(model, criterion, num_epochs=10):
    """
    Train the model using a progressive unfreezing strategy:
    1. First train only the classification head with the backbone frozen
    2. Then unfreeze the last few transformer layers and continue training
    3. Finally unfreeze the entire model and fine-tune with a lower learning rate
    
    Args:
        model: The ViT model to train
        criterion: Loss function
        num_epochs: Total number of epochs to train (default: 10)
        
    Returns:
        model: Trained model
        history: Training history
    """
    since = time.time()
    
    # Define the three phases of training
    phases = [
        {"name": "Phase 1: Train classification head only", "epochs": 3},
        {"name": "Phase 2: Fine-tune last few transformer layers", "epochs": 3},
        {"name": "Phase 3: Full fine-tuning", "epochs": 4}
    ]
    
    # Ensure total epochs match the requested number
    total_epochs = sum(phase["epochs"] for phase in phases)
    assert total_epochs == num_epochs, f"Total epochs ({total_epochs}) doesn't match requested epochs ({num_epochs})"
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    current_epoch = 0
    
    # Phase 1: Train only the classification head
    print(phases[0]["name"])
    print("-" * 30)
    
    # Freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only the classification head
    for param in model.heads.parameters():
        param.requires_grad = True
    
    # Set up optimizer for phase 1 (only train unfrozen parameters)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=0.001, weight_decay=0.01)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=phases[0]["epochs"])
    
    # Train for phase 1 epochs
    for epoch in range(phases[0]["epochs"]):
        current_epoch += 1
        print(f'Epoch {current_epoch}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.float() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            # Deep copy the model if it's the best validation accuracy so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    # Phase 2: Unfreeze the last few transformer layers
    print(phases[1]["name"])
    print("-" * 30)
    
    # Get the total number of encoder layers
    num_layers = len(model.encoder.layers)
    
    # Unfreeze the last 4 transformer blocks (or 1/3 of the layers if less than 12)
    layers_to_unfreeze = max(1, num_layers // 3)
    
    # Keep track of which layers are unfrozen for debugging
    unfrozen_layers = []
    
    # Iterate through named modules to find encoder layers
    for name, module in model.named_modules():
        # Check if this is an encoder layer
        if 'encoder.layers' in name:
            # Extract the layer number
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i+1 < len(parts):
                    try:
                        layer_idx = int(parts[i+1])
                        # If this is one of the last layers we want to unfreeze
                        if layer_idx >= (num_layers - layers_to_unfreeze):
                            # Unfreeze all parameters in this layer
                            for param_name, param in module.named_parameters():
                                if '.' not in param_name:  # Only direct parameters of this module
                                    param.requires_grad = True
                                    unfrozen_layers.append(f"{name}.{param_name}")
                    except (ValueError, IndexError):
                        continue
    
    print(f"Unfrozen {len(unfrozen_layers)} parameters in the last {layers_to_unfreeze} encoder layers")
    
    # Set up optimizer for phase 2 with a slightly lower learning rate
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=0.0001, weight_decay=0.01)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=phases[1]["epochs"])
    
    # Train for phase 2 epochs
    for epoch in range(phases[1]["epochs"]):
        current_epoch += 1
        print(f'Epoch {current_epoch}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.float() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            # Deep copy the model if it's the best validation accuracy so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    # Phase 3: Full fine-tuning
    print(phases[2]["name"])
    print("-" * 30)
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Set up optimizer for phase 3 with an even lower learning rate
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=phases[2]["epochs"])
    
    # Train for phase 3 epochs
    for epoch in range(phases[2]["epochs"]):
        current_epoch += 1
        print(f'Epoch {current_epoch}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.float() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            # Deep copy the model if it's the best validation accuracy so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, history

# Initialize the model
model = get_vit_model(num_classes=len(dataset.classes))
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)

# Learning rate scheduler
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloader.dataset)
            # Use float32 instead of float64 (double) for MPS compatibility
            epoch_acc = running_corrects.float() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            # Deep copy the model if it's the best validation accuracy so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, history

# Function to evaluate the model and generate confusion matrix
def evaluate_model(model, dataloader):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=dataset.classes, 
                yticklabels=dataset.classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))

# Save the model
def save_model(model, filename='vit_alzheimer_model.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': dataset.classes,
        'class_to_idx': dataset.class_to_idx
    }, filename)
    print(f'Model saved to {filename}')

# Load the model
def load_model(filename='vit_alzheimer_model.pth'):
    # Check if the file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Model file {filename} not found")
    
    # Load the checkpoint
    checkpoint = torch.load(filename, map_location=device)
    
    # Get the classes and class_to_idx
    classes = checkpoint['classes']
    class_to_idx = checkpoint['class_to_idx']
    
    # Initialize a new model with the correct number of classes
    model = get_vit_model(num_classes=len(classes))
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to the appropriate device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    print(f"Model loaded from {filename}")
    print(f"Classes: {classes}")
    
    return model, classes, class_to_idx

# Function to predict class for a single image
def predict_image(image_path, model, classes, transform=None):
    """
    Predict the class of a single image using the loaded model
    
    Args:
        image_path (str): Path to the image file
        model (nn.Module): Loaded model
        classes (list): List of class names
        transform (callable, optional): Transform to apply to the image
        
    Returns:
        tuple: (predicted_class, probabilities)
    """
    if transform is None:
        # Use the same transform as validation if none provided
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = classes[predicted_idx.item()]
    
    # Print prediction results
    print(f"Predicted class: {predicted_class}")
    print("Class probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  {classes[i]}: {prob.item():.4f}")
    
    return predicted_class, probabilities.cpu().numpy()

def visualize_predictions(model, dataset, num_samples=4):
    # Get random samples
    indices = random.sample(range(len(dataset)), num_samples)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(16, 4))
    
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, true_label = dataset[idx]
            
            # Get prediction
            input_tensor = img.unsqueeze(0).to(device)
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            
            # Convert tensor to numpy for visualization
            img = img.numpy().transpose((1, 2, 0))
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            # Display image and prediction
            axes[i].imshow(img)
            true_class = dataset.classes[true_label]
            pred_class = dataset.classes[pred.item()]
            color = 'green' if true_label == pred.item() else 'red'
            axes[i].set_title(f"True: {true_class}\nPred: {pred_class}", color=color)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # Uncomment to train the model
    # model, history = train_model(model, criterion, optimizer, scheduler, num_epochs=10)
    # evaluate_model(model, val_loader)
    # save_model(model)
    
    # For now, just visualize the data
    print(f"Using device: {device}")
    print("Data visualization complete. Uncomment the training code to train the model.")
    
    # Example of loading the model and making predictions
    # Uncomment the following lines to load a trained model and make predictions
    """
    # Load the trained model
    loaded_model, classes, class_to_idx = load_model('vit_alzheimer_model.pth')
    
    # Example: Make prediction on a sample image
    # Replace 'path_to_test_image.jpg' with the path to your test image
    if os.path.exists('path_to_test_image.jpg'):
        predict_image('path_to_test_image.jpg', loaded_model, classes)
    
    # Example: Evaluate the loaded model on the validation set
    evaluate_model(loaded_model, val_loader)
    """ 