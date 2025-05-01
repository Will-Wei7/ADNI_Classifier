"""
Alzheimer's Disease Classification using Vision Transformer (ViT)

This script trains a Vision Transformer model to classify Alzheimer's disease stages
using the ADNI dataset.
"""

# Import necessary libraries
import os
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from PIL import Image
import random
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from torchvision.models import vit_b_16, ViT_B_16_Weights
from dynamic_tanh import convert_ln_to_dyt
# Import visualization utilities
from visualization_utils import plot_training_history, plot_confusion_matrix, visualize_dataset_distribution, show_samples, visualize_predictions

# Timer class for detailed performance measurement
class Timer:
    def __init__(self, name=None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = None
        self.timing_data = {}

    def start(self):
        self.start_time = time.time()
        return self

    def stop(self):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        if self.name:
            print(f"Timer '{self.name}': {self.format_time(self.elapsed)}")
        return self.elapsed

    def record(self, label):
        """Record a split time with a specific label"""
        if self.start_time is None:
            raise ValueError("Timer must be started before recording")
        
        current = time.time()
        elapsed = current - self.start_time
        self.timing_data[label] = elapsed
        return elapsed

    def get_elapsed_time(self):
        """Get elapsed time in various formats"""
        if self.elapsed is None:
            if self.start_time is not None:
                return time.time() - self.start_time
            else:
                return 0
        return self.elapsed

    def summary(self):
        """Print a summary of all recorded times"""
        if not self.timing_data and self.elapsed is None:
            print("No timing data recorded")
            return
        
        print(f"\n===== Timer Summary: {self.name} =====")
        
        # Print individual records if any
        if self.timing_data:
            prev_time = 0
            for i, (label, total_time) in enumerate(sorted(self.timing_data.items(), key=lambda x: x[1])):
                if i == 0:
                    duration = total_time
                else:
                    duration = total_time - prev_time
                
                print(f"- {label}: {self.format_time(total_time)} (segment: {self.format_time(duration)})")
                prev_time = total_time
        
        # Print total time
        if self.elapsed is not None:
            print(f"Total time: {self.format_time(self.elapsed)}")
        elif self.start_time is not None:
            current_elapsed = time.time() - self.start_time
            print(f"Current elapsed time: {self.format_time(current_elapsed)} (still running)")
        print("=====================================")

    @staticmethod
    def format_time(seconds):
        """Format time in a human-readable format"""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = seconds // 60
            sec = seconds % 60
            return f"{int(minutes)}m {sec:.2f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            sec = seconds % 60
            return f"{int(hours)}h {int(minutes)}m {sec:.2f}s"

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
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Function to calculate class weights
def calculate_class_weights(dataset):
    """Calculate class weights based on sample distribution"""
    # Extract all labels
    labels = [label for _, label in dataset.samples]
    
    # Compute class weights using 'balanced' strategy
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    
    # Convert to tensor
    class_weights = torch.FloatTensor(class_weights)
    
    # Print class weights
    print("Class weights:")
    for cls_name, weight in zip(dataset.classes, class_weights):
        print(f"  {cls_name}: {weight:.4f}")
        
    return class_weights

# Function to create weighted sampler
def create_weighted_sampler(dataset):
    """Create a weighted sampler to balance classes during training"""
    # Extract all labels
    labels = [label for _, label in dataset.samples]
    
    # Count samples per class
    class_counts = np.bincount(labels)
    
    # Calculate weights for each sample
    weights = 1.0 / class_counts[labels]
    
    # Create and return sampler
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    
    return sampler

# Function to create ViT model
def get_vit_model(num_classes=4):
    """Create a Vision Transformer model"""
    # Load pre-trained ViT-B/16 model
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    # model = convert_ln_to_dyt(model)
    # Modify the classifier head for our number of classes
    num_features = model.heads.head.in_features
    model.heads.head = nn.Linear(num_features, num_classes)
    
    return model

# Function to visualize class weights
def visualize_class_weights(class_weights, class_names):
    """Visualize class weights"""
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_weights.numpy())
    plt.title('Class Weights for Handling Imbalance')
    plt.xlabel('Class')
    plt.ylabel('Weight')
    
    # Add weight values on top of the bars
    for i, weight in enumerate(class_weights):
        plt.text(i, weight.item() + 0.05, f"{weight.item():.4f}", ha='center')
    
    plt.tight_layout()
    plt.show()

# Evaluation function
def evaluate_model(model, dataloader):
    """Evaluate model performance on a dataset"""
    # Start evaluation timer
    eval_timer = Timer(name="Evaluation").start()
    
    model.eval()
    
    all_preds = []
    all_labels = []
    
    forward_times = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Time the forward pass
            start_time = time.time()
            outputs = model(inputs)
            elapsed = time.time() - start_time
            forward_times.append(elapsed)
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # Calculate precision, recall and F1 score
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    # Report timing information
    eval_time = eval_timer.stop()
    avg_batch_time = np.mean(forward_times)
    print(f'Evaluation completed in {Timer.format_time(eval_time)}')
    print(f'Average batch inference time: {avg_batch_time:.4f} seconds')
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, train_dataset.classes)
    
    return accuracy, precision, recall, f1, eval_time, all_labels, all_preds

# Function to save the model
def save_model(model, filename='vit_alzheimer_model.pth'):
    """Save the trained model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx
    }, filename)
    
    print(f'Model saved to {filename}')

# Main training function
def train_alzheimer_model(num_epochs=10, use_class_weights=True, use_weighted_sampler=False):
    """Main function to train and evaluate the model"""
    # Start overall timer
    total_timer = Timer(name="Total Training Pipeline").start()
    
    # Visualize dataset distribution
    print("Visualizing dataset distribution...")
    visualize_dataset_distribution(train_dataset, val_dataset, test_dataset)
    total_timer.record("Dataset visualization complete")
    
    # Calculate class weights if needed
    if use_class_weights or use_weighted_sampler:
        class_weights = calculate_class_weights(train_dataset)
        
        # Visualize class weights
        visualize_class_weights(class_weights, train_dataset.classes)
        print("Class weights calculated to handle imbalance")
    else:
        class_weights = None
        print("Not using class weights or weighted sampling")
    
    # Show sample images
    print("Showing sample images...")
    show_samples(train_dataset)
    total_timer.record("Sample visualization complete")
    
    # Initialize the model
    model_init_timer = Timer(name="Model Initialization").start()
    
    model = get_vit_model(num_classes=len(train_dataset.classes))
    
    model = model.to(device)
    model_init_time = model_init_timer.stop()
    print(f"Model initialization completed in {Timer.format_time(model_init_time)}")
    total_timer.record("Model initialization complete")
    
    # Define loss function (with class weights if enabled)
    if use_class_weights and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Using weighted CrossEntropyLoss with class weights")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")
    
    # Create a dictionary to store performance metrics
    performance_metrics = {
        'model_init_time': model_init_time,
        'training_time': None,
        'val_eval_time': None,
        'test_eval_time': None,
        'total_time': None,
        'val_accuracy': None,
        'val_precision': None,
        'val_recall': None,
        'val_f1': None,
        'test_accuracy': None,
        'test_precision': None,
        'test_recall': None,
        'test_f1': None,
        'class_weights_used': use_class_weights,
        'weighted_sampler_used': use_weighted_sampler
    }
    
    # Create data loaders (with weighted sampler if enabled)
    if use_weighted_sampler:
        sampler = create_weighted_sampler(train_dataset)
        train_loader_balanced = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=sampler,
            num_workers=4
        )
        print("Using weighted sampler for balanced batches")
    else:
        train_loader_balanced = train_loader
        print("Using standard random sampling")
    
    # Train the model
    training_timer = Timer(name="Model Training").start()
    
    print("Training model...")
    # Define optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Train the model
    model, history = train_model(
        model, 
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs,
        train_loader=train_loader_balanced  # Use balanced loader if weighted sampling is enabled
    )
    
    training_time = training_timer.stop()
    performance_metrics['training_time'] = training_time
    print(f"Model training completed in {Timer.format_time(training_time)}")
    total_timer.record("Model training complete")
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Evaluate model on validation set
    print("\nEvaluating model on validation set:")
    val_accuracy, val_precision, val_recall, val_f1, val_eval_time, val_labels, val_preds = evaluate_model(model, val_loader)
    performance_metrics['val_accuracy'] = val_accuracy
    performance_metrics['val_precision'] = val_precision
    performance_metrics['val_recall'] = val_recall
    performance_metrics['val_f1'] = val_f1
    performance_metrics['val_eval_time'] = val_eval_time
    total_timer.record("Validation evaluation complete")
    
    # Evaluate model on test set
    print("\nEvaluating model on test set:")
    test_accuracy, test_precision, test_recall, test_f1, test_eval_time, test_labels, test_preds = evaluate_model(model, test_loader)
    performance_metrics['test_accuracy'] = test_accuracy
    performance_metrics['test_precision'] = test_precision
    performance_metrics['test_recall'] = test_recall
    performance_metrics['test_f1'] = test_f1
    performance_metrics['test_eval_time'] = test_eval_time
    total_timer.record("Test evaluation complete")
    
    # Visualize predictions
    print("\nVisualizing model predictions...")
    visualize_predictions(model, test_dataset, device, num_samples=4)
    
    # Save the trained model
    save_timer = Timer(name="Model Saving").start()
    model_filename = f'vit_alzheimer_model_cw{use_class_weights}_ws{use_weighted_sampler}.pth'
    save_model(model, model_filename)
    save_time = save_timer.stop()
    total_timer.record("Model saving complete")
    
    # Record total time
    total_time = total_timer.stop()
    performance_metrics['total_time'] = total_time
    
    # Print summary of performance metrics
    print("\n===== Performance Summary =====")
    print(f"Model initialization time: {Timer.format_time(performance_metrics['model_init_time'])}")
    print(f"Training time: {Timer.format_time(performance_metrics['training_time'])}")
    print(f"Validation evaluation time: {Timer.format_time(performance_metrics['val_eval_time'])}")
    print(f"Test evaluation time: {Timer.format_time(performance_metrics['test_eval_time'])}")
    print(f"Total pipeline time: {Timer.format_time(performance_metrics['total_time'])}")
    print(f"Class weights used: {performance_metrics['class_weights_used']}")
    print(f"Weighted sampler used: {performance_metrics['weighted_sampler_used']}")
    print(f"Validation accuracy: {performance_metrics['val_accuracy']:.4f}")
    print(f"Validation precision: {performance_metrics['val_precision']:.4f}")
    print(f"Validation recall: {performance_metrics['val_recall']:.4f}")
    print(f"Validation F1 score: {performance_metrics['val_f1']:.4f}")
    print(f"Test accuracy: {performance_metrics['test_accuracy']:.4f}")
    print(f"Test precision: {performance_metrics['test_precision']:.4f}")
    print(f"Test recall: {performance_metrics['test_recall']:.4f}")
    print(f"Test F1 score: {performance_metrics['test_f1']:.4f}")
    print("===============================")
    
    total_timer.summary()
    
    return model, val_accuracy, test_accuracy, performance_metrics

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs, train_loader=None):
    """Train the model"""
    # Create and start main training timer
    timer = Timer(name="Training").start()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'timing': {
            'epochs': [],
            'train_time': [],
            'val_time': [],
            'total_epoch_time': []
        }
    }
    
    timer.record("Setup complete")
    
    # Use provided train loader or default global one
    if train_loader is None:
        train_loader = globals()['train_loader']
    
    for epoch in range(num_epochs):
        # Create epoch timer
        epoch_timer = Timer(name=f"Epoch {epoch+1}/{num_epochs}").start()
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            phase_timer = Timer(name=f"Phase: {phase}").start()
            
            if phase == 'train':
                model.train()
                dataloader = train_loader  # Use the provided train_loader (may be balanced)
            else:
                model.eval()
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
                running_loss += loss.detach().item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.float() / len(dataloader.dataset)
            
            # Record phase time
            phase_time = phase_timer.stop()
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {Timer.format_time(phase_time)}')
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                history['timing']['train_time'].append(phase_time)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                history['timing']['val_time'].append(phase_time)
            
            # Deep copy the model if it's the best validation accuracy so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        # Record total epoch time
        epoch_time = epoch_timer.stop()
        history['timing']['epochs'].append(epoch+1)
        history['timing']['total_epoch_time'].append(epoch_time)
        
        # Record checkpoint in main timer
        timer.record(f"Completed epoch {epoch+1}/{num_epochs}")
        print()
    
    # Stop main timer and print total time
    total_time = timer.stop()
    print(f'Training complete in {Timer.format_time(total_time)}')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Print detailed timer summary
    timer.summary()
    
    return model, history

# If this script is run directly
if __name__ == "__main__":
    # Set random seed for reproducibility
    seed = 545
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set up device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    print(f"PyTorch version: {torch.__version__}")
    # Load datasets
    train_dataset = AlzheimerDataset(root_dir='ADNI_Dataset/train', transform=data_transforms['train'])
    val_dataset = AlzheimerDataset(root_dir='ADNI_Dataset/valid', transform=data_transforms['val'])
    test_dataset = AlzheimerDataset(root_dir='ADNI_Dataset/test', transform=data_transforms['test'])

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Print dataset information
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Train the model with class weights to handle imbalance
    model, val_accuracy, test_accuracy, performance_metrics = train_alzheimer_model(
        num_epochs=5,
        use_class_weights=False,     # Enable class weights in loss function
        use_weighted_sampler=False  # Optionally enable weighted sampler
    )
    
    # Print final performance summary
    print("\nTraining completed successfully!")
    print(f"Final validation accuracy: {val_accuracy:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Final validation precision: {performance_metrics['val_precision']:.4f}")
    print(f"Final validation recall: {performance_metrics['val_recall']:.4f}")
    print(f"Final validation F1 score: {performance_metrics['val_f1']:.4f}")
    print(f"Final test precision: {performance_metrics['test_precision']:.4f}")
    print(f"Final test recall: {performance_metrics['test_recall']:.4f}")
    print(f"Final test F1 score: {performance_metrics['test_f1']:.4f}")
    print(f"Total time taken: {Timer.format_time(performance_metrics['total_time'])}")