"""
Visualization Utilities for Alzheimer's Disease Classification

This script contains visualization functions to analyze the dataset and model predictions.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
from sklearn.metrics import confusion_matrix
import torch
from PIL import Image

def visualize_dataset_distribution(train_dataset, val_dataset, test_dataset):
    """
    Visualize the distribution of classes in train, validation, and test datasets
    """
    datasets = {
        'Train': train_dataset,
        'Validation': val_dataset,
        'Test': test_dataset
    }
    
    class_counts = {}
    for name, dataset in datasets.items():
        class_counts[name] = {}
        for _, label in dataset.samples:
            class_name = dataset.classes[label]
            if class_name in class_counts[name]:
                class_counts[name][class_name] += 1
            else:
                class_counts[name][class_name] = 1
    
    # Plot class distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (name, counts) in enumerate(class_counts.items()):
        axes[i].bar(counts.keys(), counts.values())
        axes[i].set_title(f'{name} Set Class Distribution')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Number of Images')
        axes[i].tick_params(axis='x', rotation=0)
        
        # Add count labels
        for j, (key, value) in enumerate(counts.items()):
            axes[i].text(j, value + 5, str(value), ha='center')
    
    plt.tight_layout()
    plt.show()

def show_samples(dataset, num_samples=3):
    """
    Display sample images from each class in the dataset
    """
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

def visualize_predictions(model, dataset, device, num_samples=4):
    """
    Visualize model predictions on random samples
    """
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

def plot_training_history(history):
    """Plot training history with 4 subplots: Loss, Accuracy, Phase Duration, Total Epoch Duration"""
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot Loss
    epochs = range(len(history['train_loss']))
    axs[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axs[0, 0].plot(epochs, history['val_loss'], 'orange', label='Validation')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    
    # Plot Accuracy
    axs[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axs[0, 1].plot(epochs, history['val_acc'], 'orange', label='Validation')
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].set_ylim([0.5, 1.0])  # Set y-axis limits similar to example
    axs[0, 1].legend()
    
    # Plot Phase Duration
    axs[1, 0].plot(history['timing']['epochs'], history['timing']['train_time'], 'b-o', label='Train')
    axs[1, 0].plot(history['timing']['epochs'], history['timing']['val_time'], 'r-o', label='Validation')
    axs[1, 0].set_title('Phase Duration')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Time (seconds)')
    axs[1, 0].legend()
    
    # Plot Total Epoch Duration
    axs[1, 1].plot(history['timing']['epochs'], history['timing']['total_epoch_time'], 'g-o')
    axs[1, 1].set_title('Total Epoch Duration')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def plot_confusion_matrix(true_labels, predictions, class_names):
    """
    Generate and plot confusion matrix
    """
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show() 