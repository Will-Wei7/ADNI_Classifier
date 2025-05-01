import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adni_dataset import ADNIDataset, get_transforms
from torchvision import transforms

def denormalize(tensor):
    """Denormalize a tensor image with ImageNet mean and std."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def visualize_samples(dataset, num_samples=5, class_names=None):
    """Visualize random samples from the dataset."""
    if class_names is None:
        class_names = ['AD', 'CN', 'EMCI', 'LMCI']
    
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    
    plt.figure(figsize=(15, 3*num_samples))
    
    for i in range(num_samples):
        idx = np.random.randint(len(dataset))
        img, label = dataset[idx]
        
        # Denormalize
        img = denormalize(img)
        
        # Convert to numpy for plotting
        img = img.permute(1, 2, 0).numpy()
        # Clip values to be between 0 and 1
        img = np.clip(img, 0, 1)
        
        class_name = idx_to_class[label]
        
        plt.subplot(num_samples, 4, i*4 + 1)
        plt.imshow(img)
        plt.title(f"Class: {class_name}")
        plt.axis('off')
        
        # Plot image for each class
        for j, cn in enumerate(class_names):
            # Find an image from this class
            class_indices = [i for i, (_, lbl) in enumerate(dataset.data) if idx_to_class[lbl] == cn]
            if class_indices:
                class_idx = np.random.choice(class_indices)
                img, _ = dataset[class_idx]
                
                # Denormalize
                img = denormalize(img)
                
                # Convert to numpy for plotting
                img = img.permute(1, 2, 0).numpy()
                # Clip values to be between 0 and 1
                img = np.clip(img, 0, 1)
                
                plt.subplot(num_samples, 4, i*4 + j + 1)
                plt.imshow(img)
                plt.title(f"{cn}")
                plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png')
    plt.close()
    print(f"Sample visualization saved to 'dataset_samples.png'")

if __name__ == "__main__":
    # Check if dataset exists
    data_dir = Path("ADNI_Dataset")
    if not data_dir.exists():
        print("ADNI Dataset not found. Please run create_adni_dataset.py first.")
        exit(1)
    
    # Simple transforms for visualization (we'll still normalize but will denormalize for display)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create dataset
    train_dataset = ADNIDataset(
        root_dir=data_dir,
        split='train',
        transform=transform
    )
    
    # Visualize samples
    print(f"Visualizing samples from dataset with {len(train_dataset)} images...")
    visualize_samples(train_dataset, num_samples=3) 