import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ADNIDataset(Dataset):
    """ADNI MRI Dataset for Alzheimer's Disease classification."""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of the ADNI dataset.
            split (str): Dataset split ('train', 'valid', 'test').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Class mapping
        self.class_to_idx = {
            'AD': 0,     # Alzheimer's Disease
            'CN': 1,     # Cognitively Normal
            'EMCI': 2,   # Early Mild Cognitive Impairment
            'LMCI': 3    # Late Mild Cognitive Impairment
        }
        
        # Load all data paths and labels
        self.data = []
        split_dir = self.root_dir / split
        
        for class_name in self.class_to_idx.keys():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
                
            class_idx = self.class_to_idx[class_name]
            for img_path in class_dir.glob('*.jpg'):
                self.data.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(split):
    """
    Get appropriate transforms for each split.
    
    Args:
        split (str): Dataset split ('train', 'valid', 'test')
        
    Returns:
        transforms.Compose: Composition of transforms
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Create dataloaders for training, validation and testing.
    
    Args:
        data_dir (str): Path to the ADNI dataset
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders
        
    Returns:
        dict: Dictionary containing dataloaders for each split
    """
    dataloaders = {}
    
    for split in ['train', 'valid', 'test']:
        dataset = ADNIDataset(
            root_dir=data_dir,
            split=split,
            transform=get_transforms(split)
        )
        
        shuffle = (split == 'train')
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders


if __name__ == "__main__":
    # Test the dataset
    data_dir = "ADNI_Dataset"
    
    # Create dataset and dataloader
    train_dataset = ADNIDataset(
        root_dir=data_dir,
        split='train',
        transform=get_transforms('train')
    )
    
    # Print dataset information
    print(f"Dataset size: {len(train_dataset)}")
    
    # Check a sample
    image, label = train_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}") 