import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from adni_dataset import get_dataloaders

# Configuration
DATA_DIR = "ADNI_Dataset"
BATCH_SIZE = 32
NUM_WORKERS = 4

# Class mapping
ID_TO_LABEL = {
    0: 'AD',     # Alzheimer's Disease
    1: 'CN',     # Cognitively Normal
    2: 'EMCI',   # Early Mild Cognitive Impairment
    3: 'LMCI'    # Late Mild Cognitive Impairment
}

def count_samples_per_class(data_dir):
    """
    Count the number of samples per class in each split.
    
    Args:
        data_dir (str): Path to the ADNI dataset
        
    Returns:
        pd.DataFrame: DataFrame with counts per class per split
    """
    counts = {}
    
    for split in ['train', 'valid', 'test']:
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            continue
            
        counts[split] = {}
        
        for class_name in ID_TO_LABEL.values():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                counts[split][class_name] = 0
                continue
                
            # Count files in directory
            counts[split][class_name] = len(list(class_dir.glob('*.jpg')))
    
    # Convert to DataFrame
    counts_df = pd.DataFrame(counts)
    return counts_df

def plot_class_distribution(counts_df, save_path=None):
    """
    Plot the class distribution across splits.
    
    Args:
        counts_df (pd.DataFrame): DataFrame with counts per class per split
        save_path (str, optional): Path to save the plot
    """
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Plot the counts
    counts_df.plot(kind='bar', ax=plt.gca())
    
    # Add labels and title
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('ADNI Dataset Class Distribution')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add total count as text on top of each bar
    for i, split in enumerate(counts_df.columns):
        for j, count in enumerate(counts_df[split]):
            plt.text(j + (i-1)/3, count + 50, str(count), 
                     ha='center', va='bottom', fontsize=10)
    
    # Add legend
    plt.legend(title='Split')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        print(f"Class distribution saved to {save_path}")
    
    plt.show()

def plot_class_imbalance_heatmap(counts_df, save_path=None):
    """
    Plot a heatmap of the class distribution to highlight imbalance.
    
    Args:
        counts_df (pd.DataFrame): DataFrame with counts per class per split
        save_path (str, optional): Path to save the plot
    """
    # Set up the figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(counts_df, annot=True, fmt='d', cmap='YlGnBu')
    
    # Add title
    plt.title('ADNI Dataset Class Distribution Heatmap')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        print(f"Class imbalance heatmap saved to {save_path}")
    
    plt.show()

def calculate_class_weights(counts_df, split='train'):
    """
    Calculate class weights inversely proportional to class frequency.
    
    Args:
        counts_df (pd.DataFrame): DataFrame with counts per class per split
        split (str): Split to calculate weights for (default: 'train')
        
    Returns:
        dict: Class weights
    """
    if split not in counts_df.columns:
        raise ValueError(f"Split '{split}' not found in counts DataFrame")
    
    # Get class counts for the specified split
    class_counts = counts_df[split]
    
    # Calculate weights: n_samples / (n_classes * class_count)
    n_samples = class_counts.sum()
    n_classes = len(class_counts)
    class_weights = n_samples / (n_classes * class_counts)
    
    # Create dictionary mapping class names to weights
    weights_dict = {class_name: weight for class_name, weight in zip(class_counts.index, class_weights)}
    
    # Print the weights
    print(f"Class weights for {split} split:")
    for class_name, weight in weights_dict.items():
        print(f"  {class_name}: {weight:.4f}")
    
    return weights_dict

if __name__ == "__main__":
    # Count samples per class
    counts_df = count_samples_per_class(DATA_DIR)
    print("\nSample counts per class:")
    print(counts_df)
    
    # Plot class distribution
    plot_class_distribution(counts_df, save_path="class_distribution.png")
    
    # Plot class imbalance heatmap
    plot_class_imbalance_heatmap(counts_df, save_path="class_imbalance_heatmap.png")
    
    # Calculate class weights for training
    weights = calculate_class_weights(counts_df)
    
    # Print total counts
    print("\nTotal samples per split:")
    print(counts_df.sum())
    
    # Print class distribution percentages
    print("\nClass distribution percentages:")
    print((counts_df / counts_df.sum() * 100).round(2)) 