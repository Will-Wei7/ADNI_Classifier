import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
import argparse
from tqdm import tqdm

from mri_dataset import ADNIDataset, get_transforms
from models import TransferLearningModel


def visualize_sample(img_data, prediction=None, true_label=None, save_path=None):
    """
    Visualize a 3D MRI volume with predictions
    
    Args:
        img_data: 3D numpy array of MRI data
        prediction: Predicted class (int or string)
        true_label: True class (int or string)
        save_path: Path to save the visualization
    """
    # If input has channel dimension, remove it
    if len(img_data.shape) == 4:
        img_data = img_data[0]
    
    # Get middle slices
    d, h, w = img_data.shape
    mid_z = d // 2
    mid_y = h // 2
    mid_x = w // 2
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot axial view (top-down)
    axes[0].imshow(img_data[mid_z, :, :], cmap='gray')
    axes[0].set_title('Axial (z)')
    axes[0].axis('off')
    
    # Plot coronal view (front-back)
    axes[1].imshow(img_data[:, mid_y, :], cmap='gray')
    axes[1].set_title('Coronal (y)')
    axes[1].axis('off')
    
    # Plot sagittal view (left-right)
    axes[2].imshow(img_data[:, :, mid_x], cmap='gray')
    axes[2].set_title('Sagittal (x)')
    axes[2].axis('off')
    
    # Add prediction and true label as title
    class_map = {0: 'AD', 1: 'CN', 2: 'MCI'}
    pred_label = class_map[prediction] if isinstance(prediction, int) else prediction
    true_class = class_map[true_label] if isinstance(true_label, int) else true_label
    
    if prediction is not None and true_label is not None:
        correct = prediction == true_label
        color = 'green' if correct else 'red'
        fig.suptitle(f'True: {true_class}, Pred: {pred_label}', 
                    fontsize=16, color=color)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def predict_and_visualize(model, dataset, output_dir, device, num_samples=10):
    """
    Make predictions and visualize results for a dataset
    
    Args:
        model: Trained model
        dataset: Dataset with samples to visualize
        output_dir: Directory to save visualizations
        device: Device for inference
        num_samples: Number of samples to visualize (per class if per_class=True)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    # Create subdirectories for each class
    class_map = {0: 'AD', 1: 'CN', 2: 'MCI'}
    class_correct_dirs = {}
    class_incorrect_dirs = {}
    
    for class_idx, class_name in class_map.items():
        class_correct_dirs[class_idx] = os.path.join(output_dir, f"{class_name}_correct")
        class_incorrect_dirs[class_idx] = os.path.join(output_dir, f"{class_name}_incorrect")
        os.makedirs(class_correct_dirs[class_idx], exist_ok=True)
        os.makedirs(class_incorrect_dirs[class_idx], exist_ok=True)
    
    # Count samples per class
    samples_per_class = {0: 0, 1: 0, 2: 0}
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Predicting and visualizing"):
            sample = dataset[idx]
            img = sample['image']
            true_label = sample['label']
            patient_id = sample['patient_id']
            
            # Skip if already have enough samples for this class
            if samples_per_class[true_label] >= num_samples:
                continue
            
            # Make prediction
            inputs = img.unsqueeze(0).to(device)  # Add batch dimension
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            prediction = predicted.item()
            
            # Get image data for visualization
            img_data = img.cpu().numpy()
            
            # Determine if correct prediction
            is_correct = prediction == true_label
            
            # Choose directory based on correctness
            if is_correct:
                save_dir = class_correct_dirs[true_label]
            else:
                save_dir = class_incorrect_dirs[true_label]
            
            # Save visualization
            save_path = os.path.join(save_dir, f"{patient_id}.png")
            visualize_sample(
                img_data, 
                prediction=prediction, 
                true_label=true_label, 
                save_path=save_path
            )
            
            # Update counter
            samples_per_class[true_label] += 1
            
            # Check if we have enough samples for all classes
            if all(count >= num_samples for count in samples_per_class.values()):
                break
    
    print(f"Visualizations saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize MRI predictions')
    parser.add_argument('--data_dir', type=str, default='./ADNI_split/test',
                       help='Path to test data directory')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default='resnet',
                       choices=['resnet', 'densenet', 'medicalnet', 'senet', 'vit'],
                       help='Model architecture used')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                       help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize per class')
    parser.add_argument('--target_shape', type=int, nargs=3, default=[128, 128, 128],
                       help='Target shape for MRI volumes')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    transform = get_transforms(mode='test', target_shape=tuple(args.target_shape))
    dataset = ADNIDataset(args.data_dir, transform=transform, target_shape=tuple(args.target_shape))
    
    # Create model
    model = TransferLearningModel(
        model_name=args.model_name,
        num_classes=3,
        pretrained=False,
        freeze_backbone=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Loaded checkpoint from {args.checkpoint_path}")
    
    # Predict and visualize
    predict_and_visualize(
        model=model,
        dataset=dataset,
        output_dir=args.output_dir,
        device=device,
        num_samples=args.num_samples
    )

if __name__ == "__main__":
    main() 