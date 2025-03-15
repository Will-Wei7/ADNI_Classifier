import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import Counter
from torchvision.models import vit_b_16
import torch.nn as nn
import torchvision.transforms as transforms

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT requires 224x224 input
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to load the model
def load_model(filename='vit_alzheimer_model.pth'):
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if the file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file {filename} not found")
    
    # Load the model state dict
    checkpoint = torch.load(filename, map_location=device)
    
    # Get model architecture
    num_classes = len(checkpoint['classes'])
    model = vit_b_16()
    
    # Modify the classifier head for our number of classes
    num_features = model.heads.head.in_features
    model.heads.head = nn.Linear(num_features, num_classes)
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint['classes'], checkpoint['class_to_idx'], device

# Function to predict a single image
def predict_single_image(image_path, model, classes, device):
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Warning: Image file {image_path} not found, skipping")
        return None, None
    
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, preds = torch.max(outputs, 1)
            pred_class = classes[preds.item()]
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        return pred_class, probabilities.cpu().numpy()
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

# Function to process a directory of images
def process_directory(directory, model, classes, device, expected_class=None, output_csv=None):
    # Check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} not found")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No image files found in {directory}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    
    for image_path in tqdm(image_files, desc="Processing images"):
        pred_class, probabilities = predict_single_image(image_path, model, classes, device)
        
        if pred_class is not None:
            result = {
                'image_path': image_path,
                'predicted_class': pred_class,
                'expected_class': expected_class
            }
            
            # Add probabilities for each class
            for i, cls in enumerate(classes):
                result[f'prob_{cls}'] = probabilities[i]
            
            results.append(result)
    
    # Create a DataFrame from the results
    if results:
        df = pd.DataFrame(results)
        
        # Calculate accuracy if expected class is provided
        if expected_class is not None:
            accuracy = (df['predicted_class'] == expected_class).mean() * 100
            print(f"\nAccuracy for class '{expected_class}': {accuracy:.2f}%")
        
        # Count predictions by class
        class_counts = Counter(df['predicted_class'])
        print("\nPrediction distribution:")
        for cls, count in class_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {cls}: {count} ({percentage:.2f}%)")
        
        # Save to CSV if specified
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\nResults saved to {output_csv}")
        
        return df
    
    return None

# Function to visualize results
def visualize_results(df, classes, output_dir=None):
    if df is None or df.empty:
        print("No data to visualize")
        return
    
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot prediction distribution
    plt.figure(figsize=(10, 6))
    class_counts = Counter(df['predicted_class'])
    classes_found = list(class_counts.keys())
    counts = [class_counts[cls] for cls in classes_found]
    
    bars = plt.bar(classes_found, counts)
    plt.title('Prediction Distribution')
    plt.xlabel('Predicted Class')
    plt.ylabel('Count')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'))
    else:
        plt.show()
    
    # Plot average probability for each class
    plt.figure(figsize=(10, 6))
    avg_probs = [df[f'prob_{cls}'].mean() for cls in classes]
    
    bars = plt.bar(classes, avg_probs)
    plt.title('Average Probability by Class')
    plt.xlabel('Class')
    plt.ylabel('Average Probability')
    
    # Add probability labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'average_probabilities.png'))
    else:
        plt.show()
    
    # If expected class is provided, plot confusion matrix
    if 'expected_class' in df.columns and df['expected_class'].notna().any():
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        
        # Get unique classes
        unique_classes = sorted(set(classes) | set(df['expected_class'].unique()))
        
        # Create confusion matrix
        cm = confusion_matrix(
            df['expected_class'], 
            df['predicted_class'],
            labels=unique_classes
        )
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Batch predict Alzheimer\'s class for all images in a directory')
    parser.add_argument('directory', help='Path to the directory containing images')
    parser.add_argument('--model', default='vit_alzheimer_model.pth', help='Path to the model file')
    parser.add_argument('--expected-class', help='Expected class for all images in the directory')
    parser.add_argument('--output-csv', help='Path to save the results CSV file')
    parser.add_argument('--output-dir', help='Directory to save visualization plots')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization plots')
    
    args = parser.parse_args()
    
    try:
        # Load the model
        print("Loading model...")
        model, classes, class_to_idx, device = load_model(args.model)
        print(f"Model loaded successfully. Classes: {classes}")
        print(f"Using device: {device}")
        
        # Process the directory
        print(f"\nProcessing images in {args.directory}...")
        results_df = process_directory(
            args.directory,
            model,
            classes,
            device,
            args.expected_class,
            args.output_csv
        )
        
        # Visualize results if requested
        if args.visualize and results_df is not None:
            print("\nGenerating visualizations...")
            visualize_results(results_df, classes, args.output_dir)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 