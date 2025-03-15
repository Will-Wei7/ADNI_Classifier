import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from torchvision.models import vit_b_16
import torch.nn as nn

# Parse command line arguments
parser = argparse.ArgumentParser(description='Predict Alzheimer\'s class from an MRI image')
parser.add_argument('image_path', type=str, nargs='?', default='31 (2).jpg', help='Path to the MRI image')
args = parser.parse_args()

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT requires 224x224 input
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to load the model
def load_model(filename='vit_alzheimer_model.pth'):
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
    
    return model, checkpoint['classes'], checkpoint['class_to_idx']

# Function to predict the class of an image
def predict_image(image_path, model, classes):
    # Check if the image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found")
    
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
        
    # Display the image and prediction
    plt.figure(figsize=(10, 6))
    
    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Prediction: {pred_class}")
    plt.axis('off')
    
    # Display the probabilities
    plt.subplot(1, 2, 2)
    bars = plt.bar(classes, probabilities.cpu().numpy())
    plt.title('Class Probabilities')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    
    # Highlight the predicted class
    bars[preds.item()].set_color('red')
    
    plt.tight_layout()
    plt.show()
    
    # Print the results
    print(f"\nPrediction: {pred_class}")
    print("\nClass Probabilities:")
    for i, cls in enumerate(classes):
        print(f"{cls}: {probabilities[i].item():.4f}")
    
    return pred_class, probabilities

# Main execution
if __name__ == "__main__":
    try:
        # Load the model
        model, classes, class_to_idx = load_model()
        print(f"Model loaded successfully. Classes: {classes}")
        
        # Make prediction
        pred_class, _ = predict_image(args.image_path, model, classes)
        
    except Exception as e:
        print(f"Error: {e}") 