# 3D MRI Transfer Learning Pipeline for Alzheimer's Classification

This project implements a robust 3D MRI transfer learning pipeline for classifying Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Cognitive Normal (CN) brain scans from the ADNI dataset.

## Features

- **Advanced Preprocessing**: Registration, intensity normalization, and spatial standardization
- **Transfer Learning**: Multiple model backbones with gradual unfreezing strategy
- **Aggressive Data Augmentation**: Spatial and intensity augmentations to maximize limited data usage
- **Flexible Training Pipeline**: Supports various architectures and hyperparameters

## Project Structure

```
.
├── main.py                  # Main script for training and evaluation
├── models.py                # Model architectures with transfer learning capabilities
├── mri_dataset.py           # Dataset and data loading utilities
├── trainer.py               # Training pipeline with gradual unfreezing
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

The code assumes the following structure for the ADNI dataset:
```
ADNI_split/
├── train/
│   ├── AD/         # Alzheimer's Disease
│   ├── CN/         # Cognitive Normal
│   └── MCI/        # Mild Cognitive Impairment
└── test/
    ├── AD/
    ├── CN/
    └── MCI/
```

Each patient directory contains MRI scans in NIfTI (.nii) format.

## Usage

### Training

```bash
# Basic usage with default parameters
python main.py

# Train with specific model and parameters
python main.py --model_name resnet --batch_size 8 --lr 1e-3 --num_epochs 50 --unfreeze_strategy

# Use pretrained weights and freeze backbone initially
python main.py --model_name densenet --pretrained --freeze_backbone --unfreeze_strategy

# Enable logging with Weights & Biases
python main.py --use_wandb --project_name "ADNI-Classification"
```

### Evaluation

```bash
# Evaluate trained model with specific checkpoint
python main.py --eval_only --checkpoint_path ./checkpoints/best_acc_epoch_30.pth
```

## Pipeline Details

### 1. Preprocessing

- **Registration**: Images are registered to a standard space
- **Normalization**: Z-score normalization of voxel intensities
- **Standardization**: Resampling to 1mm isotropic voxels and resizing to a common dimension

### 2. Transfer Learning Model Selection

Available backbones:
- ResNet50 (3D)
- DenseNet121 (3D)
- MedicalNet (custom 3D ResNet for medical images)
- SENet154 (3D)
- Vision Transformer (ViT)

### 3. Fine-Tuning Strategy

The pipeline employs a gradual unfreezing strategy:
1. Initially train only the classification head (backbone frozen)
2. Unfreeze top layers of the backbone with reduced learning rate
3. Optionally unfreeze more layers with further reduced learning rate

### 4. Data Augmentation

Aggressive 3D augmentations include:
- Spatial: rotations, translations, elastic deformations, scaling, flipping
- Intensity: brightness/contrast adjustments, noise, gamma transformations

## Performance Tracking

Training metrics are tracked:
- Loss and accuracy for training and validation sets
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping to prevent overfitting
- Detailed evaluation metrics including balanced accuracy and confusion matrix

## Citation

If you use this code in your research, please cite:

```
@misc{adni_transfer_learning,
  author = {Your Name},
  title = {3D MRI Transfer Learning Pipeline for Alzheimer's Classification},
  year = {2023},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yourusername/adni-transfer-learning}}
}
``` 