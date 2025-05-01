# Alzheimer's Disease Classification using Vision Transformer (ViT)

This project implements a deep learning approach to classify Alzheimer's disease stages using brain MRI images from the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset. The model uses a Vision Transformer (ViT) architecture and includes an implementation of Dynamic Tanh activation function.

## Features

- Vision Transformer (ViT) based classification of Alzheimer's stages
- Support for Dynamic Tanh (DyT) activation function
- Comprehensive data visualization tools
- Class weighting and sampling strategies to handle imbalanced datasets
- Performance evaluation with detailed metrics
- Modular dataset handling for ADNI MRI images

## Dataset

The project uses the ADNI dataset with four classes:
- AD: Alzheimer's Disease
- CN: Cognitively Normal
- EMCI: Early Mild Cognitive Impairment
- LMCI: Late Mild Cognitive Impairment

## Project Structure

- `alzheimer_vit_train.py`: Main training script with ViT model implementation
- `adni_dataset.py`: Dataset handling for ADNI MRI images
- `create_adni_dataset.py`: Script to prepare the ADNI dataset
- `dynamic_tanh.py`: Implementation of Dynamic Tanh activation function
- `visualization_utils.py`: Utilities for visualizing results and training progress
- `visualize_class_distribution.py`: Script to visualize class distribution in dataset
- `visualize_dataset.py`: Script to visualize dataset samples

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Preparing the Dataset

```python
python create_adni_dataset.py
```

### Training the Model

```python
python alzheimer_vit_train.py
```

You can customize the training process with arguments:
- `--use_class_weights`: Use class weights to handle imbalanced data
- `--use_weighted_sampler`: Use weighted sampling during training
- `--epochs`: Number of training epochs

### Visualizing Results

```python
python visualize_class_distribution.py
python visualize_dataset.py
```

## Model Architecture

The project uses the ViT-B/16 model architecture with options to replace LayerNorm with Dynamic Tanh for improved performance.

## Results

The model achieves competitive performance in classifying Alzheimer's disease stages. Detailed metrics and visualizations are generated during training.

## References

- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [Dynamic Tanh](https://arxiv.org/abs/2503.10622)
- [ADNI Dataset](http://adni.loni.usc.edu/) 