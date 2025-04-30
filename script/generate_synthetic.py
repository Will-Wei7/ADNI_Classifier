# @title generate_synthetic.py
# Script to generate synthetic images using a trained WGAN-GP Generator

import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# --- Configuration ---
# --- !! ADJUST THESE PARAMETERS !! ---
# Generator Model Parameters (MUST match the trained model)
LATENT_SIZE = 128
FEATURES_GEN = 64
CHANNELS_IMG = 1
INIT_H = 6
INIT_W = 5

# Paths and Generation Settings
MODEL_TYPE = "CN" # Specify 'AD' or 'CN' - determines which model to load and output folder name
MODEL_PATH = f"E:\EECS_PROJECT\gan_models_CN\generator_final_CN.pth" # Path to the trained generator .pth file
# MODEL_PATH = f"./gan_models/generator_epoch_600_{MODEL_TYPE}.pth" # Or specify an epoch

NUM_IMAGES_TO_GENERATE = 1000 # How many synthetic images to generate for this class
OUTPUT_DIR_SYNTHETIC = f"E:/EECS_PROJECT/ADNI_synthetic/train/{MODEL_TYPE}/" # Output directory for synthetic images

# Generation Batch Size (doesn't need to match training BATCH_SIZE)
GEN_BATCH_SIZE = 64
# --- End Configuration ---

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR_SYNTHETIC, exist_ok=True)

# --- Re-define Generator Class (Needs to match the definition in GAN.py) ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: LATENT_SIZE -> FEATURES_GEN*16 * INIT_H * INIT_W
            nn.Linear(LATENT_SIZE, FEATURES_GEN * 16 * INIT_H * INIT_W),
            nn.BatchNorm1d(FEATURES_GEN * 16 * INIT_H * INIT_W),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (FEATURES_GEN * 16, INIT_H, INIT_W)), # -> (bs, 1024, 6, 5)

            # Block 1: -> (bs, 512, 12, 10)
            nn.ConvTranspose2d(FEATURES_GEN * 16, FEATURES_GEN * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: -> (bs, 256, 24, 20)
            nn.ConvTranspose2d(FEATURES_GEN * 8, FEATURES_GEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: -> (bs, 128, 48, 40)
            nn.ConvTranspose2d(FEATURES_GEN * 4, FEATURES_GEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4: -> (bs, 64, 96, 80)
            nn.ConvTranspose2d(FEATURES_GEN * 2, FEATURES_GEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 5: -> (bs, 1, 192, 160)
            nn.ConvTranspose2d(FEATURES_GEN, CHANNELS_IMG, 4, 2, 1, bias=False),
            nn.Tanh() # Output [-1, 1]
        )

    def forward(self, input):
        return self.net(input)

# --- Main Generation Logic ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Generator
    gen = Generator().to(device)

    # Load trained state dictionary
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()

    try:
        gen.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded generator model from: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        exit()

    # Set generator to evaluation mode
    gen.eval()

    # Transformation to convert tensor back to PIL Image (0-255)
    # Inverse of the ToTensor() and Normalize((0.5,), (0.5,)) transforms
    transform_to_pil = transforms.Compose([
        transforms.Normalize((-1,), (2,)), # Equivalent to (img * 0.5) + 0.5 -> maps [-1, 1] to [0, 1]
        transforms.ToPILImage()
    ])

    generated_count = 0
    num_batches = (NUM_IMAGES_TO_GENERATE + GEN_BATCH_SIZE - 1) // GEN_BATCH_SIZE

    print(f"Generating {NUM_IMAGES_TO_GENERATE} synthetic images for class '{MODEL_TYPE}'...")

    with torch.no_grad(): # No need to track gradients during generation
        for i in tqdm(range(num_batches)):
            # Determine how many images to generate in this batch
            remaining = NUM_IMAGES_TO_GENERATE - generated_count
            current_batch_size = min(GEN_BATCH_SIZE, remaining)

            if current_batch_size <= 0:
                break

            # Generate noise
            noise = torch.randn(current_batch_size, LATENT_SIZE).to(device)

            # Generate images
            fake_images = gen(noise).detach().cpu() # Move to CPU for saving

            # Save images
            for j in range(fake_images.size(0)):
                img_tensor = fake_images[j]
                # Apply inverse transform to get PIL image in 0-255 range
                pil_img = transform_to_pil(img_tensor)

                # Save as grayscale PNG
                output_filename = f"synthetic_{MODEL_TYPE}_{generated_count + 1}.png"
                output_path = os.path.join(OUTPUT_DIR_SYNTHETIC, output_filename)
                try:
                    pil_img.convert('L').save(output_path)
                    generated_count += 1
                except Exception as e:
                    print(f"Error saving image {output_path}: {e}")

                if generated_count >= NUM_IMAGES_TO_GENERATE:
                    break

    print(f"\nFinished generating {generated_count} synthetic images.")
    print(f"Saved to: {OUTPUT_DIR_SYNTHETIC}")