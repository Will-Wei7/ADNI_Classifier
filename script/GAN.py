import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import torch.autograd as autograd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms # Import transforms
# --- Add this for Windows multiprocessing if needed ---
from multiprocessing import freeze_support
from tqdm import tqdm # Added for progress bar in main loop

# --- Parameters (Adjust PATHS and potentially AXIAL_AXIS) ---
LATENT_SIZE = 128
IMAGE_SIZE_H = 192
IMAGE_SIZE_W = 160
CHANNELS_IMG = 1
BATCH_SIZE = 32
NUM_EPOCHS = 600 # Or the desired number of epochs
LEARNING_RATE = 0.0001
BETA1 = 0.5
BETA2 = 0.9
CRITIC_ITERATIONS = 5 # Training ratio
LAMBDA_GP = 10 # Gradient penalty weight
FEATURES_GEN = 64 # Base feature size for generator
FEATURES_CRIT = 32 # Base feature size for critic
INIT_H = 6 # Initial height for generator based on architecture
INIT_W = 5 # Initial width for generator based on architecture

# --- Specify the path to the processed PNG data ---
# --- !! ADJUST THIS PATH !! (e.g., for AD or CN training data) ---
# Example Path: Change '/path/to/processed_data/' to your actual path
# Make sure this path is correct for your environment (e.g., Google Drive mount)
# DATA_DIR = '/content/drive/MyDrive/ADNI_dataset_processed_pytorch/train/AD/' # Example Colab path
DATA_DIR = r'E:\EECS_PROJECT\processed_dataset\train\CN' # Example Windows path - ADJUST

# --- Directory for saving results (optional) ---
OUTPUT_IMAGE_DIR = './gan_output_images_CN/'
MODEL_SAVE_DIR = './gan_models_CN/'
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# --- Define Custom Dataset ---
class ImageSliceDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # Find all PNG files recursively within the directory
        self.img_files = []
        if not os.path.isdir(img_dir):
             raise FileNotFoundError(f"Image directory not found: {img_dir}")
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    self.img_files.append(os.path.join(root, file))
        if not self.img_files:
             raise FileNotFoundError(f"No PNG images found in directory: {img_dir}")
        print(f"Found {len(self.img_files)} images in {img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        try:
            # Open image using PIL, ensure grayscale
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor or skip - returning None might cause DataLoader issues
            # For simplicity, let's return a zero tensor of the expected shape
            return torch.zeros((CHANNELS_IMG, IMAGE_SIZE_H, IMAGE_SIZE_W))


# --- Define Transformations ---
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE_H, IMAGE_SIZE_W), interpolation=transforms.InterpolationMode.LANCZOS), # H, W order
    transforms.ToTensor(), # Scales to [0, 1]
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
])


# --- Generator Class ---
# Simplified architecture based on typical DCGAN structure for WGAN-GP
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

# --- Critic Class ---
# Simplified architecture based on typical DCGAN structure for WGAN-GP
class Critic(nn.Module):
     def __init__(self):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            # Input: (1, 192, 160) -> (32, 96, 80)
            nn.Conv2d(CHANNELS_IMG, FEATURES_CRIT, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 1: -> (64, 48, 40)
            nn.Conv2d(FEATURES_CRIT, FEATURES_CRIT * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(FEATURES_CRIT * 2, affine=True), # Using InstanceNorm
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: -> (128, 24, 20)
            nn.Conv2d(FEATURES_CRIT * 2, FEATURES_CRIT * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(FEATURES_CRIT * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: -> (256, 12, 10)
            nn.Conv2d(FEATURES_CRIT * 4, FEATURES_CRIT * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(FEATURES_CRIT * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

             # Block 4: -> (512, 6, 5)
            nn.Conv2d(FEATURES_CRIT * 8, FEATURES_CRIT * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(FEATURES_CRIT * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Final Layer: -> (1, 1, 1)
            nn.Conv2d(FEATURES_CRIT * 16, 1, kernel_size=(INIT_H, INIT_W), stride=1, padding=0, bias=False),
        )

     def forward(self, input):
         return self.net(input).view(-1) # Reshape to (batch_size)


# --- Gradient Penalty Function ---
def gradient_penalty(critic, real_data, fake_data, device):
    batch_size, C, H, W = real_data.shape
    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(device)
    alpha = alpha.expand_as(real_data)

    min_batch_size = min(real_data.size(0), fake_data.size(0))
    interpolated_images = (alpha[:min_batch_size] * real_data[:min_batch_size] + \
                           (1 - alpha[:min_batch_size]) * fake_data[:min_batch_size])
    interpolated_images = interpolated_images.to(device)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images)

    gradient = autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradient = gradient.view(gradient.size(0), -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


# --- Weight Initialization Function ---
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1: # Handles BatchNorm1d and BatchNorm2d
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('InstanceNorm') != -1: # Handles InstanceNorm2d
         # InstanceNorm might not have bias, check weight presence
         if m.weight is not None:
             nn.init.normal_(m.weight.data, 1.0, 0.02)
         if m.bias is not None:
             nn.init.constant_(m.bias.data, 0)


# --- Main execution block ---
if __name__ == '__main__':
    freeze_support() # Needed for Windows multiprocessing

    # --- Create Dataset and DataLoader ---
    try:
        dataset = ImageSliceDataset(img_dir=DATA_DIR, transform=transform)
        # Set num_workers=0 if multiprocessing causes issues
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the DATA_DIR path is correct and the preprocessing script has been run.")
        exit()
    except Exception as e:
        print(f"Error creating Dataset/DataLoader: {e}")
        exit()


    # --- Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gen = Generator().to(device)
    crit = Critic().to(device)

    # Apply Weight Initialization
    gen.apply(weights_init)
    crit.apply(weights_init)


    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    opt_crit = optim.Adam(crit.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    # --- Fixed noise for visualization ---
    fixed_noise = torch.randn(64, LATENT_SIZE).to(device) # Generate 64 samples for an 8x8 grid

    # --- Training Loop ---
    crit_losses = []
    gen_losses = []

    print("Starting Training Loop...")
    try:
        for epoch in range(NUM_EPOCHS):
            loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
            epoch_crit_loss = 0.0
            epoch_gen_loss = 0.0
            num_batches = 0

            for batch_idx, real in loop:

                real = real.to(device)
                cur_batch_size = real.shape[0]

                if cur_batch_size == 0:
                    continue

                # --- Train Critic ---
                mean_iteration_critic_loss = 0
                crit.zero_grad() # Zero grad once before the critic loop
                for _ in range(CRITIC_ITERATIONS):
                    noise = torch.randn(cur_batch_size, LATENT_SIZE).to(device)
                    with torch.no_grad():
                        fake = gen(noise)
                    crit_real = crit(real)
                    crit_fake = crit(fake)
                    gp = gradient_penalty(crit, real, fake.detach(), device=device) # Detach fake here for GP
                    loss_crit = -(torch.mean(crit_real) - torch.mean(crit_fake)) + LAMBDA_GP * gp

                    # Accumulate gradients for critic iterations before stepping
                    # Divide loss by iterations for stable averaging before backward pass
                    (loss_crit / CRITIC_ITERATIONS).backward()
                    mean_iteration_critic_loss += loss_crit.item() / CRITIC_ITERATIONS
                
                opt_crit.step() # Step optimizer once after accumulating gradients


                # --- Train Generator ---
                gen.zero_grad() # Zero gen grad before generator step
                noise = torch.randn(cur_batch_size, LATENT_SIZE).to(device)
                fake = gen(noise)
                gen_fake = crit(fake)
                loss_gen = -torch.mean(gen_fake)

                loss_gen.backward()
                opt_gen.step()

                epoch_crit_loss += mean_iteration_critic_loss # Already averaged
                epoch_gen_loss += loss_gen.item()
                num_batches += 1

                # --- Update tqdm postfix ---
                # Corrected the f-string formatting for set_postfix
                loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
                loop.set_postfix(loss_C=f"{mean_iteration_critic_loss:.4f}", loss_G=f"{loss_gen.item():.4f}")


            # --- End of Epoch Logging & Visualization ---
            avg_epoch_crit_loss = epoch_crit_loss / num_batches if num_batches > 0 else 0
            avg_epoch_gen_loss = epoch_gen_loss / num_batches if num_batches > 0 else 0
            crit_losses.append(avg_epoch_crit_loss)
            gen_losses.append(avg_epoch_gen_loss)

            print(f"\n--- Epoch {epoch+1} Summary ---")
            print(f"Average Critic Loss: {avg_epoch_crit_loss:.4f}")
            print(f"Average Generator Loss: {avg_epoch_gen_loss:.4f}")

            # Save generated images from fixed noise at the end of each epoch
            with torch.no_grad():
                 gen.eval()
                 fake_samples = gen(fixed_noise)
                 gen.train()
                 fake_samples = (fake_samples + 1) / 2 # Rescale [-1, 1] to [0, 1]
                 img_grid = vutils.make_grid(fake_samples, padding=2, normalize=False)
                 vutils.save_image(img_grid, f"{OUTPUT_IMAGE_DIR}/epoch_{epoch+1}.png")


            # --- Save model checkpoints periodically ---
            if (epoch + 1) % 50 == 0:
                 torch.save(gen.state_dict(), f'{MODEL_SAVE_DIR}/generator_epoch_{epoch+1}.pth')
                 torch.save(crit.state_dict(), f'{MODEL_SAVE_DIR}/critic_epoch_{epoch+1}.pth')
                 print(f"*** Saved models at epoch {epoch+1} ***")


    except KeyboardInterrupt:
        print("\n--- Training interrupted by user ---")
    except Exception as e:
        print(f"\n--- An error occurred during training loop ---")
        if 'epoch' in locals(): print(f"Epoch: {epoch}")
        if 'batch_idx' in locals(): print(f"Batch Index: {batch_idx}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Saving final models (if training started)...")
        if 'gen' in locals() and epoch >=0: # Check if training actually started
             torch.save(gen.state_dict(), f'{MODEL_SAVE_DIR}/generator_final.pth')
        if 'crit' in locals() and epoch >=0:
             torch.save(crit.state_dict(), f'{MODEL_SAVE_DIR}/critic_final.pth')


    print("\nTraining Finished.")

    # --- Plot losses (optional) ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Critic Loss During Training (Epoch Averages)")
        plt.plot(gen_losses, label="G")
        plt.plot(crit_losses, label="C")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{OUTPUT_IMAGE_DIR}/loss_plot_epochs.png")
        # plt.show() # Comment out if running non-interactively
        plt.close() # Close plot
    except ImportError:
        print("Matplotlib not found. Skipping loss plot generation.")
    except Exception as e:
        print(f"Error generating loss plot: {e}")