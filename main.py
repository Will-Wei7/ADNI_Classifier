import os
import argparse
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import random
import wandb
import sys

from mri_dataset import get_data_loaders
from models import TransferLearningModel
from trainer import ADNITrainer

def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Train 3D MRI Transfer Learning Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./ADNI_split',
                       help='Path to dataset directory')
    parser.add_argument('--target_shape', type=int, nargs=3, default=[128, 128, 128],
                       help='Target shape for MRI volumes (depth, height, width)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of workers for data loading (use 0 for Windows)')
    parser.add_argument('--flatten_dir', action='store_true',
                       help='Assume the data directory has a flattened structure without class subdirectories')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='resnet',
                       choices=['resnet', 'densenet', 'medicalnet', 'senet', 'vit'],
                       help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained model weights if available')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Initially freeze backbone weights for transfer learning')
    
    # Training arguments
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    parser.add_argument('--unfreeze_strategy', action='store_true',
                       help='Use gradual unfreezing strategy')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--project_name', type=str, default='ADNI-Transfer-Learning',
                       help='Project name for Weights & Biases')
    
    # Miscellaneous arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate the model, do not train')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to model checkpoint for evaluation')
    
    return parser.parse_args()

def main():
    """Main function"""
    try:
        # Parse arguments
        args = parse_args()
        
        # Set seed for reproducibility
        set_seed(args.seed)
        
        # Convert data_dir to absolute path
        args.data_dir = os.path.abspath(args.data_dir)
        
        # 检查数据目录是否存在
        if not os.path.exists(args.data_dir):
            print(f"ERROR: Data directory {args.data_dir} does not exist!")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Available directories: {[d for d in os.listdir('.') if os.path.isdir(d)]}")
            return
        
        # Windows-specific settings
        if sys.platform.startswith('win'):
            # Set num_workers to 0 if running on Windows
            args.num_workers = 0
            print("Running on Windows - setting num_workers to 0")
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Set up logging with Weights & Biases
        if args.use_wandb:
            wandb.init(project=args.project_name)
            wandb.config.update(vars(args))
        
        # Get data loaders
        print("Loading data...")
        try:
            data_loaders = get_data_loaders(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                target_shape=tuple(args.target_shape),
                num_workers=args.num_workers,
                flatten_dir=args.flatten_dir
            )
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            
            # 尝试使用最简单的方式，不假设任何数据结构
            print("\nAttempting to load data with minimal assumptions...")
            from mri_dataset import ADNIDataset
            from torch.utils.data import DataLoader, random_split
            
            # 创建一个自定义数据集用于测试
            test_dataset = ADNIDataset(
                args.data_dir,
                transform=None,
                target_shape=tuple(args.target_shape),
                flatten_dir=True
            )
            
            if len(test_dataset) == 0:
                print(f"ERROR: Could not find any valid samples in {args.data_dir}.")
                print("Please check your data directory structure.")
                print("Expected structure:")
                print("  1) Class-based: data_dir/[AD|CN|MCI]/patient_id/*.nii[.gz]")
                print("  2) Flattened: data_dir/patient_id/patient_id_preprocessed.nii[.gz]")
                return
            
            # 使用找到的数据集
            train_size = int(0.8 * len(test_dataset))
            val_size = int(0.1 * len(test_dataset))
            test_size = len(test_dataset) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                test_dataset, [train_size, val_size, test_size], 
                generator=torch.Generator().manual_seed(args.seed)
            )
            
            print(f"Created datasets with: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples")
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            
            data_loaders = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            }
            
        # Create model
        print(f"Creating {args.model_name} model...")
        model = TransferLearningModel(
            model_name=args.model_name,
            num_classes=3,  # AD, CN, MCI
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone
        )
        
        # Count model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Check if GPU is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available, using CPU for training.")
        else:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Set up trainer
        trainer = ADNITrainer(
            model=model,
            train_loader=data_loaders['train'],
            val_loader=data_loaders['val'],
            test_loader=data_loaders['test'],
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            use_wandb=args.use_wandb,
            project_name=args.project_name
        )
        
        # Train or evaluate
        if args.eval_only:
            if args.checkpoint_path:
                print(f"Loading checkpoint from {args.checkpoint_path}...")
                trainer.load_checkpoint(args.checkpoint_path)
            
            print("Evaluating model...")
            results = trainer.evaluate()
            print(f"Test accuracy: {results['accuracy']:.2f}%")
            print(f"Balanced accuracy: {results['balanced_accuracy']:.4f}")
        else:
            print("Training model...")
            trainer.train(
                num_epochs=args.num_epochs,
                patience=args.patience,
                unfreeze_strategy=args.unfreeze_strategy
            )
            
            print("Training complete! Evaluating on test set...")
            results = trainer.evaluate()
            print(f"Test accuracy: {results['accuracy']:.2f}%")
            print(f"Balanced accuracy: {results['balanced_accuracy']:.4f}")
        
        # Close wandb run
        if args.use_wandb:
            wandb.finish()
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        
        # Close wandb run on error
        if 'args' in locals() and hasattr(args, 'use_wandb') and args.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main() 