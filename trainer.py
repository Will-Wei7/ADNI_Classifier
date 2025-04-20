import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import wandb

class ADNITrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer=None,
        criterion=None,
        lr=1e-3,
        weight_decay=1e-4,
        device=None,
        checkpoint_dir='./checkpoints',
        use_wandb=False,
        project_name='ADNI-Transfer-Learning'
    ):
        """
        Trainer for ADNI MRI classification
        
        Args:
            model: PyTorch model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            optimizer: PyTorch optimizer (default: AdamW)
            criterion: Loss function (default: CrossEntropyLoss)
            lr: Learning rate
            weight_decay: Weight decay for regularization
            device: Device to use (default: CUDA if available, else CPU)
            checkpoint_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases for logging
            project_name: Project name for Weights & Biases
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device if device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        self.project_name = project_name
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 计算类别权重以处理不平衡
        class_weights = self._compute_class_weights(train_loader)
        if class_weights is not None:
            print(f"Using weighted cross-entropy with weights: {class_weights}")
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            # 使用默认损失函数
            self.criterion = criterion if criterion else nn.CrossEntropyLoss()
            print("Using standard cross-entropy loss")
        
        # Set up optimizer
        self.optimizer = optimizer if optimizer else optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Set up learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Class names for reporting
        self.class_names = ['AD', 'CN', 'MCI']
        
        # Initialize Weights & Biases
        if self.use_wandb:
            wandb.init(project=self.project_name)
            wandb.config.update({
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "model": model.__class__.__name__,
                "optimizer": self.optimizer.__class__.__name__,
                "criterion": self.criterion.__class__.__name__,
                "device": self.device.type
            })
            wandb.watch(self.model)
    
    def _compute_class_weights(self, train_loader):
        """计算类别权重以处理类别不平衡"""
        try:
            # 统计每个类别的样本数
            class_counts = torch.zeros(3)  # 假设3个类别: AD, CN, MCI
            for batch in train_loader:
                if isinstance(batch, dict) and 'label' in batch:
                    labels = batch['label']
                    for label in labels:
                        class_counts[label] += 1
            
            # 确保没有类别样本数为0
            if torch.any(class_counts == 0):
                print("Warning: Some classes have zero samples!")
                return None
            
            # 计算权重：反比于样本数量
            weights = 1.0 / class_counts
            # 归一化权重，使其和为类别数
            weights = weights * (len(class_counts) / weights.sum())
            
            print(f"Class counts: {class_counts}")
            print(f"Class weights: {weights}")
            
            return weights
        except Exception as e:
            print(f"Error computing class weights: {e}")
            return None
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Get data
            inputs = batch['image'].to(self.device)
            targets = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        
        if self.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "epoch": epoch
            })
        
        return train_loss, train_acc
    
    def validate_epoch(self, epoch):
        """Validate one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch_idx, batch in enumerate(pbar):
                # Get data
                inputs = batch['image'].to(self.device)
                targets = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        
        # Update learning rate scheduler
        self.scheduler.step(val_loss)
        
        # Save checkpoint if validation loss is the best we've seen so far
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self.save_checkpoint(epoch, f'best_loss_epoch_{epoch}.pth')
        
        # Save checkpoint if validation accuracy is the best we've seen so far
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self.save_checkpoint(epoch, f'best_acc_epoch_{epoch}.pth')
        
        if self.use_wandb:
            wandb.log({
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]['lr']
            })
        
        return val_loss, val_acc
    
    def train(self, num_epochs=50, patience=10, unfreeze_strategy=True):
        """
        Train the model for the specified number of epochs
        
        Args:
            num_epochs: Number of epochs to train
            patience: Number of epochs to wait for improvement before early stopping
            unfreeze_strategy: Whether to use the gradual unfreezing strategy
        
        Returns:
            Best validation accuracy
        """
        print(f'Training on {self.device}')
        
        # Track time
        start_time = time.time()
        
        # Initialize early stopping counter
        no_improve_epochs = 0
        
        for epoch in range(1, num_epochs + 1):
            print(f'\nEpoch {epoch}/{num_epochs}')
            
            # 实现更积极的解冻策略
            if unfreeze_strategy:
                if epoch == 1:
                    # 从一开始就让部分层可训练
                    print("Phase 1: Training classifier and top layers")
                    self.model.unfreeze_top_layers(num_layers=2)
                
                elif epoch == 5:
                    # 解冻更多层
                    print("Phase 2: Unfreezing more layers")
                    self.model.unfreeze_top_layers(num_layers=5)
                    # 降低学习率
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5
                
                elif epoch == 10:
                    # 解冻整个模型
                    print("Phase 3: Unfreezing all layers")
                    for param in self.model.parameters():
                        param.requires_grad = True
                    # 进一步降低学习率
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.2
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Print progress
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # Check for early stopping
            if val_loss < self.best_val_loss:
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f'Early stopping triggered after {epoch} epochs')
                    break
            
            # Save regular checkpoint every 5 epochs
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')
        
        # Print final results
        elapsed_time = time.time() - start_time
        print(f'\nTraining completed in {elapsed_time / 60:.2f} minutes')
        print(f'Best Validation Loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}')
        print(f'Best Validation Accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}')
        
        # Plot training curves
        self.plot_training_curves()
        
        return self.best_val_acc
    
    def evaluate(self, data_loader=None, checkpoint_path=None, print_metrics=True):
        """
        Evaluate the model on the test set
        
        Args:
            data_loader: DataLoader to use for evaluation (default: test_loader)
            checkpoint_path: Path to checkpoint to load (default: None, use current model state)
            print_metrics: Whether to print evaluation metrics
            
        Returns:
            Dictionary with evaluation metrics
        """
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        if data_loader is None:
            data_loader = self.test_loader
        
        self.model.eval()
        
        all_preds = []
        all_targets = []
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                # Get data
                inputs = batch['image'].to(self.device)
                targets = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Add batch results to lists
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        eval_loss = running_loss / len(data_loader)
        eval_acc = 100. * correct / total
        
        # Convert to numpy arrays for sklearn metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Compute confusion matrix and classification report
        cm = confusion_matrix(all_targets, all_preds)
        cr = classification_report(all_targets, all_preds, target_names=self.class_names, output_dict=True)
        balanced_acc = balanced_accuracy_score(all_targets, all_preds)
        
        # Print results
        if print_metrics:
            print(f'\nTest Loss: {eval_loss:.4f} | Test Acc: {eval_acc:.2f}%')
            print(f'Balanced Accuracy: {balanced_acc:.4f}')
            print('\nConfusion Matrix:')
            print(cm)
            print('\nClassification Report:')
            for cls in self.class_names:
                print(f"{cls}: Precision: {cr[cls]['precision']:.4f}, "
                      f"Recall: {cr[cls]['recall']:.4f}, "
                      f"F1-score: {cr[cls]['f1-score']:.4f}")
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "test_loss": eval_loss,
                "test_acc": eval_acc,
                "balanced_acc": balanced_acc,
                "confusion_matrix": wandb.plots.confusion_matrix(
                    preds=all_preds, y_true=all_targets, class_names=self.class_names
                )
            })
        
        return {
            'loss': eval_loss,
            'accuracy': eval_acc,
            'balanced_accuracy': balanced_acc,
            'confusion_matrix': cm,
            'classification_report': cr
        }
    
    def save_checkpoint(self, epoch, filename=None):
        """Save model checkpoint"""
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pth'
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }, filepath)
        
        print(f'Checkpoint saved to {filepath}')
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_epoch = checkpoint['best_epoch']
        
        print(f'Checkpoint loaded from {checkpoint_path}')
        print(f'Best Validation Loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}')
        print(f'Best Validation Accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}')
    
    def plot_training_curves(self):
        """Plot training and validation loss/accuracy curves"""
        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.train_losses, label='Train')
        ax1.plot(self.val_losses, label='Validation')
        ax1.set_title('Loss vs. Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.train_accs, label='Train')
        ax2.plot(self.val_accs, label='Validation')
        ax2.set_title('Accuracy vs. Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout and save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, 'training_curves.png'))
        
        if self.use_wandb:
            wandb.log({"training_curves": wandb.Image(plt)})
        
        plt.close() 