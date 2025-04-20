import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import wandb

class SimpleTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        lr=0.001,
        weight_decay=1e-5,
        device=None,
        checkpoint_dir='./checkpoints',
        use_wandb=False
    ):
        """
        简单的训练器，专注于3D MRI分类
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 设置损失函数，使用类别平衡的权重
        class_weights = self._compute_class_weights()
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            print(f"Using weighted cross-entropy loss with weights: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("Using standard cross-entropy loss")
        
        # 设置优化器，使用AdamW
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        # 跟踪性能指标
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # 移动模型到设备
        self.model.to(self.device)
        
        # 初始化wandb
        if self.use_wandb:
            wandb.init(project="ADNI-Simple-Classifier")
            wandb.config.update({
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "model": model.__class__.__name__,
                "optimizer": self.optimizer.__class__.__name__,
            })
            wandb.watch(self.model)
    
    def _compute_class_weights(self):
        """计算类别权重以处理可能的类别不平衡"""
        try:
            # 统计每个类别的样本数
            class_counts = torch.zeros(3)  # 假设3个类别: AD, CN, MCI
            for batch in self.train_loader:
                if isinstance(batch, dict) and 'label' in batch:
                    labels = batch['label']
                    for label in labels:
                        class_counts[label] += 1
            
            # 确保没有类别样本数为0
            if torch.any(class_counts == 0):
                print("Warning: Some classes have zero samples!")
                return None
            
            # 计算权重：类别的倒数，以便对少数类别给予更大权重
            weights = 1.0 / class_counts
            # 归一化权重
            weights = weights * (len(class_counts) / weights.sum())
            
            print(f"Class counts: {class_counts}")
            print(f"Class weights: {weights}")
            
            return weights
        except Exception as e:
            print(f"Error computing class weights: {e}")
            return None
    
    def train(self, num_epochs=30, patience=10):
        """训练模型"""
        print(f"Training on {self.device}")
        best_accuracy = 0.0
        counter = 0  # 早停计数器
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # 训练一个轮次
            train_loss, train_acc = self._train_epoch()
            
            # 验证
            val_loss, val_acc, val_balanced_acc = self._validate()
            
            # 更新学习率
            self.scheduler.step(val_balanced_acc)
            
            # 打印结果
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Balanced Acc: {val_balanced_acc:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 记录指标
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # 早停检查
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                counter = 0
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                
                # 保存最佳模型
                self._save_checkpoint(epoch, is_best=True)
                print(f"Saved best model with accuracy: {val_acc:.2f}%")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
            
            # 每5个轮次保存一次检查点
            if epoch % 5 == 0:
                self._save_checkpoint(epoch)
            
            # 记录到wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_balanced_acc': val_balanced_acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        # 训练完成统计
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time / 60:.2f} minutes")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
        
        # 绘制训练曲线
        self._plot_training_curves()
        
        # 加载最佳模型并评估
        best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            self._load_checkpoint(best_model_path)
            return self.evaluate()
        
        return self.best_val_acc
    
    def _train_epoch(self):
        """训练一个轮次"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 使用tqdm显示进度条
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            # 获取数据并移动到设备
            inputs = batch['image'].to(self.device)
            targets = batch['label'].to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播并优化
            loss.backward()
            self.optimizer.step()
            
            # 更新统计信息
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条信息
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def _validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # 获取数据并移动到设备
                inputs = batch['image'].to(self.device)
                targets = batch['label'].to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 更新统计信息
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 保存预测和目标，用于计算混淆矩阵和平衡准确率
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # 计算平衡准确率
        balanced_acc = balanced_accuracy_score(all_targets, all_preds)
        
        # 打印混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)
        print("\nConfusion Matrix:")
        print(cm)
        
        return avg_loss, accuracy, balanced_acc
    
    def evaluate(self):
        """在测试集上评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # 获取数据并移动到设备
                inputs = batch['image'].to(self.device)
                targets = batch['label'].to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                # 更新统计信息
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 保存预测和目标
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = 100. * correct / total
        balanced_acc = balanced_accuracy_score(all_targets, all_preds)
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)
        
        print("\nTest Results:")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        
        # 记录到wandb
        if self.use_wandb:
            wandb.log({
                'test_accuracy': accuracy,
                'test_balanced_accuracy': balanced_acc,
                'confusion_matrix': wandb.plot.confusion_matrix(
                    y_true=np.array(all_targets),
                    preds=np.array(all_preds),
                    class_names=['AD', 'CN', 'MCI']
                )
            })
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def _save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        filename = f'checkpoint_epoch_{epoch}.pth' if not is_best else 'best_model.pth'
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }, filepath)
    
    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Training Accuracy')
        plt.plot(self.val_accs, label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, 'training_curves.png'))
        plt.close()
        
        if self.use_wandb:
            wandb.log({"training_curves": wandb.Image(os.path.join(self.checkpoint_dir, 'training_curves.png'))}) 