import os
import argparse
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 导入我们的自定义模块
from simple_model import create_model
from data_loader_simple import SimpleADNIDataset, create_data_loaders  # 恢复使用我们的数据加载器

def parse_args():
    parser = argparse.ArgumentParser(description='训练简单的3D MRI分类模型')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--model_type', type=str, default='simple', choices=['simple', 'conv3d'], help='选择模型类型')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器的工作进程数')
    parser.add_argument('--target_shape', type=str, default='80,96,80', help='目标体积形状，格式：D,H,W')
    parser.add_argument('--normalize', action='store_true', help='是否对输入数据进行归一化')
    parser.add_argument('--augment', action='store_true', help='是否使用数据增强')
    parser.add_argument('--val_split', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--test_split', type=float, default=0.15, help='测试集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--checkpoint', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--verbose', action='store_true', help='是否显示详细输出')
    parser.add_argument('--early_stopping', type=int, default=10, help='早停的耐心值')
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子以确保结果可重现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    preds_list = []
    targets_list = []
    
    for inputs, targets in tqdm(train_loader, desc="训练中"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        
        # 收集预测和标签
        _, preds = torch.max(outputs, 1)
        preds_list.extend(preds.cpu().numpy())
        targets_list.extend(targets.cpu().numpy())
    
    # 计算平均损失和准确率
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(targets_list, preds_list)
    
    return epoch_loss, epoch_acc, preds_list, targets_list

def validate(model, val_loader, criterion, device):
    """在验证集上评估模型"""
    model.eval()
    running_loss = 0.0
    preds_list = []
    targets_list = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="验证中"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            
            # 收集预测和标签
            _, preds = torch.max(outputs, 1)
            preds_list.extend(preds.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
    
    # 计算平均损失和准确率
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(targets_list, preds_list)
    
    return epoch_loss, epoch_acc, preds_list, targets_list

def plot_metrics(train_losses, val_losses, train_accs, val_accs, output_dir):
    """绘制训练过程中的指标变化"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('训练与验证损失')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('训练与验证准确率')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()

def plot_confusion_matrix(cm, class_names, output_dir):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 添加文本注释
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def main(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析目标形状
    if ',' in args.target_shape:
        target_shape = tuple(map(int, args.target_shape.split(',')))
    else:
        # 如果 target_shape 参数格式为 'D H W'
        target_shape = tuple(map(int, args.target_shape.split()))
    
    # 构建数据加载器
    print("正在加载数据集...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        target_shape=target_shape,
        augment_train=args.augment,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        seed=args.seed,
        verbose=args.verbose
    )
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    print(f"创建 {args.model_type} 模型...")
    model = create_model(
        model_name=args.model_type,
        in_channels=1,
        num_classes=3  # AD, CN, MCI
    )
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 恢复检查点（如果有）
    start_epoch = 0
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"加载检查点 {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        train_accs = checkpoint.get('train_accs', [])
        val_accs = checkpoint.get('val_accs', [])
        print(f"恢复训练，从epoch {start_epoch}开始")
    
    # 早停
    patience = args.early_stopping
    patience_counter = 0
    
    # 训练循环
    print(f"开始训练，共 {args.epochs} 轮...")
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练一个epoch
        train_loss, train_acc, _, _ = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        
        # 验证
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f"保存最佳模型，验证准确率: {val_acc:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
            }, os.path.join(args.output_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停激活 - {patience} 轮未提升")
                break
        
        # 保存检查点
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # 训练结束，绘制指标
    plot_metrics(train_losses, val_losses, train_accs, val_accs, args.output_dir)
    
    # 加载最佳模型进行测试
    print("\n加载最佳模型进行测试...")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 在测试集上评估
    test_loss, test_acc, test_preds, test_targets = validate(model, test_loader, criterion, device)
    print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")
    
    # 计算混淆矩阵
    cm = confusion_matrix(test_targets, test_preds)
    class_names = ['AD', 'CN', 'MCI']  # 类别名称
    plot_confusion_matrix(cm, class_names, args.output_dir)
    
    # 打印分类报告
    report = classification_report(test_targets, test_preds, target_names=class_names)
    print("\n分类报告:")
    print(report)
    
    # 将分类报告保存到文件
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # 打印训练时间
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n训练总时间: {int(hours)}时 {int(minutes)}分 {int(seconds)}秒")

if __name__ == '__main__':
    args = parse_args()
    main(args) 