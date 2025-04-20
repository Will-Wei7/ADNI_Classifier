import os
import glob
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import ndimage
import random
from pathlib import Path
import torchio as tio
from tqdm import tqdm

class SimpleADNIDataset(Dataset):
    """
    简化版ADNI数据集加载器，专注于稳定的数据处理
    支持多级目录结构：train -> AD -> subject_id -> .nii 图像
    """
    def __init__(
        self,
        data_dir,
        transform=None,
        target_shape=(80, 96, 80),
        normalize=True,
        augment=False,
        verbose=False
    ):
        """
        初始化ADNI数据集
        
        Args:
            data_dir (str): 数据目录路径，应包含AD、CN、MCI子目录
            transform: 数据转换（可选）
            target_shape (tuple): 调整大小的目标形状
            normalize (bool): 是否标准化图像数据
            augment (bool): 是否使用数据增强
            verbose (bool): 是否打印详细信息
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_shape = target_shape
        self.normalize = normalize
        self.augment = augment
        self.verbose = verbose
        
        # 类别到标签的映射
        self.class_to_idx = {'AD': 0, 'CN': 1, 'MCI': 2}
        
        # 加载所有文件路径
        self.file_paths = []
        self.labels = []
        
        # 查找所有 .nii 和 .nii.gz 文件
        found_files = 0
        
        # 遍历每个类别文件夹
        for class_name in ['AD', 'CN', 'MCI']:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                if self.verbose:
                    print(f"警告: 未找到目录 {class_dir}")
                continue
            
            # 递归搜索所有 .nii 和 .nii.gz 文件 (不限制层级)
            nii_files = list(class_dir.glob('**/*.nii'))
            nii_files.extend(list(class_dir.glob('**/*.nii.gz')))
            
            found_files += len(nii_files)
            
            if self.verbose:
                print(f"Found {len(nii_files)} files in {class_name} directory")
            
            # 添加文件路径和对应标签
            for file_path in nii_files:
                self.file_paths.append(file_path)
                self.labels.append(self.class_to_idx[class_name])
        
        # 确保数据集非空
        if len(self.file_paths) == 0:
            if found_files > 0:
                raise ValueError(f"由于某种原因，无法添加 {found_files} 个找到的文件")
            else:
                raise ValueError(f"未在 {self.data_dir} 中找到有效的数据文件，请检查路径是否正确")
            
        # 打印数据集统计信息
        if self.verbose:
            print(f"共加载 {len(self.file_paths)} 个样本")
            classes, counts = np.unique(self.labels, return_counts=True)
            for cls, count in zip(classes, counts):
                class_name = list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(cls)]
                print(f"类别 {class_name} ({cls}): {count} 个样本")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # 加载.nii文件
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # 使用nibabel加载数据
            nii_img = nib.load(file_path)
            img_data = nii_img.get_fdata()
            
            # 处理维度，确保3D
            if len(img_data.shape) > 3:
                img_data = img_data[:, :, :, 0]  # 取第一个时间点
            
            # 调整大小到目标形状
            if self.target_shape and img_data.shape != self.target_shape:
                scale_factors = [t / s for s, t in zip(img_data.shape, self.target_shape)]
                img_data = ndimage.zoom(img_data, scale_factors, order=1)
            
            # 标准化
            if self.normalize:
                # 强度归一化: z-score标准化，但限制极值影响
                mean = np.mean(img_data)
                std = np.std(img_data)
                if std > 0:
                    img_data = (img_data - mean) / std
                    # 裁剪极值，避免异常值影响
                    img_data = np.clip(img_data, -3, 3)
                    # 重新缩放到[0,1]范围
                    img_data = (img_data + 3) / 6
            
            # 数据增强
            if self.augment:
                # 简单的随机翻转
                if np.random.rand() > 0.5:
                    img_data = np.flip(img_data, axis=0)
                if np.random.rand() > 0.5:
                    img_data = np.flip(img_data, axis=1)
                
                # 添加微小的高斯噪声
                if np.random.rand() > 0.7:
                    noise = np.random.normal(0, 0.03, img_data.shape)
                    img_data = img_data + noise
                    img_data = np.clip(img_data, 0, 1)  # 确保值在[0,1]范围内
            
            # 转换为torch张量
            img_tensor = torch.from_numpy(img_data.astype(np.float32))
            
            # 添加通道维度
            img_tensor = img_tensor.unsqueeze(0)  # [1, H, W, D]
            
            # 应用其他转换
            if self.transform:
                img_tensor = self.transform(img_tensor)
            
            # 仅返回图像张量和标签，不包含路径（避免collate错误）
            return img_tensor, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
            # 返回一个空张量和原始标签
            empty_tensor = torch.zeros((1, *self.target_shape), dtype=torch.float32)
            return empty_tensor, torch.tensor(label, dtype=torch.long)
    
    def visualize_sample(self, idx):
        """可视化数据集中的一个样本"""
        img_tensor, label = self[idx]
        img = img_tensor.numpy()[0]  # 移除通道维度
        file_path = self.file_paths[idx]  # 获取文件路径用于显示
        
        # 获取类别名称
        class_name = list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(label.item())]
        
        # 创建中间切片的可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 显示三个不同视角的中心切片
        x_center, y_center, z_center = img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2
        
        axes[0].imshow(img[x_center, :, :], cmap='gray')
        axes[0].set_title(f'Sagittal View (x={x_center})')
        axes[0].axis('off')
        
        axes[1].imshow(img[:, y_center, :], cmap='gray')
        axes[1].set_title(f'Coronal View (y={y_center})')
        axes[1].axis('off')
        
        axes[2].imshow(img[:, :, z_center], cmap='gray')
        axes[2].set_title(f'Axial View (z={z_center})')
        axes[2].axis('off')
        
        plt.suptitle(f'Class: {class_name} (Label: {label.item()})\nFile: {os.path.basename(file_path)}')
        plt.tight_layout()
        plt.show()
        return fig

def create_data_augmentation():
    """创建数据增强变换的管道"""
    spatial_transforms = tio.Compose([
        tio.RandomAffine(
            scales=(0.9, 1.1),
            degrees=10,
            translation=5,
            p=0.5
        ),
        tio.RandomFlip(axes=(0,), p=0.5),
    ])
    
    intensity_transforms = tio.Compose([
        tio.RandomNoise(std=0.1, p=0.3),
        tio.RandomGamma(p=0.3),
        tio.RandomBiasField(p=0.3),
    ])
    
    return tio.Compose([
        spatial_transforms,
        intensity_transforms
    ])

def create_data_loaders(
    data_dir, 
    batch_size=4, 
    target_shape=(80, 96, 80),
    val_split=0.15,
    test_split=0.15,
    num_workers=4,
    augment_train=True,
    seed=42,
    verbose=True
):
    """
    从目录创建训练、验证和测试数据加载器
    
    Args:
        data_dir: 数据目录，包含类别子文件夹
        batch_size: 批次大小
        target_shape: 目标图像形状
        val_split: 验证集比例
        test_split: 测试集比例
        num_workers: 数据加载的工作线程数
        augment_train: 是否对训练集进行数据增强
        seed: 随机种子
        verbose: 是否打印详细信息
    
    Returns:
        train_loader, val_loader, test_loader: 训练、验证和测试的数据加载器
    """
    # 确保 data_dir 是 Path 对象
    data_dir = Path(data_dir)
    
    # 打印数据目录信息
    if verbose:
        print(f"加载数据目录: {data_dir}")
        for class_name in ['AD', 'CN', 'MCI']:
            class_dir = data_dir / class_name
            if class_dir.exists():
                files = list(class_dir.glob('**/*.nii'))
                files.extend(list(class_dir.glob('**/*.nii.gz')))
                print(f"  {class_name}: {len(files)} 个文件")
            else:
                print(f"  警告: 未找到目录 {class_dir}")
    
    # 创建完整数据集以获取文件路径和标签
    full_dataset = SimpleADNIDataset(
        data_dir=data_dir,
        target_shape=target_shape,
        normalize=True,
        augment=False,  # 先不增强，等分割后再设置
        verbose=verbose
    )
    
    # 获取所有文件路径和标签
    all_paths = full_dataset.file_paths
    all_labels = full_dataset.labels
    
    # 进行分层随机分割，保持各个类别的比例
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels, 
        test_size=test_split, 
        random_state=seed, 
        stratify=all_labels
    )
    
    # 从训练集中分出验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, 
        test_size=val_split/(1-test_split),  # 调整比例
        random_state=seed, 
        stratify=train_labels
    )
    
    if verbose:
        print(f"数据集分割: 训练 {len(train_paths)}, 验证 {len(val_paths)}, 测试 {len(test_paths)} 个样本")
    
    # 手动设置文件路径和标签，创建三个单独的数据集
    train_dataset = SimpleADNIDataset(
        data_dir=data_dir,
        target_shape=target_shape,
        normalize=True,
        augment=augment_train,
        verbose=False  # 避免重复打印
    )
    train_dataset.file_paths = train_paths
    train_dataset.labels = train_labels
    
    val_dataset = SimpleADNIDataset(
        data_dir=data_dir,
        target_shape=target_shape,
        normalize=True,
        augment=False,  # 不对验证集进行增强
        verbose=False
    )
    val_dataset.file_paths = val_paths
    val_dataset.labels = val_labels
    
    test_dataset = SimpleADNIDataset(
        data_dir=data_dir,
        target_shape=target_shape,
        normalize=True,
        augment=False,  # 不对测试集进行增强
        verbose=False
    )
    test_dataset.file_paths = test_paths
    test_dataset.labels = test_labels
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # 打印类别分布信息
    if verbose:
        for name, dataset in [("训练", train_dataset), ("验证", val_dataset), ("测试", test_dataset)]:
            classes, counts = np.unique(dataset.labels, return_counts=True)
            print(f"{name}集类别分布:")
            for cls, count in zip(classes, counts):
                class_name = list(dataset.class_to_idx.keys())[list(dataset.class_to_idx.values()).index(cls)]
                print(f"  {class_name} ({cls}): {count} 个样本 ({count/len(dataset.labels)*100:.1f}%)")
    
    return train_loader, val_loader, test_loader 