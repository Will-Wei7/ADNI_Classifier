import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchio as tio
import nibabel as nib
from pathlib import Path
import monai
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityd,
    NormalizeIntensityd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    CropForegroundd,
    SpatialPadd,
    RandSpatialCropd,
    Resized
)

class ADNIDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_shape=(128, 128, 128), flatten_dir=False):
        """
        Dataset for Alzheimer's Disease MRI classification
        
        Args:
            data_dir (str): Path to the dataset directory with class folders (AD/CN/MCI)
                            或者预处理后的数据目录
            transform (callable, optional): Optional transform to be applied to the data
            target_shape (tuple): Target shape for resizing the MRI volumes
            flatten_dir (bool): 如果True，则假设所有样本都在一级目录中，通过文件名判断类别
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_shape = target_shape
        self.flatten_dir = flatten_dir
        
        # Get all image paths and labels
        self.samples = []
        self.class_map = {'AD': 0, 'CN': 1, 'MCI': 2}  # Maps folder names to class indices
        
        # 打印数据目录结构，帮助调试
        print(f"Loading data from: {self.data_dir}")
        if os.path.exists(self.data_dir):
            print(f"Data directory exists. Contents: {os.listdir(self.data_dir)}")
        else:
            print(f"WARNING: Data directory {self.data_dir} does not exist!")
            return
            
        # 检查是否是预处理后的扁平结构
        if self.flatten_dir:
            self._load_preprocessed_data()
        else:
            self._load_class_structured_data()
            
        # 如果仍然没有样本，尝试自动检测数据结构
        if not self.samples:
            print("No samples found with the expected structure. Attempting to auto-detect data structure...")
            # 先尝试扁平结构
            self.flatten_dir = True
            self._load_preprocessed_data()
            
            # 如果仍然没有找到样本，尝试在第一级子目录中查找NIfTI文件
            if not self.samples:
                print("Looking for NIfTI files in first-level subdirectories...")
                for subdir in self.data_dir.iterdir():
                    if subdir.is_dir():
                        nii_files = list(subdir.glob('*.nii')) + list(subdir.glob('*.nii.gz'))
                        for nii_file in nii_files:
                            # 尝试从文件名或目录名推断类别
                            class_name = self._guess_class_from_path(nii_file)
                            if class_name in self.class_map:
                                self.samples.append({
                                    'image_path': str(nii_file),
                                    'label': self.class_map[class_name],
                                    'patient_id': subdir.name,
                                    'class_name': class_name
                                })
        
        print(f"Found {len(self.samples)} samples across {len(set([s['class_name'] for s in self.samples]))} classes")
        # 按类别打印样本数量
        class_counts = {}
        for sample in self.samples:
            class_name = sample['class_name']
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        for class_name, count in class_counts.items():
            print(f"  - {class_name}: {count} samples")
    
    def _load_class_structured_data(self):
        """
        加载按类别组织的数据结构 (data_dir/class_name/patient_id/*.nii)
        """
        for class_name in self.class_map.keys():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Class directory not found: {class_dir}")
                continue
            
            print(f"Processing class directory: {class_dir}")
            for patient_dir in class_dir.iterdir():
                if not patient_dir.is_dir():
                    continue
                
                # Find all .nii files in the patient directory (recursively)
                nii_files = list(patient_dir.glob('**/*.nii')) + list(patient_dir.glob('**/*.nii.gz'))
                if nii_files:
                    # Use the first .nii file found
                    self.samples.append({
                        'image_path': str(nii_files[0]),
                        'label': self.class_map[class_name],
                        'patient_id': patient_dir.name,
                        'class_name': class_name
                    })
                else:
                    print(f"No NIfTI files found in patient directory: {patient_dir}")
    
    def _load_preprocessed_data(self):
        """
        加载预处理后的扁平数据结构
        预期结构: data_dir/patient_id/patient_id_preprocessed.nii[.gz]
        或者 data_dir/patient_id/*.nii[.gz]
        """
        # 列出所有子目录
        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        for subdir in subdirs:
            # 查找该目录下的所有NIfTI文件
            nii_files = list(subdir.glob('*.nii')) + list(subdir.glob('*.nii.gz'))
            
            if not nii_files:
                print(f"No NIfTI files found in directory: {subdir}")
                continue
                
            # 从目录名或文件名中推断类别
            patient_id = subdir.name
            
            # 首先尝试使用预处理后的文件
            preprocessed_files = [f for f in nii_files if 'preprocessed' in f.name.lower()]
            if preprocessed_files:
                nii_file = preprocessed_files[0]  # 使用第一个预处理文件
            else:
                nii_file = nii_files[0]  # 使用第一个找到的文件
            
            # 从文件名或目录名尝试推断类别
            class_name = self._guess_class_from_path(nii_file, patient_id)
            
            if class_name in self.class_map:
                self.samples.append({
                    'image_path': str(nii_file),
                    'label': self.class_map[class_name],
                    'patient_id': patient_id,
                    'class_name': class_name
                })
                print(f"Added sample: {nii_file} (Class: {class_name})")
            else:
                print(f"Could not determine class for: {nii_file}, patient_id: {patient_id}")
    
    def _guess_class_from_path(self, file_path, patient_id=None):
        """
        尝试从文件路径或病人ID中推断类别
        
        常见ADNI命名模式:
        - AD_002_S_0295/002_S_0295_preprocessed.nii
        - CN_123_S_0088/123_S_0088_preprocessed.nii
        - 002_S_0295/002_S_0295_preprocessed.nii (假设所有AD在同一个ID范围)
        """
        path_str = str(file_path).upper()
        
        # 直接检查是否含有类别关键词
        if '_AD_' in path_str or '/AD/' in path_str or 'ALZHEIMER' in path_str:
            return 'AD'
        elif '_CN_' in path_str or '/CN/' in path_str or 'CONTROL' in path_str:
            return 'CN'
        elif '_MCI_' in path_str or '/MCI/' in path_str or 'MILD_COGNITIVE' in path_str:
            return 'MCI'
        
        # 如果提供了patient_id，尝试通过ID范围推断
        if patient_id:
            # 如果ID带有类别前缀
            if patient_id.upper().startswith('AD_'):
                return 'AD'
            elif patient_id.upper().startswith('CN_'):
                return 'CN'
            elif patient_id.upper().startswith('MCI_'):
                return 'MCI'
            
            # 通过ID中的数字部分猜测类别
            # 以ADNI数据集为例，可以根据ID范围推断类别
            # 注意：这只是一个示例，实际范围可能不同
            try:
                # 假设ID格式为 "123_S_4567"，提取第一个数字部分
                id_parts = patient_id.split('_')
                if len(id_parts) >= 1:
                    try:
                        num = int(id_parts[0])
                        # 这里使用假设的ID范围
                        if 0 <= num < 300:  # 假设ID范围
                            return 'AD'
                        elif 300 <= num < 600:  # 假设ID范围
                            return 'CN'
                        elif 600 <= num < 900:  # 假设ID范围
                            return 'MCI'
                    except ValueError:
                        pass
            except:
                pass
        
        # 尝试解析ADNI文件命名模式
        try:
            # 提取文件名
            file_name = file_path.name
            parts = file_name.split('_')
            
            # 检查文件名中的类别标识
            for part in parts:
                if part.upper() == 'AD':
                    return 'AD'
                elif part.upper() == 'CN':
                    return 'CN'
                elif part.upper() == 'MCI':
                    return 'MCI'
        except:
            pass
        
        # 如果无法确定类别，返回None
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load the NIfTI file
            img_path = os.path.abspath(sample['image_path'])
            nii_img = nib.load(img_path)
            
            # Convert to numpy array
            img_data = nii_img.get_fdata()
            
            # Ensure the data has the expected dimensions (add channel if needed)
            if len(img_data.shape) == 3:
                img_data = img_data[np.newaxis, ...]  # Add channel dimension
            
            # Convert to torch tensor
            img_tensor = torch.from_numpy(img_data).float()
            
            # Apply transforms if provided
            if self.transform:
                if isinstance(self.transform, tio.Transform):
                    # TorchIO transform
                    subject = tio.Subject(image=tio.ScalarImage(tensor=img_tensor))
                    transformed_subject = self.transform(subject)
                    img_tensor = transformed_subject['image'].data
                else:
                    # Regular transform
                    img_tensor = self.transform(img_tensor)
            
            return {
                'image': img_tensor, 
                'label': sample['label'],
                'patient_id': sample['patient_id'],
                'class_name': sample['class_name']
            }
            
        except Exception as e:
            print(f"Error loading {sample['image_path']}: {e}")
            # Return empty tensor with proper shape as fallback
            empty_tensor = torch.zeros((1,) + self.target_shape)
            return {
                'image': empty_tensor,
                'label': sample['label'],
                'patient_id': sample['patient_id'],
                'class_name': sample['class_name']
            }

def get_transforms(mode='train', target_shape=(128, 128, 128)):
    """
    Get the transforms for data preprocessing and augmentation
    
    Args:
        mode (str): 'train' or 'val' or 'test'
        target_shape (tuple): Target shape for resizing
        
    Returns:
        transform: Composition of transforms
    """
    base_transforms = [
        # Skull stripping would normally happen here, but we assume it's already done
        # Registration to standard space would also be done here if needed
        
        # Basic orientation, spacing standardization
        tio.ToCanonical(),
        tio.Resample(1.0),  # Resample to 1mm isotropic
        
        # Brain extraction and foreground focus
        tio.CropOrPad(target_shape),
        
        # Intensity normalization
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ]
    
    if mode == 'train':
        # Add augmentations for training
        augmentations = [
            # Spatial augmentations
            tio.RandomAffine(
                scales=(0.9, 1.1),
                degrees=10,
                translation=10,
                p=0.75
            ),
            tio.RandomElasticDeformation(
                num_control_points=7,
                max_displacement=5,
                p=0.5
            ),
            tio.RandomFlip(axes=(0,), p=0.5),  # Sagittal flip
            
            # Intensity augmentations
            tio.RandomGamma(p=0.3),
            tio.RandomNoise(p=0.3),
            tio.RandomBiasField(p=0.3)
        ]
        return tio.Compose(base_transforms + augmentations)
    else:
        return tio.Compose(base_transforms)

def get_monai_transforms(mode='train', target_shape=(128, 128, 128)):
    """
    Alternative implementation using MONAI transforms
    
    Args:
        mode (str): 'train' or 'val' or 'test'
        target_shape (tuple): Target shape for resizing
        
    Returns:
        transform: Composition of transforms
    """
    # Common transforms for all modes
    common_transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=target_shape),
        Resized(keys=["image"], spatial_size=target_shape),
        ScaleIntensityd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)
    ]
    
    if mode == 'train':
        train_transforms = [
            RandAffined(
                keys=["image"],
                prob=0.75,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                translate_range=(10, 10, 10),
                padding_mode="zeros"
            ),
            RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
            RandGaussianNoised(keys=["image"], prob=0.3, std=0.01),
        ]
        return monai.transforms.Compose(common_transforms + train_transforms)
    else:
        return monai.transforms.Compose(common_transforms)

def prepare_monai_dataset(data_dir, mode='train', target_shape=(128, 128, 128)):
    """
    Prepare a MONAI Dataset for the ADNI data
    
    Args:
        data_dir (str): Path to the data directory
        mode (str): 'train' or 'val' or 'test'
        target_shape (tuple): Target shape for resizing
        
    Returns:
        MONAI Dataset
    """
    from monai.data import Dataset
    
    # Collect data
    data_dir = Path(data_dir)
    data = []
    class_map = {'AD': 0, 'CN': 1, 'MCI': 2}
    
    for class_name in class_map.keys():
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        
        for patient_dir in class_dir.iterdir():
            if not patient_dir.is_dir():
                continue
            
            # Find all .nii files in the patient directory (recursively)
            nii_files = list(patient_dir.glob('**/*.nii')) + list(patient_dir.glob('**/*.nii.gz'))
            if nii_files:
                # Use the first .nii file found
                data.append({
                    'image': str(nii_files[0]),
                    'label': class_map[class_name],
                    'patient_id': patient_dir.name
                })
    
    # Create transform
    transform = get_monai_transforms(mode=mode, target_shape=target_shape)
    
    # Create dataset
    dataset = Dataset(data=data, transform=transform)
    
    return dataset

def get_data_loaders(data_dir="./ADNI_split", batch_size=4, target_shape=(128, 128, 128), num_workers=4, 
                    flatten_dir=None):
    """
    Get data loaders for training, validation, and testing
    
    Args:
        data_dir (str): Path to the data directory
        batch_size (int): Batch size
        target_shape (tuple): Target shape for resizing
        num_workers (int): Number of workers for data loading
        flatten_dir (bool): Whether the data directory has a flattened structure
                           If None, will auto-detect
        
    Returns:
        dict: Dictionary with train, val, and test data loaders
    """
    print(f"Setting up data loaders for directory: {data_dir}")
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # 检查train和test子目录是否存在
    train_exists = os.path.exists(train_dir)
    test_exists = os.path.exists(test_dir)
    
    # 如果不存在，使用整个目录
    if not (train_exists and test_exists):
        print(f"Train/test subdirectories not found. Using entire directory: {data_dir}")
        train_dir = data_dir
        test_dir = data_dir
    else:
        print(f"Using train directory: {train_dir}")
        print(f"Using test directory: {test_dir}")
    
    # 自动检测数据结构
    if flatten_dir is None:
        # 检查目录结构
        class_dirs_exist = any(os.path.isdir(os.path.join(train_dir, d)) and d in ['AD', 'CN', 'MCI'] 
                              for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
        flatten_dir = not class_dirs_exist
        print(f"Auto-detected data structure: {'flattened' if flatten_dir else 'class-based'}")
    
    train_transform = get_transforms(mode='train', target_shape=target_shape)
    val_transform = get_transforms(mode='val', target_shape=target_shape)
    test_transform = get_transforms(mode='test', target_shape=target_shape)
    
    train_dataset = ADNIDataset(train_dir, transform=train_transform, target_shape=target_shape, flatten_dir=flatten_dir)
    test_dataset = ADNIDataset(test_dir, transform=test_transform, target_shape=target_shape, flatten_dir=flatten_dir)
    
    # 确保数据集不为空
    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found in {train_dir}. Please check the data directory structure.")
    
    if len(test_dataset) == 0:
        print(f"Warning: No test samples found in {test_dir}. Will use a portion of training data for testing.")
        test_dataset = train_dataset
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    # Custom collate function to handle potential errors
    def collate_fn(batch):
        # Filter out any None batches
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            # If no valid samples, return empty batch with correct structure
            return {
                'image': torch.empty((0, 1) + target_shape),
                'label': torch.empty(0, dtype=torch.long),
                'patient_id': [],
                'class_name': []
            }
        
        # Standard collation
        return torch.utils.data.default_collate(batch)
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    
    print(f"Created data loaders with: {len(train_subset)} training, {len(val_subset)} validation, {len(test_dataset)} test samples")
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

# Example usage:
if __name__ == "__main__":
    # For debugging
    data_dir = "./ADNI_preprocessed"
    loaders = get_data_loaders(data_dir)
    
    for i, batch in enumerate(loaders['train']):
        print(f"Batch {i}, shape: {batch['image'].shape}, labels: {batch['label']}")
        if i == 0:
            break 