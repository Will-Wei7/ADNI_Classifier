import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConv3DClassifier(nn.Module):
    """
    简单的3D卷积分类器，用于MRI图像分类
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        base_filters=8,
        dropout_rate=0.3
    ):
        """
        初始化简单的3D CNN模型
        
        Args:
            in_channels: 输入通道数
            num_classes: 分类类别数（AD, CN, MCI）
            base_filters: 基础滤波器数量，后续层会翻倍
            dropout_rate: Dropout比率
        """
        super(SimpleConv3DClassifier, self).__init__()
        
        # 保存参数
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_filters = base_filters
        
        # 第一个编码块
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)  # 降采样
        )
        
        # 第二个编码块
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_filters, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_filters*2, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)  # 降采样
        )
        
        # 第三个编码块
        self.enc3 = nn.Sequential(
            nn.Conv3d(base_filters*2, base_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters*4),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_filters*4, base_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters*4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)  # 降采样
        )
        
        # 第四个编码块
        self.enc4 = nn.Sequential(
            nn.Conv3d(base_filters*4, base_filters*8, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters*8),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_filters*8, base_filters*8, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters*8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)  # 降采样
        )
        
        # 计算全连接层的输入特征数量
        # 假设输入大小为 (80, 96, 80)
        # 经过4次2x2x2池化后，大小变为 (5, 6, 5)
        self.fc_input_size = base_filters * 8 * 5 * 6 * 5
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [B, C, D, H, W]
        
        Returns:
            logits: 输出预测，形状为 [B, num_classes]
        """
        # 编码路径
        x = self.enc1(x)  # [B, base_filters, D/2, H/2, W/2]
        x = self.enc2(x)  # [B, base_filters*2, D/4, H/4, W/4]
        x = self.enc3(x)  # [B, base_filters*4, D/8, H/8, W/8]
        x = self.enc4(x)  # [B, base_filters*8, D/16, H/16, W/16]
        
        # 展平
        x = x.view(x.size(0), -1)  # [B, base_filters*8*D/16*H/16*W/16]
        
        # 全连接层
        logits = self.fc(x)  # [B, num_classes]
        
        return logits
    
    def predict(self, x):
        """
        预测函数，返回类别概率和预测类别
        
        Args:
            x: 输入张量
        
        Returns:
            probs: 类别概率
            preds: 预测类别
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return probs, preds


class SimpleMRIClassifier(nn.Module):
    """
    一个更轻量级的MRI分类器，适用于资源受限的环境
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        dropout_rate=0.3
    ):
        """
        初始化轻量级的3D CNN模型
        
        Args:
            in_channels: 输入通道数
            num_classes: 分类类别数（AD, CN, MCI）
            dropout_rate: Dropout比率
        """
        super(SimpleMRIClassifier, self).__init__()
        
        # 编码层
        self.encoder = nn.Sequential(
            # 第一层: 输入 -> 16通道
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            
            # 第二层: 16 -> 32通道
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # 第三层: 32 -> 64通道
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # 第四层: 64 -> 128通道
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # 空间平均池化
            nn.AdaptiveAvgPool3d(1)  # 输出: [B, 128, 1, 1, 1]
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [B, C, D, H, W]
        
        Returns:
            logits: 输出预测，形状为 [B, num_classes]
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
    
    def predict(self, x):
        """
        预测函数，返回类别概率和预测类别
        
        Args:
            x: 输入张量
        
        Returns:
            probs: 类别概率
            preds: 预测类别
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return probs, preds


# 工厂函数，便于创建不同大小的模型
def create_model(model_name='simple', in_channels=1, num_classes=3, **kwargs):
    """
    创建模型的工厂函数
    
    Args:
        model_name: 模型名称，'simple'或'conv3d'
        in_channels: 输入通道数
        num_classes: 分类类别数
        **kwargs: 其他参数，传递给对应的模型构造函数
    
    Returns:
        model: 创建的模型实例
    """
    if model_name == 'simple':
        return SimpleMRIClassifier(in_channels=in_channels, num_classes=num_classes, **kwargs)
    elif model_name == 'conv3d':
        return SimpleConv3DClassifier(in_channels=in_channels, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_name}") 