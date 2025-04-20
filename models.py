import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121, SENet154, ViT
# 修复 ResNet 导入
try:
    # 尝试导入新版本的 MONAI ResNet
    from monai.networks.nets import ResNet
    HAS_RESNET_CLASS = True
except (ImportError, AttributeError):
    # 如果导入失败，设置标志
    HAS_RESNET_CLASS = False

class MedicalNet3DResNet(nn.Module):
    """
    3D ResNet model for medical imaging based on MedicalNet's architecture
    """
    def __init__(self, num_classes=3, pretrained=True, model_depth=50, input_channels=1):
        super(MedicalNet3DResNet, self).__init__()
        
        # Model type selection
        model_dict = {
            18: (BasicBlock, [2, 2, 2, 2]),
            34: (BasicBlock, [3, 4, 6, 3]),
            50: (Bottleneck, [3, 4, 6, 3]),
            101: (Bottleneck, [3, 4, 23, 3]),
            152: (Bottleneck, [3, 8, 36, 3])
        }
        
        self.inplanes = 64
        block, layers = model_dict[model_depth]
        
        # Initial convolution
        self.conv1 = nn.Conv3d(
            input_channels, 64, kernel_size=7, stride=(2, 2, 2), padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Final classification layer
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Feature dimension depends on the block type
        if isinstance(block, BasicBlock):
            feature_dim = 512
        else:  # Bottleneck
            feature_dim = 512 * block.expansion
        
        self.fc = nn.Linear(feature_dim, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if pretrained:
            # In a real scenario, we would load pretrained weights here
            # This is a placeholder to show where you would load pretrained weights
            # self._load_pretrained_weights()
            print("Pretrained model would be loaded here if available")
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def _load_pretrained_weights(self):
        # This method would load pretrained weights from a file
        pass


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class TransferLearningModel(nn.Module):
    """
    Class for transfer learning with various 3D model backbones
    """
    def __init__(self, model_name='resnet', num_classes=3, pretrained=True, freeze_backbone=True):
        super(TransferLearningModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        # Choose backbone model
        if model_name == 'densenet':
            self.backbone = DenseNet121(
                spatial_dims=3,
                in_channels=1,
                out_channels=num_classes,
                pretrained=pretrained
            )
            self.feature_dim = 1024  # DenseNet121 final feature dim
            
        elif model_name == 'resnet':
            try:
                if HAS_RESNET_CLASS:
                    # 使用 MONAI 的 ResNet 类
                    self.backbone = ResNet(
                        pretrained=pretrained,
                        spatial_dims=3,
                        n_input_channels=1,
                        num_classes=num_classes
                    )
                else:
                    # 如果 MONAI 的 ResNet 类不可用，使用自定义 ResNet
                    raise ImportError("MONAI ResNet not available")
            except Exception as e:
                print(f"Error initializing MONAI ResNet: {e}")
                print("Falling back to custom MedicalNet3DResNet")
                self.model_name = 'medicalnet'
                self.backbone = MedicalNet3DResNet(
                    num_classes=0,  # No classification layer
                    pretrained=pretrained,
                    model_depth=50,
                    input_channels=1
                )
            
            self.feature_dim = 2048  # ResNet50 final feature dim
            
        elif model_name == 'senet':
            try:
                self.backbone = SENet154(
                    spatial_dims=3,
                    in_channels=1,
                    num_classes=num_classes,
                    pretrained=pretrained
                )
                self.feature_dim = 2048  # SENet154 final feature dim
            except Exception as e:
                print(f"Error loading SENet154: {e}")
                print("Falling back to custom MedicalNet3DResNet")
                self.model_name = 'medicalnet'
                self.backbone = MedicalNet3DResNet(
                    num_classes=0,  # No classification layer
                    pretrained=pretrained,
                    model_depth=50,
                    input_channels=1
                )
                self.feature_dim = 2048  # ResNet50 final feature dim
            
        elif model_name == 'vit':
            try:
                self.backbone = ViT(
                    in_channels=1,
                    img_size=(128, 128, 128),
                    patch_size=(16, 16, 16),
                    num_classes=num_classes,
                    spatial_dims=3
                )
                self.feature_dim = 768  # ViT final feature dim
            except Exception as e:
                print(f"Error loading ViT: {e}")
                print("Falling back to custom MedicalNet3DResNet")
                self.model_name = 'medicalnet'
                self.backbone = MedicalNet3DResNet(
                    num_classes=0,  # No classification layer
                    pretrained=pretrained,
                    model_depth=50,
                    input_channels=1
                )
                self.feature_dim = 2048  # ResNet50 final feature dim
            
        else:
            self.backbone = MedicalNet3DResNet(
                num_classes=0,  # No classification layer
                pretrained=pretrained,
                model_depth=50,
                input_channels=1
            )
            self.feature_dim = 2048  # ResNet50 final feature dim
        
        # Replace or add custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all parameters in the backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_top_layers(self, num_layers=3):
        """Unfreeze the top N layers of the backbone for fine-tuning"""
        # This method needs to be adapted based on the specific architecture
        # Here's a simplified version
        all_layers = list(self.backbone.modules())
        for layer in all_layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        try:
            # Extract features from the backbone
            if self.model_name in ['densenet', 'resnet', 'senet', 'vit']:
                # For MONAI models, we need to extract features before the final layer
                # Different models may have different feature extraction methods
                if hasattr(self.backbone, 'features'):
                    features = self.backbone.features(x)
                    features = F.adaptive_avg_pool3d(features, 1)
                    features = torch.flatten(features, 1)
                else:
                    # For newer MONAI models with different structure
                    # Temporarily disable classifier if it exists
                    if hasattr(self.backbone, 'class_layers'):
                        orig_class_layers = self.backbone.class_layers
                        self.backbone.class_layers = nn.Identity()
                        features = self.backbone(x)
                        self.backbone.class_layers = orig_class_layers
                    elif hasattr(self.backbone, 'fc'):
                        # For ResNet models
                        x = self.backbone.conv1(x)
                        x = self.backbone.bn1(x)
                        x = self.backbone.relu(x)
                        x = self.backbone.maxpool(x)
                        
                        x = self.backbone.layer1(x)
                        x = self.backbone.layer2(x)
                        x = self.backbone.layer3(x)
                        x = self.backbone.layer4(x)
                        
                        features = self.backbone.avgpool(x)
                        features = torch.flatten(features, 1)
                    else:
                        # Generic fallback
                        x_copy = x.clone()
                        try:
                            features = self.backbone(x_copy)
                            # If features has the same shape as num_classes, then we need to extract earlier features
                            if features.shape[1] == self.num_classes:
                                raise ValueError("Model returned classification outputs, need features")
                        except:
                            # Use MedicalNet approach if backbone features extraction fails
                            x = self.backbone.conv1(x) if hasattr(self.backbone, 'conv1') else x
                            x = self.backbone.bn1(x) if hasattr(self.backbone, 'bn1') else x
                            x = self.backbone.relu(x) if hasattr(self.backbone, 'relu') else x
                            x = self.backbone.maxpool(x) if hasattr(self.backbone, 'maxpool') else x
                            
                            x = self.backbone.layer1(x) if hasattr(self.backbone, 'layer1') else x
                            x = self.backbone.layer2(x) if hasattr(self.backbone, 'layer2') else x
                            x = self.backbone.layer3(x) if hasattr(self.backbone, 'layer3') else x
                            x = self.backbone.layer4(x) if hasattr(self.backbone, 'layer4') else x
                            
                            features = F.adaptive_avg_pool3d(x, 1)
                            features = torch.flatten(features, 1)
            else:
                # For custom MedicalNet model
                x = self.backbone.conv1(x)
                x = self.backbone.bn1(x)
                x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                
                x = self.backbone.layer1(x)
                x = self.backbone.layer2(x)
                x = self.backbone.layer3(x)
                x = self.backbone.layer4(x)
                
                features = self.backbone.avgpool(x)
                features = torch.flatten(features, 1)
            
            # Pass features through the classifier
            output = self.classifier(features)
            
            return output
        
        except Exception as e:
            print(f"Error in forward pass: {e}")
            # Fall back to using our custom model implementation
            x = self.backbone.conv1(x) if hasattr(self.backbone, 'conv1') else x
            x = self.backbone.bn1(x) if hasattr(self.backbone, 'bn1') else x
            x = self.backbone.relu(x) if hasattr(self.backbone, 'relu') else x
            x = self.backbone.maxpool(x) if hasattr(self.backbone, 'maxpool') else x
            
            x = self.backbone.layer1(x) if hasattr(self.backbone, 'layer1') else x
            x = self.backbone.layer2(x) if hasattr(self.backbone, 'layer2') else x
            x = self.backbone.layer3(x) if hasattr(self.backbone, 'layer3') else x
            x = self.backbone.layer4(x) if hasattr(self.backbone, 'layer4') else x
            
            # Global average pooling and flatten
            x = F.adaptive_avg_pool3d(x, (1, 1, 1))
            features = torch.flatten(x, 1)
            
            # Pass features through the classifier
            output = self.classifier(features)
            
            return output 