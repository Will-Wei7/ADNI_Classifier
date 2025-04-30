import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

class ADNIViTModel(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, frozen_layers=0):
        """
        Initialize the ViT model for ADNI dataset classification
        
        Args:
            num_classes (int): Number of output classes (default: 3 for AD/CN/MCI)
            pretrained (bool): Whether to use pretrained weights (default: True)
            frozen_layers (int): Number of layers to freeze (default: 0)
        """
        super(ADNIViTModel, self).__init__()
        
        if pretrained:
            # Load pretrained ViT model
            print("Loading pre-trained ViT model...")
            self.vit = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224', 
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            # Initialize with default configuration
            print("Initializing ViT model with random weights...")
            config = ViTConfig(
                image_size=224,
                patch_size=16,
                num_channels=1,  # Grayscale images
                num_labels=num_classes,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
            )
            self.vit = ViTForImageClassification(config)
        
        # Freeze layers if specified
        if frozen_layers > 0:
            print(f"Freezing first {frozen_layers} encoder layers...")
            for i, layer in enumerate(self.vit.vit.encoder.layer[:frozen_layers]):
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, x, labels=None):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            labels (torch.Tensor, optional): Ground truth labels
            
        Returns:
            tuple: (loss, logits) if labels provided, logits otherwise
        """
        # ViT expects [B, C, H, W] but transformers expect [B, H, W, C]
        # x = x.permute(0, 2, 3, 1)
        
        if labels is not None:
            outputs = self.vit(pixel_values=x, labels=labels)
            return outputs.loss, outputs.logits
        else:
            outputs = self.vit(pixel_values=x)
            return outputs.logits

def get_model(model_config):
    """
    Factory function to create model instance
    
    Args:
        model_config (dict): Configuration dictionary for model
        
    Returns:
        ADNIViTModel: Model instance
    """
    return ADNIViTModel(
        num_classes=model_config.get('num_classes', 3),
        pretrained=model_config.get('pretrained', True),
        frozen_layers=model_config.get('frozen_layers', 0)
    ) 