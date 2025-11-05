"""
PoreD^2 Model Implementation for Pore Detection

This module implements a custom deep learning architecture optimized for pore detection
and analysis. PoreD^2 (Pore Detection and Diameter) combines CNN feature extraction
with advanced post-processing for accurate pore identification and measurement.

The model architecture is inspired by recent advances in computer vision and
specifically adapted for microscopy image analysis.

Reference:
Custom implementation based on best practices in deep learning for biomedical imaging
and pore analysis methodologies from materials science literature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import numpy as np
from typing import Tuple, Dict


class SeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution
    
    This reduces computational complexity while maintaining performance,
    making it ideal for resource-efficient pore detection.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super(SeparableConv2d, self).__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class AttentionModule(nn.Module):
    """
    Spatial Attention Module
    
    Helps the model focus on relevant pore regions while suppressing
    background noise and artifacts.
    """
    
    def __init__(self, channels: int):
        super(AttentionModule, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, channels),
            nn.Sigmoid()
        )
        
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = (avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = torch.sigmoid(self.spatial_conv(spatial_att))
        x = x * spatial_att
        
        return x


class PoreDetectionBlock(nn.Module):
    """
    Specialized block for pore feature extraction
    
    Combines separable convolutions with attention mechanisms
    for enhanced pore detection capability.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(PoreDetectionBlock, self).__init__()
        
        self.sep_conv1 = SeparableConv2d(in_channels, out_channels)
        self.sep_conv2 = SeparableConv2d(out_channels, out_channels)
        self.attention = AttentionModule(out_channels)
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) \
                       if in_channels != out_channels else nn.Identity()
        
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        identity = self.residual(x)
        
        out = self.sep_conv1(x)
        out = self.sep_conv2(out)
        out = self.attention(out)
        out = self.dropout(out)
        
        out += identity
        return F.relu(out, inplace=True)


class PoreD2Model(nn.Module):
    """
    PoreD^2: Advanced Pore Detection and Diameter Analysis Model
    
    This model combines modern CNN architectures with specialized modules
    for accurate pore detection in microscopy images. The architecture
    includes:
    
    1. Multi-scale feature extraction
    2. Attention-based feature refinement
    3. Specialized pore detection blocks
    4. Multi-task learning for detection and measurement
    """
    
    def __init__(self, n_channels: int = 1, n_classes: int = 1):
        super(PoreD2Model, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Encoder blocks with increasing feature depth
        self.encoder_blocks = nn.ModuleList([
            PoreDetectionBlock(32, 64),
            PoreDetectionBlock(64, 128),
            PoreDetectionBlock(128, 256),
            PoreDetectionBlock(256, 512)
        ])
        
        # Multi-scale feature fusion
        self.fusion_conv = nn.Conv2d(960, 256, kernel_size=1, bias=False)  # 64+128+256+512 = 960
        self.fusion_bn = nn.BatchNorm2d(256)
        
        # Decoder for segmentation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Final segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, n_classes, kernel_size=1)
        )
        
        # Global pooling for pore statistics
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.pore_stats_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)  # avg_pore_size, max_pore_size, pore_count
        )
        
    def forward(self, x):
        # Store input size for proper upsampling
        input_size = x.shape[-2:]
        
        # Initial feature extraction
        x = self.stem(x)
        
        # Multi-scale feature extraction
        features = []
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Multi-scale feature fusion
        fused_features = []
        target_size = features[-1].shape[-2:]
        
        for feat in features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            fused_features.append(feat)
        
        fused = torch.cat(fused_features, dim=1)
        fused = F.relu(self.fusion_bn(self.fusion_conv(fused)), inplace=True)
        
        # Segmentation path
        seg_features = self.decoder(fused)
        
        # Upsample to original input size
        seg_features = F.interpolate(seg_features, size=input_size, mode='bilinear', align_corners=False)
        segmentation = self.segmentation_head(seg_features)
        
        # Pore statistics path
        global_features = self.global_pool(fused).flatten(1)
        pore_stats = self.pore_stats_head(global_features)
        
        return {
            'segmentation': segmentation,
            'pore_stats': pore_stats
        }
    
    def predict(self, x):
        """
        Generate predictions with timing information
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with predictions and inference time
        """
        start_time = time.time()
        
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Apply sigmoid to segmentation for binary output
            segmentation = torch.sigmoid(outputs['segmentation'])
            
            # Apply appropriate activations to pore stats
            pore_stats = outputs['pore_stats']
            pore_stats[:, :2] = F.relu(pore_stats[:, :2])  # Ensure positive sizes
            pore_stats[:, 2] = F.relu(pore_stats[:, 2])    # Ensure positive count
            
        inference_time = time.time() - start_time
        
        return {
            'segmentation': segmentation,
            'pore_stats': pore_stats,
            'inference_time': inference_time
        }


class PoreD2Loss(nn.Module):
    """
    Multi-task loss function for PoreD^2 model
    
    Combines segmentation loss with regression loss for pore statistics.
    """
    
    def __init__(self, seg_weight: float = 0.7, stats_weight: float = 0.3):
        super(PoreD2Loss, self).__init__()
        self.seg_weight = seg_weight
        self.stats_weight = stats_weight
        
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.stats_loss = nn.MSELoss()
    
    def forward(self, outputs: Dict, targets: Dict):
        """
        Calculate combined loss
        
        Args:
            outputs: Model outputs dictionary
            targets: Target values dictionary
        """
        seg_loss = self.seg_loss(outputs['segmentation'], targets['segmentation'])
        
        # Only calculate stats loss if targets are provided
        if 'pore_stats' in targets:
            stats_loss = self.stats_loss(outputs['pore_stats'], targets['pore_stats'])
            total_loss = self.seg_weight * seg_loss + self.stats_weight * stats_loss
        else:
            total_loss = seg_loss
            stats_loss = torch.tensor(0.0)
        
        return {
            'total_loss': total_loss,
            'segmentation_loss': seg_loss,
            'stats_loss': stats_loss
        }


def create_pored2_model(device: torch.device, pretrained_path: str = None) -> PoreD2Model:
    """
    Create and initialize a PoreD^2 model
    
    Args:
        device: Device to place the model on
        pretrained_path: Path to pretrained weights (optional)
        
    Returns:
        Initialized PoreD^2 model
    """
    model = PoreD2Model(n_channels=1, n_classes=1)
    
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    return model


def save_pored2_checkpoint(model: PoreD2Model, optimizer, epoch: int, loss: float,
                          save_path: str, image_type: str):
    """
    Save PoreD^2 model checkpoint
    
    Args:
        model: PoreD^2 model to save
        optimizer: Optimizer state
        epoch: Current training epoch
        loss: Current loss value
        save_path: Directory to save checkpoint
        image_type: Type of imaging data
    """
    os.makedirs(save_path, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'image_type': image_type,
        'model_architecture': 'PoreD^2'
    }
    
    filename = f"pored2_{image_type.lower()}_epoch_{epoch}.pth"
    filepath = os.path.join(save_path, filename)
    
    torch.save(checkpoint, filepath)
    print(f"PoreD^2 checkpoint saved: {filepath}")


def count_parameters(model: PoreD2Model) -> int:
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the PoreD^2 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_pored2_model(device)
    print(f"PoreD^2 model created with {count_parameters(model):,} trainable parameters")
    
    # Test forward pass
    test_input = torch.randn(1, 1, 512, 512).to(device)
    
    start_time = time.time()
    outputs = model(test_input)
    inference_time = time.time() - start_time
    
    print(f"Test forward pass completed:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Segmentation output shape: {outputs['segmentation'].shape}")
    print(f"  Pore stats output shape: {outputs['pore_stats'].shape}")
    print(f"  Inference time: {inference_time:.4f} seconds")
    
    # Test prediction method
    predictions = model.predict(test_input)
    print(f"  Prediction time: {predictions['inference_time']:.4f} seconds")
