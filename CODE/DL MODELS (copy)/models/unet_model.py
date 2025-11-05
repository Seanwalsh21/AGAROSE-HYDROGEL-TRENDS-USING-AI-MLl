"""
U-Net Model Implementation for Pore Detection

This module implements a U-Net architecture optimized for pore detection in microscopy images.
The U-Net model is particularly well-suited for semantic segmentation tasks in biomedical imaging.

Reference:
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks 
for biomedical image segmentation. In International Conference on Medical image 
computing and computer-assisted intervention (pp. 234-241). Springer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from typing import Tuple


class DoubleConv(nn.Module):
    """
    Double Convolution Block for U-Net
    
    This block performs two consecutive convolutions with batch normalization
    and ReLU activation, which is the standard building block of U-Net.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling Block with MaxPool followed by DoubleConv
    
    This reduces the spatial dimensions while increasing the feature depth,
    allowing the network to capture both local and global features.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling Block with ConvTranspose2d followed by DoubleConv
    
    This increases the spatial dimensions while reducing feature depth,
    and combines features from the encoder path via skip connections.
    """
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(Up, self).__init__()

        if bilinear:
            # Use bilinear interpolation for upsampling
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # Use transposed convolution for upsampling
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle potential size mismatches due to padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output Convolution Block
    
    Final 1x1 convolution to produce the desired number of output channels.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net Architecture for Pore Detection
    
    This implementation follows the original U-Net paper with modifications
    for single-channel input (grayscale microscopy images) and binary
    segmentation output (pore vs background).
    
    Architecture:
    - Encoder: 4 downsampling blocks with skip connections
    - Decoder: 4 upsampling blocks with skip connections
    - Output: Binary segmentation mask
    """
    
    def __init__(self, n_channels: int = 1, n_classes: int = 1, bilinear: bool = True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder path
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder path with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits

    def predict(self, x):
        """
        Generate predictions with timing information
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predictions, inference_time)
        """
        start_time = time.time()
        
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.sigmoid(logits)
            
        inference_time = time.time() - start_time
        
        return predictions, inference_time


class UNetLoss(nn.Module):
    """
    Combined loss function for U-Net training
    
    Combines Binary Cross Entropy with Dice Loss for better
    segmentation performance on imbalanced datasets.
    """
    
    def __init__(self, alpha: float = 0.7):
        super(UNetLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred, target, smooth=1):
        """Calculate Dice Loss"""
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


def create_unet_model(device: torch.device, pretrained_path: str = None) -> UNet:
    """
    Create and initialize a U-Net model
    
    Args:
        device: Device to place the model on
        pretrained_path: Path to pretrained weights (optional)
        
    Returns:
        Initialized U-Net model
    """
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    return model


def save_model_checkpoint(model: UNet, optimizer, epoch: int, loss: float, 
                         save_path: str, image_type: str):
    """
    Save model checkpoint with training information
    
    Args:
        model: U-Net model to save
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
        'model_architecture': 'U-Net'
    }
    
    filename = f"unet_{image_type.lower()}_epoch_{epoch}.pth"
    filepath = os.path.join(save_path, filename)
    
    torch.save(checkpoint, filepath)
    print(f"Model checkpoint saved: {filepath}")


def count_parameters(model: UNet) -> int:
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the U-Net model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_unet_model(device)
    print(f"U-Net model created with {count_parameters(model):,} trainable parameters")
    
    # Test forward pass
    test_input = torch.randn(1, 1, 512, 512).to(device)
    
    start_time = time.time()
    output = model(test_input)
    inference_time = time.time() - start_time
    
    print(f"Test forward pass completed:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Inference time: {inference_time:.4f} seconds")
    
    # Test prediction method
    predictions, pred_time = model.predict(test_input)
    print(f"  Prediction time: {pred_time:.4f} seconds")
