"""
Comprehensive Pore Detection System for Multiple Imaging Modalities

This module provides utilities for loading and preprocessing different types of imaging data
(AFM, Confocal, Cryo-SEM, STED) for pore detection using deep learning models.

References:
- U-Net: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation.
- PoreD^2: Modified from existing implementations for pore detection
- YOLO: Ultralytics YOLO implementation adapted for pore detection
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List, Dict
import glob
from pathlib import Path


class PoreDetectionDataset(Dataset):
    """
    Dataset class for loading and preprocessing pore detection images
    
    Supports multiple imaging modalities:
    - AFM (Atomic Force Microscopy)
    - Confocal Microscopy
    - Cryo-SEM (Scanning Electron Microscopy)
    - STED (Stimulated Emission Depletion)
    """
    
    def __init__(self, 
                 data_dir: str, 
                 image_type: str, 
                 is_training: bool = True,
                 transform=None,
                 image_size: Tuple[int, int] = (512, 512)):
        """
        Initialize the dataset
        
        Args:
            data_dir: Root directory containing the dataset
            image_type: Type of imaging ('AFM', 'CONFOCAL', 'CRYO-SEM', 'STED')
            is_training: Whether to load training or test data
            transform: Data augmentation transforms
            image_size: Target image size for resizing
        """
        self.data_dir = data_dir
        self.image_type = image_type.upper()
        self.is_training = is_training
        self.image_size = image_size
        
        # Map image types to their folder names
        self.folder_mapping = {
            'AFM': 'AFM folder',
            'CONFOCAL': 'Confocal folder', 
            'CRYO-SEM': 'Cryo-sem Folder',
            'STED': 'STED Folder'
        }
        
        # Set up transforms
        if transform is None:
            if is_training:
                self.transform = A.Compose([
                    A.Resize(image_size[0], image_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.GaussianBlur(blur_limit=3, p=0.2),
                    A.Normalize(mean=[0.485], std=[0.229]),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(image_size[0], image_size[1]),
                    A.Normalize(mean=[0.485], std=[0.229]),
                    ToTensorV2()
                ])
        else:
            self.transform = transform
            
        self.image_paths = self._load_image_paths()
        
    def _load_image_paths(self) -> List[str]:
        """Load all image paths for the specified imaging type"""
        folder_name = self.folder_mapping[self.image_type]
        
        if self.is_training:
            # Load training data (augmented images)
            if self.image_type == 'AFM':
                base_path = os.path.join(self.data_dir, folder_name, 'AFM Training')
            elif self.image_type == 'CONFOCAL':
                base_path = os.path.join(self.data_dir, folder_name, 'CONFOCAL Training')
            elif self.image_type == 'CRYO-SEM':
                base_path = os.path.join(self.data_dir, folder_name, 'CRYO-SEM_Training')
            else:  # STED
                base_path = os.path.join(self.data_dir, folder_name, 'STED Training')
        else:
            # Load test data (original images)
            if self.image_type == 'AFM':
                base_path = os.path.join(self.data_dir, folder_name, 'AFM')
            elif self.image_type == 'CONFOCAL':
                base_path = os.path.join(self.data_dir, folder_name, 'CONFOCAL')
            elif self.image_type == 'CRYO-SEM':
                base_path = os.path.join(self.data_dir, folder_name, 'CRYO-SEM')
            else:  # STED
                base_path = os.path.join(self.data_dir, folder_name, 'STED')
        
        # Recursively find all .tif files
        image_paths = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
                    image_paths.append(os.path.join(root, file))
        
        return sorted(image_paths)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single item from the dataset
        
        Returns:
            Dictionary containing:
            - image: Preprocessed image tensor
            - path: Original image path
            - filename: Image filename
        """
        img_path = self.image_paths[idx]
        
        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Ensure image is float tensor with correct dimensions
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # Add channel dimension
        
        return {
            'image': image.float(),
            'path': img_path,
            'filename': os.path.basename(img_path)
        }


def create_dataloaders(data_dir: str, 
                      image_type: str, 
                      batch_size: int = 8,
                      image_size: Tuple[int, int] = (512, 512),
                      num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and testing dataloaders for a specific imaging type
    
    Args:
        data_dir: Root directory containing the dataset
        image_type: Type of imaging ('AFM', 'CONFOCAL', 'CRYO-SEM', 'STED')
        batch_size: Batch size for dataloaders
        image_size: Target image size
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    
    # Create datasets
    train_dataset = PoreDetectionDataset(
        data_dir=data_dir,
        image_type=image_type,
        is_training=True,
        image_size=image_size
    )
    
    test_dataset = PoreDetectionDataset(
        data_dir=data_dir,
        image_type=image_type,
        is_training=False,
        image_size=image_size
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Created dataloaders for {image_type}:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Testing samples: {len(test_dataset)}")
    
    return train_dataloader, test_dataloader


def get_image_statistics(data_dir: str, image_type: str) -> Dict:
    """
    Calculate mean and std statistics for normalization
    
    Args:
        data_dir: Root directory containing the dataset
        image_type: Type of imaging
        
    Returns:
        Dictionary with mean and std values
    """
    dataset = PoreDetectionDataset(
        data_dir=data_dir,
        image_type=image_type,
        is_training=False,
        transform=A.Compose([
            A.Resize(512, 512),
            ToTensorV2()
        ])
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    mean = 0.0
    std = 0.0
    total_samples = 0
    
    for batch in dataloader:
        images = batch['image']
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return {'mean': mean.item(), 'std': std.item()}


if __name__ == "__main__":
    # Test the dataset loading
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    data_dir = os.path.join(project_root,'CODE','DL MODELS (copy)','Dataset')
    #data_dir = "C:\Users\walsh\Downloads\DL MODELS (copy)\Dataset"
    
    for image_type in ['AFM', 'CONFOCAL', 'CRYO-SEM', 'STED']:
        print(f"\nTesting {image_type} dataset loading...")
        try:
            train_loader, test_loader = create_dataloaders(
                data_dir=data_dir,
                image_type=image_type,
                batch_size=4
            )
            
            # Test loading a batch
            for batch in train_loader:
                print(f"  Batch shape: {batch['image'].shape}")
                print(f"  Sample filename: {batch['filename'][0]}")
                break
                
        except Exception as e:
            print(f"  Error loading {image_type}: {e}")
