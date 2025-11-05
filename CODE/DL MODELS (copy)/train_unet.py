"""
Training Script for U-Net Pore Detection

This script provides comprehensive training functionality for U-Net models
across different imaging modalities (AFM, Confocal, Cryo-SEM, STED).

The training process includes:
1. Data loading and preprocessing
2. Model initialization and training
3. Performance monitoring and validation
4. Model checkpointing and saving
5. Timing analysis for each step

Reference:
U-Net implementation based on Ronneberger et al. (2015) with optimizations
for pore detection in microscopy images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# Import our custom modules
import sys
sys.path.append('.')
from data_utils import create_dataloaders, PoreDetectionDataset
from models.unet_model import UNet, UNetLoss, create_unet_model, save_model_checkpoint


class UNetTrainer:
    """
    Comprehensive trainer for U-Net pore detection models
    
    This class handles the complete training pipeline including:
    - Data loading and augmentation
    - Model training and validation
    - Loss tracking and visualization
    - Model checkpointing
    - Performance metrics calculation
    """
    
    def __init__(self, 
                 data_dir: str,
                 image_type: str,
                 batch_size: int = 8,
                 learning_rate: float = 1e-4,
                 num_epochs: int = 100,
                 image_size: Tuple[int, int] = (512, 512),
                 device: str = None):
        """
        Initialize the U-Net trainer
        
        Args:
            data_dir: Root directory containing the dataset
            image_type: Type of imaging ('AFM', 'CONFOCAL', 'CRYO-SEM', 'STED')
            batch_size: Training batch size
            learning_rate: Initial learning rate
            num_epochs: Number of training epochs
            image_size: Input image size
            device: Device to use for training
        """
        
        self.data_dir = data_dir
        self.image_type = image_type.upper()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.image_size = image_size
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        print(f"Training U-Net for {self.image_type} pore detection")
        
        # Initialize components
        self._setup_data_loaders()
        self._setup_model()
        self._setup_training()
        self._setup_logging()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.training_times = []
        
    def _setup_data_loaders(self):
        """Set up training and validation data loaders"""
        print("Setting up data loaders...")
        
        start_time = time.time()
        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=self.data_dir,
            image_type=self.image_type,
            batch_size=self.batch_size,
            image_size=self.image_size,
            num_workers=4
        )
        
        setup_time = time.time() - start_time
        print(f"Data loaders setup completed in {setup_time:.2f} seconds")
        
        # Create simple target masks using thresholding for unsupervised learning
        self.use_pseudo_labels = True
        
    def _setup_model(self):
        """Initialize the U-Net model"""
        print("Initializing U-Net model...")
        
        self.model = create_unet_model(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model initialized with {num_params:,} trainable parameters")
        
    def _setup_training(self):
        """Set up optimizer, loss function, and scheduler"""
        print("Setting up training components...")
        
        # Loss function
        self.criterion = UNetLoss(alpha=0.7)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
    def _setup_logging(self):
        """Set up tensorboard logging"""
        log_dir = f"logs/unet_{self.image_type.lower()}_{int(time.time())}"
        self.writer = SummaryWriter(log_dir)
        print(f"Tensorboard logs will be saved to: {log_dir}")
        
    def create_pseudo_labels(self, images: torch.Tensor) -> torch.Tensor:
        """
        Create pseudo labels using image processing techniques
        
        Args:
            images: Input image batch
            
        Returns:
            Pseudo label masks
        """
        pseudo_labels = []
        
        for img in images:
            # Convert to numpy for processing
            img_np = img.squeeze().cpu().numpy()
            
            # Normalize to 0-255
            img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
            
            # Apply adaptive thresholding to detect dark regions (pores)
            import cv2
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(img_np, (5, 5), 0)
            
            # Adaptive thresholding to detect pores
            binary = cv2.adaptiveThreshold(
                blurred, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                11, 
                2
            )
            
            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Normalize to 0-1
            binary = binary.astype(np.float32) / 255.0
            
            pseudo_labels.append(binary)
        
        # Convert back to tensor
        pseudo_labels = torch.tensor(np.array(pseudo_labels), dtype=torch.float32)
        pseudo_labels = pseudo_labels.unsqueeze(1).to(self.device)
        
        return pseudo_labels
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        epoch_start_time = time.time()
        
        with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                
                # Create pseudo labels
                if self.use_pseudo_labels:
                    targets = self.create_pseudo_labels(images)
                else:
                    # For when real labels are available
                    targets = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Avg Loss': f"{total_loss/num_batches:.4f}"
                })
                
                # Log to tensorboard
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / num_batches
        
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds, Average Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate_epoch(self, epoch: int) -> float:
        """
        Validate for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        val_start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                
                # Create pseudo labels for validation too
                if self.use_pseudo_labels:
                    targets = self.create_pseudo_labels(images)
                else:
                    targets = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        val_time = time.time() - val_start_time
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        print(f"Validation completed in {val_time:.2f} seconds, Average Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def train(self) -> Dict:
        """
        Complete training process
        
        Returns:
            Training results dictionary
        """
        print(f"\nStarting U-Net training for {self.image_type}")
        print("=" * 60)
        
        total_start_time = time.time()
        best_val_loss = float('inf')
        best_epoch = 0
        
        # Create save directory
        save_dir = f"checkpoints/unet_{self.image_type.lower()}"
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 40)
            
            # Training
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation_Epoch', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                
                save_model_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    save_path=save_dir,
                    image_type=self.image_type
                )
                
                print(f"✓ New best model saved (Val Loss: {val_loss:.4f})")
            
            # Early stopping check
            if epoch - best_epoch > 20:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        total_training_time = time.time() - total_start_time
        
        # Final results
        results = {
            'image_type': self.image_type,
            'total_training_time': total_training_time,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'final_train_loss': self.train_losses[-1],
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_save_path': save_dir
        }
        
        print(f"\nTraining completed!")
        print(f"Total time: {total_training_time:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
        
        # Save training plots
        self._save_training_plots(save_dir)
        
        # Close tensorboard writer
        self.writer.close()
        
        return results
    
    def _save_training_plots(self, save_dir: str):
        """Save training loss plots"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'U-Net Training Loss - {self.image_type}')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss Detail - {self.image_type}')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'unet_training_plots_{self.image_type.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
data_dir = os.path.join(project_root,'CODE','DL MODELS (copy)')


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train U-Net for Pore Detection')
    parser.add_argument('--data_dir', type=str, default=os.path.join(data_dir,'Dataset'),
                       help='Root directory containing the dataset')
    parser.add_argument('--image_type', type=str, required=True,
                       choices=['AFM', 'CONFOCAL', 'CRYO-SEM', 'STED'],
                       help='Type of imaging data to train on')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                       help='Input image size (height width)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = UNetTrainer(
        data_dir=args.data_dir,
        image_type=args.image_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        image_size=tuple(args.image_size)
    )
    
    # Start training
    results = trainer.train()
    
    # Print final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Image Type: {results['image_type']}")
    print(f"Training Time: {results['total_training_time']:.2f} seconds")
    print(f"Best Validation Loss: {results['best_val_loss']:.6f}")
    print(f"Model saved to: {results['model_save_path']}")


if __name__ == "__main__":
    main()
