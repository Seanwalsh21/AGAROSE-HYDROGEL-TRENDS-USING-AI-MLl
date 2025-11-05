"""
Training Script for PoreD^2 Pore Detection

This script provides comprehensive training functionality for PoreD^2 models
across different imaging modalities (AFM, Confocal, Cryo-SEM, STED).

PoreD^2 is a custom deep learning architecture that combines:
1. Multi-scale feature extraction
2. Attention mechanisms for pore localization
3. Multi-task learning for segmentation and pore statistics
4. Advanced data augmentation for microscopy images
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
import cv2

# Import our custom modules
import sys
sys.path.append('.')
from data_utils import create_dataloaders, PoreDetectionDataset
from models.pored2_model import PoreD2Model, PoreD2Loss, create_pored2_model, save_pored2_checkpoint


class PoreD2Trainer:
    """
    Comprehensive trainer for PoreD^2 pore detection models
    
    This class handles the complete training pipeline for the advanced
    PoreD^2 architecture with multi-task learning capabilities.
    """
    
    def __init__(self, 
                 data_dir: str,
                 image_type: str,
                 batch_size: int = 4,  # Smaller batch size due to model complexity
                 learning_rate: float = 5e-5,
                 num_epochs: int = 150,
                 image_size: Tuple[int, int] = (512, 512),
                 device: str = None):
        """
        Initialize the PoreD^2 trainer
        
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
        print(f"Training PoreD^2 for {self.image_type} pore detection")
        
        # Initialize components
        self._setup_data_loaders()
        self._setup_model()
        self._setup_training()
        self._setup_logging()
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.seg_losses = []
        self.stats_losses = []
        
    def _setup_data_loaders(self):
        """Set up training and validation data loaders"""
        print("Setting up data loaders...")
        
        start_time = time.time()
        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=self.data_dir,
            image_type=self.image_type,
            batch_size=self.batch_size,
            image_size=self.image_size,
            num_workers=2  # Reduced for stability
        )
        
        setup_time = time.time() - start_time
        print(f"Data loaders setup completed in {setup_time:.2f} seconds")
        
    def _setup_model(self):
        """Initialize the PoreD^2 model"""
        print("Initializing PoreD^2 model...")
        
        self.model = create_pored2_model(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"PoreD^2 model initialized with {num_params:,} trainable parameters")
        
    def _setup_training(self):
        """Set up optimizer, loss function, and scheduler"""
        print("Setting up training components...")
        
        # Multi-task loss function
        self.criterion = PoreD2Loss(seg_weight=0.8, stats_weight=0.2)
        
        # Optimizer with different learning rates for different parts
        self.optimizer = optim.AdamW([
            {'params': self.model.encoder_blocks.parameters(), 'lr': self.learning_rate * 0.5},
            {'params': self.model.decoder.parameters(), 'lr': self.learning_rate},
            {'params': self.model.segmentation_head.parameters(), 'lr': self.learning_rate},
            {'params': self.model.pore_stats_head.parameters(), 'lr': self.learning_rate * 2}
        ], weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-7
        )
        
    def _setup_logging(self):
        """Set up tensorboard logging"""
        log_dir = f"logs/pored2_{self.image_type.lower()}_{int(time.time())}"
        self.writer = SummaryWriter(log_dir)
        print(f"Tensorboard logs will be saved to: {log_dir}")
    
    def create_enhanced_pseudo_labels(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Create enhanced pseudo labels for multi-task learning
        
        Args:
            images: Input image batch
            
        Returns:
            Dictionary containing segmentation masks and pore statistics
        """
        segmentation_masks = []
        pore_statistics = []
        
        for img in images:
            # Convert to numpy for processing
            img_np = img.squeeze().cpu().numpy()
            
            # Normalize to 0-255
            img_normalized = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
            
            # Advanced pore detection pipeline
            mask, stats = self._advanced_pore_detection(img_normalized)
            
            segmentation_masks.append(mask)
            pore_statistics.append(stats)
        
        # Convert to tensors
        seg_masks = torch.tensor(np.array(segmentation_masks), dtype=torch.float32)
        seg_masks = seg_masks.unsqueeze(1).to(self.device)
        
        pore_stats = torch.tensor(np.array(pore_statistics), dtype=torch.float32).to(self.device)
        
        return {
            'segmentation': seg_masks,
            'pore_stats': pore_stats
        }
    
    def _advanced_pore_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced pore detection using multiple image processing techniques
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (binary_mask, pore_statistics)
        """
        # Apply multiple preprocessing techniques
        
        # 1. Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # 2. CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # 3. Multiple thresholding approaches
        # Otsu's thresholding
        _, binary_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Adaptive thresholding
        binary_adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Combine thresholding results
        binary_combined = cv2.bitwise_and(binary_otsu, binary_adaptive)
        
        # 4. Morphological operations for noise removal
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Remove small noise
        binary_clean = cv2.morphologyEx(binary_combined, cv2.MORPH_OPEN, kernel_small)
        
        # Fill small holes
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel_large)
        
        # 5. Extract pore statistics
        contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area to remove noise
        min_area = 10  # Minimum pore area in pixels
        max_area = image.shape[0] * image.shape[1] * 0.1  # Maximum 10% of image
        
        valid_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
        
        # Calculate pore statistics
        if valid_contours:
            areas = [cv2.contourArea(c) for c in valid_contours]
            avg_area = np.mean(areas)
            max_area = np.max(areas)
            pore_count = len(valid_contours)
        else:
            avg_area = 0.0
            max_area = 0.0
            pore_count = 0.0
        
        # Normalize mask to 0-1
        mask = binary_clean.astype(np.float32) / 255.0
        
        # Normalize statistics (these are approximate normalizations)
        avg_area_norm = min(avg_area / 1000.0, 1.0)  # Normalize by expected max area
        max_area_norm = min(max_area / 5000.0, 1.0)  # Normalize by expected max area
        count_norm = min(pore_count / 100.0, 1.0)    # Normalize by expected max count
        
        stats = np.array([avg_area_norm, max_area_norm, count_norm], dtype=np.float32)
        
        return mask, stats
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with average losses for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_stats_loss = 0.0
        num_batches = 0
        
        epoch_start_time = time.time()
        
        with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                
                # Create enhanced pseudo labels
                targets = self.create_enhanced_pseudo_labels(images)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total_loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                total_seg_loss += loss_dict['segmentation_loss'].item()
                total_stats_loss += loss_dict['stats_loss'].item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Seg': f"{loss_dict['segmentation_loss'].item():.4f}",
                    'Stats': f"{loss_dict['stats_loss'].item():.4f}"
                })
                
                # Log to tensorboard
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/Train_Total', loss.item(), global_step)
                self.writer.add_scalar('Loss/Train_Segmentation', 
                                     loss_dict['segmentation_loss'].item(), global_step)
                self.writer.add_scalar('Loss/Train_Stats', 
                                     loss_dict['stats_loss'].item(), global_step)
        
        # Update learning rate
        self.scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        
        avg_losses = {
            'total': total_loss / num_batches,
            'segmentation': total_seg_loss / num_batches,
            'stats': total_stats_loss / num_batches,
            'epoch_time': epoch_time
        }
        
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
        print(f"Avg Losses - Total: {avg_losses['total']:.4f}, "
              f"Seg: {avg_losses['segmentation']:.4f}, "
              f"Stats: {avg_losses['stats']:.4f}")
        
        return avg_losses
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with average validation losses
        """
        self.model.eval()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_stats_loss = 0.0
        num_batches = 0
        
        val_start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                
                # Create enhanced pseudo labels
                targets = self.create_enhanced_pseudo_labels(images)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss_dict = self.criterion(outputs, targets)
                
                total_loss += loss_dict['total_loss'].item()
                total_seg_loss += loss_dict['segmentation_loss'].item()
                total_stats_loss += loss_dict['stats_loss'].item()
                num_batches += 1
        
        val_time = time.time() - val_start_time
        
        avg_losses = {
            'total': total_loss / num_batches if num_batches > 0 else 0.0,
            'segmentation': total_seg_loss / num_batches if num_batches > 0 else 0.0,
            'stats': total_stats_loss / num_batches if num_batches > 0 else 0.0,
            'val_time': val_time
        }
        
        print(f"Validation completed in {val_time:.2f} seconds")
        print(f"Val Losses - Total: {avg_losses['total']:.4f}, "
              f"Seg: {avg_losses['segmentation']:.4f}, "
              f"Stats: {avg_losses['stats']:.4f}")
        
        return avg_losses
    
    def train(self) -> Dict:
        """
        Complete training process
        
        Returns:
            Training results dictionary
        """
        print(f"\nStarting PoreD^2 training for {self.image_type}")
        print("=" * 70)
        
        total_start_time = time.time()
        best_val_loss = float('inf')
        best_epoch = 0
        
        # Create save directory
        save_dir = f"checkpoints/pored2_{self.image_type.lower()}"
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            # Training
            train_losses = self.train_epoch(epoch)
            self.train_losses.append(train_losses['total'])
            self.seg_losses.append(train_losses['segmentation'])
            self.stats_losses.append(train_losses['stats'])
            
            # Validation
            val_losses = self.validate_epoch(epoch)
            self.val_losses.append(val_losses['total'])
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train_Epoch', train_losses['total'], epoch)
            self.writer.add_scalar('Loss/Validation_Epoch', val_losses['total'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                best_epoch = epoch
                
                save_pored2_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=val_losses['total'],
                    save_path=save_dir,
                    image_type=self.image_type
                )
                
                print(f"✓ New best model saved (Val Loss: {val_losses['total']:.4f})")
            
            # Early stopping check
            if epoch - best_epoch > 30:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        total_training_time = time.time() - total_start_time
        
        # Final results
        results = {
            'image_type': self.image_type,
            'model_architecture': 'PoreD^2',
            'total_training_time': total_training_time,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'final_train_loss': self.train_losses[-1],
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'seg_losses': self.seg_losses,
            'stats_losses': self.stats_losses,
            'model_save_path': save_dir
        }
        
        print(f"\nPoreD^2 Training completed!")
        print(f"Total time: {total_training_time:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
        
        # Save training plots
        self._save_training_plots(save_dir)
        
        # Close tensorboard writer
        self.writer.close()
        
        return results
    
    def _save_training_plots(self, save_dir: str):
        """Save training loss plots"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Training Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title(f'PoreD^2 Total Loss - {self.image_type}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(self.seg_losses, label='Segmentation Loss', color='blue', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Segmentation Loss')
        plt.title(f'Segmentation Loss - {self.image_type}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(self.stats_losses, label='Statistics Loss', color='red', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Statistics Loss')
        plt.title(f'Pore Statistics Loss - {self.image_type}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'pored2_training_plots_{self.image_type.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
data_dir = os.path.join(project_root,'CODE','DL MODELS (copy)')

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train PoreD^2 for Pore Detection')
    parser.add_argument('--data_dir', type=str, default=os.path.join(data_dir,'Dataset'),
                       help='Root directory containing the dataset')
    parser.add_argument('--image_type', type=str, required=True,
                       choices=['AFM', 'CONFOCAL', 'CRYO-SEM', 'STED'],
                       help='Type of imaging data to train on')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Initial learning rate')
    parser.add_argument('--num_epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                       help='Input image size (height width)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = PoreD2Trainer(
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
    print("\n" + "=" * 70)
    print("PORED^2 TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Image Type: {results['image_type']}")
    print(f"Training Time: {results['total_training_time']:.2f} seconds")
    print(f"Best Validation Loss: {results['best_val_loss']:.6f}")
    print(f"Model saved to: {results['model_save_path']}")


if __name__ == "__main__":
    main()
