"""
Comprehensive Inference Script for Pore Detection Models

This script provides inference capabilities for all three deep learning models:
1. U-Net - Semantic segmentation approach
2. PoreD^2 - Advanced multi-task learning approach  
3. YOLO - Object detection approach

The script generates binary TIF images with pores in black and background in white,
records timing information, and provides detailed explanations of each step.

Usage:
python inference.py --model_type unet --image_type AFM --input_image path/to/image.tif
python inference.py --model_type pored2 --image_type CONFOCAL --input_image path/to/image.tif
python inference.py --model_type yolo --image_type CRYO-SEM --input_image path/to/image.tif
"""

import torch
import cv2
import numpy as np
import time
import os
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image

# Import our model modules
import sys
sys.path.append('.')
from models.unet_model import UNet, create_unet_model
from models.pored2_model import PoreD2Model, create_pored2_model
from models.yolo_model import PoreYOLOModel, create_yolo_model
from data_utils import PoreDetectionDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PoreInferenceEngine:
    """
    Comprehensive inference engine for pore detection models
    
    This class provides a unified interface for running inference with
    U-Net, PoreD^2, and YOLO models on microscopy images.
    """
    
    def __init__(self, model_type: str, image_type: str, device: str = None):
        """
        Initialize the inference engine
        
        Args:
            model_type: Type of model ('unet', 'pored2', 'yolo')
            image_type: Type of imaging ('AFM', 'CONFOCAL', 'CRYO-SEM', 'STED')
            device: Device to use for inference
        """
        
        self.model_type = model_type.lower()
        self.image_type = image_type.upper()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Initializing {self.model_type.upper()} inference engine")
        print(f"Target imaging type: {self.image_type}")
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
        # Setup preprocessing
        self._setup_preprocessing()
        
        # Initialize timing dictionary
        self.timing_info = {}
        
    def _load_model(self):
        """Load the appropriate model based on model_type"""
        print(f"Loading {self.model_type.upper()} model...")
        
        if self.model_type == 'unet':
            model = create_unet_model(self.device)
            # Try to load trained weights
            checkpoint_path = f"checkpoints/unet_{self.image_type.lower()}"
            self._load_checkpoint(model, checkpoint_path, 'unet')
            
        elif self.model_type == 'pored2':
            model = create_pored2_model(self.device)
            # Try to load trained weights
            checkpoint_path = f"checkpoints/pored2_{self.image_type.lower()}"
            self._load_checkpoint(model, checkpoint_path, 'pored2')
            
        elif self.model_type == 'yolo':
            model = create_yolo_model(model_size='n', pretrained=True)
            # Try to load trained weights
            self.script_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(self.script_dir)))
            self.projectt_root = os.path.join(self.project_root,'CODE','DL MODELS (copy)')
            model_path = os.path.join(self.projectt_root,"yolo_training_cryo-sem_real","runs","pore_yolo_cryo-sem_real","weights","best.pt")
            if os.path.exists(model_path):
                model.load_trained_model(model_path)
                print(f"Loaded trained YOLO weights from {model_path}")
            else:
                print(f"Using pretrained YOLO weights (trained weights not found)")
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"{self.model_type.upper()} model loaded successfully")
        return model
    
    def _load_checkpoint(self, model, checkpoint_dir: str, model_name: str):
        """Load model checkpoint if available"""
        if not os.path.exists(checkpoint_dir):
            print(f"No checkpoint directory found: {checkpoint_dir}")
            print(f"Using randomly initialized {model_name.upper()} weights")
            return
        
        # Find the latest checkpoint
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        
        if not checkpoint_files:
            print(f"No checkpoint files found in {checkpoint_dir}")
            print(f"Using randomly initialized {model_name.upper()} weights")
            return
        
        # Sort by epoch number and get the latest
        checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"loaded trained {model_name.upper()} weights from {latest_checkpoint}")
        except Exception as e:
            print(f"failed to load checkpoint {latest_checkpoint}: {e}")
            print(f"Using randomly initialized {model_name.upper()} weights")
    
    def _setup_preprocessing(self):
        """Setup image preprocessing transforms"""
        if self.model_type in ['unet', 'pored2']:
            self.transform = A.Compose([
                A.Resize(512, 512),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2()
            ])
        else:  # YOLO
            # YOLO handles preprocessing internally
            self.transform = None
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray, Dict]:
        """
        Preprocess input image for inference
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (processed_tensor, original_image, timing_info)
        """
        print(f"\nStep 1: Image Preprocessing")
        print(f"Loading image from: {image_path}")
        
        start_time = time.time()
        
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"image loaded: {original_image.shape} pixels")
        
        if self.model_type in ['unet', 'pored2']:
            # Apply preprocessing transforms
            if self.transform:
                transformed = self.transform(image=original_image)
                processed_tensor = transformed['image']
            else:
                # Manual preprocessing
                resized = cv2.resize(original_image, (512, 512))
                normalized = resized.astype(np.float32) / 255.0
                processed_tensor = torch.from_numpy(normalized).unsqueeze(0)
            
            # Add batch dimension
            if len(processed_tensor.shape) == 2:
                processed_tensor = processed_tensor.unsqueeze(0)
            processed_tensor = processed_tensor.unsqueeze(0).to(self.device)
            
            print(f"preprocessed tensor shape: {processed_tensor.shape}")
            
        else:  # YOLO
            # YOLO uses the original image path
            # Create a temporary RGB version for YOLO
            rgb_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
            temp_rgb_path = image_path.replace('.tif', '_rgb_temp.tif')
            cv2.imwrite(temp_rgb_path, rgb_image)
            processed_tensor = temp_rgb_path
            print(f"image path prepared for YOLO: {image_path}")
        
        preprocessing_time = time.time() - start_time
        
        timing_info = {
            'preprocessing_time': preprocessing_time,
            'image_load_time': preprocessing_time  # For now, combined
        }
        
        print(f"preprocessing completed in {preprocessing_time:.4f} seconds")
        
        return processed_tensor, original_image, timing_info
    
    def run_inference(self, processed_input, original_shape: Tuple[int, int]) -> Tuple[np.ndarray, Dict]:
        """
        Run model inference
        
        Args:
            processed_input: Preprocessed input (tensor or path for YOLO)
            original_shape: Original image shape for result resizing
            
        Returns:
            Tuple of (binary_mask, inference_timing)
        """
        print(f"\nStep 2: {self.model_type.upper()} Inference")
        print(f"Running {self.model_type.upper()} model on {self.image_type} image...")
        
        start_time = time.time()
        
        if self.model_type == 'unet':
            binary_mask, inference_time = self._run_unet_inference(processed_input, original_shape)
            
        elif self.model_type == 'pored2':
            binary_mask, inference_time = self._run_pored2_inference(processed_input, original_shape)
            
        elif self.model_type == 'yolo':
            binary_mask, inference_time = self._run_yolo_inference(processed_input, original_shape)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        total_inference_time = time.time() - start_time
        
        timing_info = {
            'model_inference_time': inference_time,
            'total_inference_time': total_inference_time,
            'postprocessing_time': total_inference_time - inference_time
        }
        
        print(f"{self.model_type.upper()} inference completed in {total_inference_time:.4f} seconds")
        print(f"Model forward pass: {inference_time:.4f} seconds")
        print(f"Post-processing: {timing_info['postprocessing_time']:.4f} seconds")
        
        return binary_mask, timing_info
    
    def _run_unet_inference(self, input_tensor: torch.Tensor, 
                           original_shape: Tuple[int, int]) -> Tuple[np.ndarray, float]:
        """Run U-Net inference"""
        print("  - U-Net forward pass...")
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            start_time = time.time()
            
            # Forward pass
            logits = self.model(input_tensor)
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(logits)
            
            inference_time = time.time() - start_time
            
            # Convert to binary mask
            binary_predictions = (probabilities > 0.5).float()
            
            # Convert to numpy and resize to original dimensions
            mask = binary_predictions.squeeze().cpu().numpy()
            
            # Resize to original shape
            if mask.shape != original_shape:
                mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
            
            # Create binary mask: pores (1) -> black (0), background (0) -> white (255)
            binary_mask = np.where(mask > 0.5, 0, 255).astype(np.uint8)
        
        print(f"  - Generated binary mask: {binary_mask.shape}")
        print(f"  - Detected pore pixels: {np.sum(binary_mask == 0)}")
        
        return binary_mask, inference_time
    
    def _run_pored2_inference(self, input_tensor: torch.Tensor, 
                             original_shape: Tuple[int, int]) -> Tuple[np.ndarray, float]:
        """Run PoreD^2 inference"""
        print("  - PoreD^2 forward pass...")
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            start_time = time.time()
            
            # Forward pass
            outputs = self.model(input_tensor)
            
            inference_time = time.time() - start_time
            
            # Extract segmentation output
            segmentation = outputs['segmentation']
            pore_stats = outputs['pore_stats']
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(segmentation)
            
            # Convert to binary mask
            binary_predictions = (probabilities > 0.5).float()
            
            # Convert to numpy and resize to original dimensions
            mask = binary_predictions.squeeze().cpu().numpy()
            
            # Resize to original shape
            if mask.shape != original_shape:
                mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
            
            # Create binary mask: pores (1) -> black (0), background (0) -> white (255)
            binary_mask = np.where(mask > 0.5, 0, 255).astype(np.uint8)
            
            # Extract pore statistics
            stats = pore_stats.squeeze().cpu().numpy()
            print(f"estimated pore statistics:")
            print(f"    1.Average pore area (normalized): {stats[0]:.4f}")
            print(f"    2.Maximum pore area (normalized): {stats[1]:.4f}")
            print(f"    3.Pore count (normalized): {stats[2]:.4f}")
        
        print(f"  Generated binary mask: {binary_mask.shape}")
        print(f"  Detected pore pixels: {np.sum(binary_mask == 0)}")
        
        return binary_mask, inference_time
    
    def _run_yolo_inference(self, image_path: str, 
                           original_shape: Tuple[int, int]) -> Tuple[np.ndarray, float]:
        """Run YOLO inference"""
        print("  YOLO detection and segmentation...")
        
        # Run YOLO prediction
        results = self.model.predict(image_path, save_results=False)
        
        inference_time = results['inference_time']
        
        # Extract binary mask
        if results['binary_mask'] is not None:
            binary_mask = results['binary_mask']
            
            # Resize to original shape if needed
            if binary_mask.shape != original_shape:
                binary_mask = cv2.resize(binary_mask, (original_shape[1], original_shape[0]))
        else:
            # If no detections, create white background
            binary_mask = np.ones(original_shape, dtype=np.uint8) * 255
            print("  - No pores detected by YOLO")
        
        # Print detection statistics
        detection_count = 0
        if results['predictions']:
            for pred in results['predictions']:
                detection_count += pred.get('mask_count', 0)
        
        print(f"  - YOLO detections: {detection_count} pores")
        print(f"  - Generated binary mask: {binary_mask.shape}")
        print(f"  - Detected pore pixels: {np.sum(binary_mask == 0)}")
        
        return binary_mask, inference_time
    
    def save_results(self, binary_mask: np.ndarray, output_path: str, 
                    timing_info: Dict, original_image: np.ndarray, 
                    input_image_path: str):
        """
        Save inference results including binary mask and timing information
        
        Args:
            binary_mask: Generated binary mask
            output_path: Base output path
            timing_info: Timing information dictionary
            original_image: Original input image
            input_image_path: Path to input image
        """
        print(f"\nStep 3: Saving Results")
        
        start_time = time.time()
        
        # Create output directory
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save binary mask as TIF
        mask_filename = f"{os.path.splitext(os.path.basename(output_path))[0]}_binary_mask.tif"
        mask_path = os.path.join(output_dir, mask_filename)
        
        cv2.imwrite(mask_path, binary_mask)
        print(f"✓ Binary mask saved: {mask_path}")
        
        # Create comparison visualization
        self._create_comparison_plot(original_image, binary_mask, output_dir, 
                                   os.path.basename(input_image_path))
        
        # Save timing information
        timing_filename = f"{os.path.splitext(os.path.basename(output_path))[0]}_timing.txt"
        timing_path = os.path.join(output_dir, timing_filename)
        
        self._save_timing_info(timing_info, timing_path, input_image_path)
        
        # Save detailed analysis report
        report_filename = f"{os.path.splitext(os.path.basename(output_path))[0]}_analysis.txt"
        report_path = os.path.join(output_dir, report_filename)
        
        self._save_analysis_report(binary_mask, report_path, timing_info)
        
        save_time = time.time() - start_time
        print(f"✓ Results saved in {save_time:.4f} seconds")
        
        return {
            'binary_mask_path': mask_path,
            'timing_report_path': timing_path,
            'analysis_report_path': report_path,
            'save_time': save_time
        }
    
    def _create_comparison_plot(self, original_image: np.ndarray, binary_mask: np.ndarray,
                               output_dir: str, input_filename: str):
        """Create side-by-side comparison plot"""
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title(f'Original {self.image_type} Image')
        plt.axis('off')
        
        # Binary mask
        plt.subplot(1, 3, 2)
        plt.imshow(binary_mask, cmap='gray')
        plt.title(f'{self.model_type.upper()} Binary Mask\n(Pores: Black, Background: White)')
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        # Create colored overlay
        overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        pore_pixels = binary_mask == 0
        overlay[pore_pixels] = [255, 0, 0]  # Red pores
        
        plt.imshow(overlay)
        plt.title(f'Overlay (Pores in Red)')
        plt.axis('off')
        
        plt.tight_layout()
        
        plot_filename = f"{os.path.splitext(input_filename)[0]}_comparison.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison plot saved: {plot_path}")
    
    def _save_timing_info(self, timing_info: Dict, output_path: str, input_path: str):
        """Save detailed timing information"""
        with open(output_path, 'w') as f:
            f.write("Pore Detection Timing Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model Type: {self.model_type.upper()}\n")
            f.write(f"Image Type: {self.image_type}\n")
            f.write(f"Input Image: {input_path}\n")
            f.write(f"Analysis Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Timing Breakdown:\n")
            f.write("-" * 30 + "\n")
            
            total_time = 0
            for step, times in timing_info.items():
                if isinstance(times, dict):
                    f.write(f"\n{step.replace('_', ' ').title()}:\n")
                    for substep, substep_time in times.items():
                        f.write(f"  {substep.replace('_', ' ').title()}: {substep_time:.6f} seconds\n")
                        total_time += substep_time
                else:
                    f.write(f"{step.replace('_', ' ').title()}: {times:.6f} seconds\n")
                    total_time += times
            
            f.write(f"\nTotal Processing Time: {total_time:.6f} seconds\n")
            
            # Performance analysis
            f.write(f"\nPerformance Analysis:\n")
            f.write("-" * 30 + "\n")
            
            if 'inference' in timing_info and 'model_inference_time' in timing_info['inference']:
                inference_time = timing_info['inference']['model_inference_time']
                f.write(f"Model inference rate: {1/inference_time:.2f} images/second\n")
                f.write(f"Time per megapixel: {inference_time/(1024*1024):.8f} seconds\n")
        
        print(f"✓ Timing analysis saved: {output_path}")
    
    def _save_analysis_report(self, binary_mask: np.ndarray, output_path: str, timing_info: Dict):
        """Save detailed pore analysis report"""
        # Calculate pore statistics
        pore_pixels = np.sum(binary_mask == 0)
        total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
        pore_percentage = (pore_pixels / total_pixels) * 100
        
        # Find connected components (individual pores)
        # Invert mask for connected components (pores should be 255)
        pore_mask = 255 - binary_mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pore_mask, connectivity=8)
        
        # Remove background (label 0)
        pore_areas = stats[1:, cv2.CC_STAT_AREA]
        
        with open(output_path, 'w') as f:
            f.write("Pore Detection Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model: {self.model_type.upper()}\n")
            f.write(f"Image Type: {self.image_type}\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Image Information:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Image Dimensions: {binary_mask.shape[1]} x {binary_mask.shape[0]} pixels\n")
            f.write(f"Total Pixels: {total_pixels:,}\n\n")
            
            f.write("Pore Detection Results:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Pore Pixels Detected: {pore_pixels:,}\n")
            f.write(f"Pore Coverage: {pore_percentage:.2f}%\n")
            f.write(f"Individual Pores Detected: {len(pore_areas)}\n\n")
            
            if len(pore_areas) > 0:
                f.write("Pore Size Statistics:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Average Pore Area: {np.mean(pore_areas):.2f} pixels\n")
                f.write(f"Median Pore Area: {np.median(pore_areas):.2f} pixels\n")
                f.write(f"Largest Pore Area: {np.max(pore_areas)} pixels\n")
                f.write(f"Smallest Pore Area: {np.min(pore_areas)} pixels\n")
                f.write(f"Standard Deviation: {np.std(pore_areas):.2f} pixels\n\n")
                
                # Pore size distribution
                small_pores = np.sum(pore_areas < 50)
                medium_pores = np.sum((pore_areas >= 50) & (pore_areas < 200))
                large_pores = np.sum(pore_areas >= 200)
                
                f.write("Pore Size Distribution:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Small Pores (<50 pixels): {small_pores} ({small_pores/len(pore_areas)*100:.1f}%)\n")
                f.write(f"Medium Pores (50-200 pixels): {medium_pores} ({medium_pores/len(pore_areas)*100:.1f}%)\n")
                f.write(f"Large Pores (>200 pixels): {large_pores} ({large_pores/len(pore_areas)*100:.1f}%)\n\n")
            
            # Model-specific notes
            f.write("Model-Specific Information:\n")
            f.write("-" * 30 + "\n")
            
            if self.model_type == 'unet':
                f.write("U-Net Model:\n")
                f.write("Semantic segmentation approach\n")
                f.write("Pixel-wise classification of pore vs background\n")
                f.write("Good for detecting pore boundaries\n")
                f.write("Based on Ronneberger et al. (2015)\n\n")
                
            elif self.model_type == 'pored2':
                f.write("PoreD² Model:\n")
                f.write("Advanced multi-task learning approach\n")
                f.write("Combines segmentation with pore statistics\n")
                f.write("Attention mechanisms for improved accuracy\n")
                f.write("Custom architecture for pore analysis\n\n")
                
            elif self.model_type == 'yolo':
                f.write("YOLO Model:\n")
                f.write("Object detection approach adapted for pores\n")
                f.write("Detects individual pores as objects\n")
                f.write("Provides bounding boxes and segmentation masks\n")
                f.write("Based on Ultralytics YOLOv8\n\n")
            
            # Processing notes
            f.write("Processing Notes:\n")
            f.write("binary mask format: Pores (black/0), Background (white/255)\n")
            f.write("connected components analysis used for individual pore counting\n")
            f.write("statistics calculated on detected pore regions\n")
            f.write("results depend on model training and image quality\n")
        
        print(f"Analysis report saved: {output_path}")
    
    def process_image(self, input_image_path: str, output_path: str = None) -> Dict:
        """
        Complete image processing pipeline
        
        Args:
            input_image_path: Path to input image
            output_path: Output path for results
            
        Returns:
            Dictionary with all results and timing information
        """

        print(f"PORE DETECTION ANALYSIS - {self.model_type.upper()} Model")
        print(f"Input Image: {input_image_path}")
        print(f"Image Type: {self.image_type}")
        print(f"Model: {self.model_type.upper()}")
        
        total_start_time = time.time()
        
        # Set default output path
        if output_path is None:
            input_name = os.path.splitext(os.path.basename(input_image_path))[0]
            output_path = f"results/{self.model_type}_{self.image_type.lower()}_{input_name}"
        
        try:
            # Step 1: Preprocessing
            processed_input, original_image, preprocessing_timing = self.preprocess_image(input_image_path)
            
            # Step 2: Inference
            binary_mask, inference_timing = self.run_inference(processed_input, original_image.shape)
            
            # Step 3: Save results
            save_results = self.save_results(binary_mask, output_path, 
                                           {'preprocessing': preprocessing_timing, 
                                            'inference': inference_timing},
                                           original_image, input_image_path)
            
            total_time = time.time() - total_start_time
            
            # Compile final results
            results = {
                'success': True,
                'model_type': self.model_type,
                'image_type': self.image_type,
                'input_path': input_image_path,
                'binary_mask': binary_mask,
                'timing': {
                    'preprocessing': preprocessing_timing,
                    'inference': inference_timing,
                    'saving': {'save_time': save_results['save_time']},
                    'total_time': total_time
                },
                'output_files': save_results,
                'pore_statistics': {
                    'pore_pixels': np.sum(binary_mask == 0),
                    'total_pixels': binary_mask.shape[0] * binary_mask.shape[1],
                    'pore_percentage': (np.sum(binary_mask == 0) / (binary_mask.shape[0] * binary_mask.shape[1])) * 100
                }
            }
            

            print("ANALYSIS COMPLETED SUCCESSFULLY")
            print(f"Total Processing Time: {total_time:.4f} seconds")
            print(f"Pore Coverage: {results['pore_statistics']['pore_percentage']:.2f}%")
            print(f"Binary Mask Saved: {save_results['binary_mask_path']}")
            
            return results
            
        except Exception as e:
            print(f"error during processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_type': self.model_type,
                'image_type': self.image_type,
                'input_path': input_image_path
            }


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Pore Detection Inference')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['unet', 'pored2', 'yolo'],
                       help='Type of model to use for inference')
    parser.add_argument('--image_type', type=str, required=True,
                       choices=['AFM', 'CONFOCAL', 'CRYO-SEM', 'STED'],
                       help='Type of imaging data')
    parser.add_argument('--input_image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Output path for results (optional)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create inference engine
    engine = PoreInferenceEngine(
        model_type=args.model_type,
        image_type=args.image_type,
        device=args.device
    )
    
    # Process image
    results = engine.process_image(args.input_image, args.output_path)
    
    # Print summary
    if results['success']:
        print(f"inference completed successfully!")
        print(f"Model: {results['model_type'].upper()}")
        print(f"Total time: {results['timing']['total_time']:.4f} seconds")
        print(f"Results saved to: {results['output_files']['binary_mask_path']}")
    else:
        print(f"inference failed: {results['error']}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
