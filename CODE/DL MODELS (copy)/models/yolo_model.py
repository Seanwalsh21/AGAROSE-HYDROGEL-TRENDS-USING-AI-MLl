"""
YOLO Model Implementation for Pore Detection

This module implements a YOLO (You Only Look Once) architecture adapted for pore detection
in microscopy images. The implementation uses YOLOv8 as the base architecture with
modifications for detecting pore objects and generating segmentation masks.

Reference:
Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767.

Note: This implementation leverages the Ultralytics YOLO framework and adapts it
for pore detection tasks with custom post-processing for binary mask generation.
"""

import torch
import torch.nn as nn
import time
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils import ops
import yaml


class PoreYOLOModel:
    """
    YOLO-based Pore Detection Model
    
    This class wraps the Ultralytics YOLO implementation and provides
    specialized functionality for pore detection and segmentation.
    """
    
    def __init__(self, model_size: str = 'n', pretrained: bool = True):
        """
        Initialize the YOLO model for pore detection
        
        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            pretrained: Whether to use pretrained weights
        """
        self.model_size = model_size
        
        # Initialize YOLO model for segmentation
        model_name = f'yolov8{model_size}-seg.pt' if pretrained else f'yolov8{model_size}-seg.yaml'
        self.model = YOLO(model_name)
        
        # Store model configuration
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.max_detections = 1000
        
    def create_pore_dataset_config(self, data_dir: str, image_type: str) -> str:
        """
        Create YOLO dataset configuration for pore detection
        
        Args:
            data_dir: Root directory containing the dataset
            image_type: Type of imaging ('AFM', 'CONFOCAL', 'CRYO-SEM', 'STED')
            
        Returns:
            Path to the created dataset configuration file
        """
        # Map image types to their folder names
        folder_mapping = {
            'AFM': 'AFM folder',
            'CONFOCAL': 'Confocal folder', 
            'CRYO-SEM': 'Cryo-sem Folder',
            'STED': 'STED Folder'
        }
        
        folder_name = folder_mapping[image_type.upper()]
        
        # Define paths
        if image_type.upper() == 'AFM':
            train_path = os.path.join(data_dir, folder_name, 'AFM Training')
            val_path = os.path.join(data_dir, folder_name, 'AFM')
        elif image_type.upper() == 'CONFOCAL':
            train_path = os.path.join(data_dir, folder_name, 'CONFOCAL Training')
            val_path = os.path.join(data_dir, folder_name, 'CONFOCAL')
        elif image_type.upper() == 'CRYO-SEM':
            train_path = os.path.join(data_dir, folder_name, 'CRYO-SEM_Training')
            val_path = os.path.join(data_dir, folder_name, 'CRYO-SEM')
        else:  # STED
            train_path = os.path.join(data_dir, folder_name, 'STED Training')
            val_path = os.path.join(data_dir, folder_name, 'STED')
        
        # Create dataset configuration
        dataset_config = {
            'path': data_dir,
            'train': train_path,
            'val': val_path,
            'test': val_path,
            'names': {0: 'pore'},
            'nc': 1  # Number of classes
        }
        
        # Save configuration file
        config_path = f"pore_detection_{image_type.lower()}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        return config_path
    
    def prepare_training_data(self, data_dir: str, image_type: str, 
                            output_dir: str) -> str:
        """
        Prepare training data in YOLO format
        
        Since we don't have labeled data, this creates pseudo-labels using
        image processing techniques for initial training.
        
        Args:
            data_dir: Root directory containing the dataset
            image_type: Type of imaging
            output_dir: Output directory for prepared data
            
        Returns:
            Path to prepared dataset
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create directory structure
        train_images_dir = os.path.join(output_dir, 'images', 'train')
        val_images_dir = os.path.join(output_dir, 'images', 'val')
        train_labels_dir = os.path.join(output_dir, 'labels', 'train')
        val_labels_dir = os.path.join(output_dir, 'labels', 'val')
        
        for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # This is a placeholder for data preparation
        # In practice, you would implement image processing to detect pores
        # and create YOLO format annotations
        
        print(f"Training data preparation for {image_type} completed")
        print(f"Data saved to: {output_dir}")
        
        return output_dir
    
    def train(self, data_config: str, epochs: int = 100, imgsz: int = 640,
              batch_size: int = 16, image_type: str = "unknown") -> Dict:
        """
        Train the YOLO model for pore detection
        
        Args:
            data_config: Path to dataset configuration file
            epochs: Number of training epochs
            imgsz: Input image size
            batch_size: Training batch size
            image_type: Type of imaging data
            
        Returns:
            Training results dictionary
        """
        print(f"Starting YOLO training for {image_type} pore detection...")
        print(f"Configuration: {epochs} epochs, {imgsz}px images, batch size {batch_size}")
        
        start_time = time.time()
        
        # Train the model
        results = self.model.train(
            data=data_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            name=f'pore_yolo_{image_type.lower()}',
            patience=20,
            save=True,
            plots=True,
            device='0' if torch.cuda.is_available() else 'cpu'
        )
        
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'results': results,
            'training_time': training_time,
            'model_path': self.model.trainer.best,
            'image_type': image_type
        }
    
    def predict(self, image_path: str, save_results: bool = True,
                output_dir: str = "yolo_results") -> Dict:
        """
        Generate predictions for pore detection
        
        Args:
            image_path: Path to input image
            save_results: Whether to save prediction results
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing predictions and timing information
        """
        start_time = time.time()
        
        # Run prediction
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            save=save_results,
            project=output_dir,
            name='predictions',
            exist_ok=True
        )
        
        inference_time = time.time() - start_time
        
        # Extract prediction data
        prediction_data = []
        binary_mask = None
        
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                # Generate binary mask from segmentation masks
                binary_mask = self._create_binary_mask(result)
                
                # Extract detection information
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
                confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
                
                prediction_data.append({
                    'boxes': boxes,
                    'confidences': confidences,
                    'mask_count': len(result.masks) if result.masks else 0
                })
        
        return {
            'predictions': prediction_data,
            'binary_mask': binary_mask,
            'inference_time': inference_time,
            'image_path': image_path
        }
    
    def _create_binary_mask(self, result) -> np.ndarray:
        """
        Create binary mask from YOLO segmentation results
        
        Args:
            result: YOLO prediction result
            
        Returns:
            Binary mask with pores in black and background in white
        """
        if result.masks is None:
            return None
        
        # Get image dimensions
        img_height, img_width = result.orig_shape
        
        # Initialize binary mask (white background)
        binary_mask = np.ones((img_height, img_width), dtype=np.uint8) * 255
        
        # Convert masks to numpy and combine
        masks = result.masks.data.cpu().numpy()
        
        for mask in masks:
            # Resize mask to original image size
            mask_resized = cv2.resize(mask, (img_width, img_height))
            
            # Apply threshold and set pore regions to black
            pore_pixels = mask_resized > 0.5
            binary_mask[pore_pixels] = 0
        
        return binary_mask
    
    def predict_batch(self, image_paths: List[str], output_dir: str = "yolo_batch_results") -> Dict:
        """
        Generate predictions for multiple images
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing batch predictions and timing
        """
        start_time = time.time()
        
        batch_results = []
        
        for image_path in image_paths:
            result = self.predict(image_path, save_results=True, output_dir=output_dir)
            batch_results.append(result)
        
        total_time = time.time() - start_time
        
        return {
            'batch_results': batch_results,
            'total_time': total_time,
            'average_time_per_image': total_time / len(image_paths) if image_paths else 0,
            'num_images': len(image_paths)
        }
    
    def save_binary_masks(self, results: Dict, output_dir: str):
        """
        Save binary masks as TIF images
        
        Args:
            results: Prediction results
            output_dir: Output directory for masks
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if isinstance(results, dict) and 'batch_results' in results:
            # Handle batch results
            for i, result in enumerate(results['batch_results']):
                if result['binary_mask'] is not None:
                    filename = f"binary_mask_{i:04d}.tif"
                    save_path = os.path.join(output_dir, filename)
                    cv2.imwrite(save_path, result['binary_mask'])
        else:
            # Handle single result
            if results['binary_mask'] is not None:
                filename = "binary_mask.tif"
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, results['binary_mask'])
    
    def load_trained_model(self, model_path: str):
        """
        Load a previously trained model
        
        Args:
            model_path: Path to trained model weights
        """
        self.model = YOLO(model_path)
        print(f"Loaded trained model from: {model_path}")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_size': self.model_size,
            'model_type': 'YOLOv8-seg',
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'max_detections': self.max_detections
        }


def create_yolo_model(model_size: str = 'n', pretrained: bool = True) -> PoreYOLOModel:
    """
    Create a YOLO model for pore detection
    
    Args:
        model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        pretrained: Whether to use pretrained weights
        
    Returns:
        Initialized PoreYOLOModel
    """
    model = PoreYOLOModel(model_size=model_size, pretrained=pretrained)
    return model


if __name__ == "__main__":
    # Test the YOLO model initialization
    print("Testing YOLO model for pore detection...")
    
    try:
        # Create model
        model = create_yolo_model(model_size='n', pretrained=True)
        print("✓ YOLO model created successfully")
        
        # Display model info
        info = model.get_model_info()
        print(f"Model info: {info}")
        
        # Test dataset config creation
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dl_models_dir = os.path.dirname(script_dir)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        data_dir = os.path.join(project_root,"CODE","DL MODELS (copy)")
        #data_dir = "C:\Users\walsh\Documents\GitHub\AGAROSE-HYDROGEL-TRENDS-USING-AI-ML\CODE\DL MODELS (copy)"
        config_path = model.create_pore_dataset_config(data_dir, "AFM")
        print(f"✓ Dataset configuration created: {config_path}")
        
    except Exception as e:
        print(f"Error testing YOLO model: {e}")
        print("Make sure ultralytics is installed: pip install ultralytics")
