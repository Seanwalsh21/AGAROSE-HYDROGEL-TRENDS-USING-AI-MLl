import os
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import List, Tuple, Dict
import shutil
from tqdm import tqdm
from ultralytics import YOLO
import torch
import re

# Import our real data loader
from ilastik_data_loaderr import IlastikDataMapper

class RealIlastikYOLOConverter:
    """Converts your real Ilastik segmentation data to YOLO format"""
    
    def __init__(self, output_dir: str, image_size: int = 640):
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.mapper = IlastikDataMapper()
        
        # Create YOLO directory structure
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        
        for split in ['train', 'val']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
    
    def mask_to_yolo_polygons(self, mask: np.ndarray) -> List[List[float]]:
        """
        Convert binary mask to YOLO polygon format
        
        Args:
            mask: Binary mask (0s and 1s)
            
        Returns:
            List of normalized polygon coordinates for each connected component
        """
        polygons = []
        
        #find contours in the mask
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        height, width = mask.shape
        
        for contour in contours:
            #skip very small contours (noise)
            if cv2.contourArea(contour) < 50:  # Increased threshold for real data
                continue
            
            #simplify contour
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            #convert to normalized coordinates
            polygon = []
            for point in approx:
                x, y = point[0]
                #normalize coordinates to [0, 1]
                norm_x = max(0, min(1, x / width))
                norm_y = max(0, min(1, y / height))
                polygon.extend([norm_x, norm_y])
            
            #yolo requires at least 6 coordinates (3 points)
            if len(polygon) >= 6:
                polygons.append(polygon)
        
        return polygons
    
    def convert_modality_data(self, imaging_modality: str, val_split: float = 0.2):
        print(f"Converting {imaging_modality} data to YOLO format...")
        
        #get image-label pairs using our real data mapper
        pairs = self.mapper.load_data_pairs(imaging_modality)
        
        if not pairs:
            print(f"No valid pairs found for {imaging_modality}")
            return 0, 0
        
        print(f"Found {len(pairs)} pairs for {imaging_modality}")
        
        #split into train and validation
        np.random.shuffle(pairs)
        split_idx = int(len(pairs) * (1 - val_split))
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]
        
        #process training data
        train_count = self._process_pairs(train_pairs, 'train', imaging_modality)
        
        #process validation data
        val_count = self._process_pairs(val_pairs, 'val', imaging_modality)
        
        print(f"{imaging_modality}: {train_count} train, {val_count} val samples")
        return train_count, val_count
    
    def _process_pairs(self, pairs: List[Tuple[str, str]], split: str, modality: str) -> int:
        # process image-label pairs for a specific split
        
        processed_count = 0
        
        for img_path, label_path in tqdm(pairs, desc=f"Processing {split}"):
            try:
                # Load image
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                
                #load mask - try different methods for Ilastik files
                mask = None
                
                #method 1: Try as regular image
                mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                
                #method 2: If that fails, try with PIL for TIFF files
                if mask is None:
                    try:
                        from PIL import Image
                        mask_pil = Image.open(label_path)
                        mask = np.array(mask_pil)
                        if len(mask.shape) == 3:
                            mask = mask[:, :, 0]
                    except Exception as e:
                        print(f"Failed to load mask with PIL: {e}")
                
                #method 3: Try with skimage for complex TIFF files
                if mask is None:
                    try:
                        from skimage import io
                        mask = io.imread(label_path)
                        if len(mask.shape) == 3:
                            mask = mask[:, :, 0]
                    except Exception as e:
                        print(f"Failed to load mask with skimage: {e}")
                        continue
                
                if mask is None:
                    print(f"Failed to load mask: {label_path}")
                    continue
                
                #check mask properties
                print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, unique values: {np.unique(mask)}")
                
                #handle different mask formats from Ilastik
                if mask.dtype in [np.float32, np.float64]:
                    #probability maps from Ilastik - threshold at 0.5
                    mask = (mask > 0.5).astype(np.uint8) * 255
                elif mask.max() <= 1:
                    #binary mask with values 0,1
                    mask = (mask * 255).astype(np.uint8)
                else:
                    #regular grayscale - threshold at middle value
                    threshold = mask.max() // 2
                    mask = (mask > threshold).astype(np.uint8) * 255
                #skip if no objects found
                if mask.max() == 0:
                    print(f"No objects found in mask: {label_path}")
                    continue
                
                #resize image and mask to target size
                image_resized = cv2.resize(image, (self.image_size, self.image_size))
                mask_resized = cv2.resize(mask, (self.image_size, self.image_size), 
                                        interpolation=cv2.INTER_NEAREST)
                
                #make sure mask is binary after resize
                mask_resized = (mask_resized > 127).astype(np.uint8)
                
                #convert grayscale to RGB for YOLO
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
                
                #generate filename
                base_name = Path(img_path).stem
                #clean the filename
                clean_name = re.sub(r'[^\w\-_.]', '_', base_name)
                image_filename = f"{modality.lower()}_{processed_count:04d}_{clean_name}.jpg"
                label_filename = f"{modality.lower()}_{processed_count:04d}_{clean_name}.txt"
                
                #save image
                image_output_path = self.images_dir / split / image_filename
                cv2.imwrite(str(image_output_path), image_rgb)
                #convert mask to YOLO polygons
                polygons = self.mask_to_yolo_polygons(mask_resized)
                #save labels
                label_output_path = self.labels_dir / split / label_filename
                with open(label_output_path, 'w') as f:
                    for polygon in polygons:
                        # Format: class_id x1 y1 x2 y2 x3 y3 ...
                        coords_str = ' '.join([f"{coord:.6f}" for coord in polygon])
                        f.write(f"0 {coords_str}\n")  # class_id = 0 for 'pore'  
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        return processed_count
    def create_dataset_yaml(self, modality: str) -> str:
        """Create YOLO dataset configuration file"""
        
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/val',
            'names': {0: 'pore'},
            'nc': 1
        }
        yaml_path = self.output_dir / f"dataset_{modality.lower()}.yaml"
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return str(yaml_path)

class RealIlastikYOLOTrainer:
    # yolo trainer using real Ilastik segmentation data
    def __init__(self, 
                 imaging_modality: str,
                 model_size: str = 'n',
                 epochs: int = 50,
                 batch_size: int = 8,
                 image_size: int = 640,
                 device: str = 'auto'):
        self.imaging_modality = imaging_modality.upper()
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size
        
        #set device - default to CPU if CUDA not available
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        #setup output directory
        self.output_dir = Path(f"yolo_training_{self.imaging_modality.lower()}_real")
        self.data_dir = self.output_dir / "data"
        
        #initialize converter
        self.converter = RealIlastikYOLOConverter(str(self.data_dir), image_size)
        
        print(f"Initialized REAL YOLO trainer for {self.imaging_modality}")
        print(f"Model: YOLOv11{self.model_size}-seg")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
    
    def prepare_data(self) -> str:
        #prepare YOLO dataset from real Ilastik data"""
        print("preparing YOLO dataset from REAL Ilastik data...")
        #convert data
        train_count, val_count = self.converter.convert_modality_data(
            self.imaging_modality, val_split=0.2
        )
        if train_count == 0:
            raise ValueError(f"No training data found for {self.imaging_modality}")
        #create dataset configuration
        yaml_path = self.converter.create_dataset_yaml(self.imaging_modality)
        print(f"dataset prepared: {train_count} train, {val_count} val samples")
        print(f"config saved to: {yaml_path}")
        
        return yaml_path
    
    def train(self) -> Dict:
        try:
            #prepare dataset
            dataset_config = self.prepare_data()
            
            #initialize YOLO model
            model_name = f"yolo11{self.model_size}-seg.pt"
            model = YOLO(model_name)
            
            print(f"starting REAL YOLO training...")
            print(f"model: {model_name}")
            print(f"dataset: {dataset_config}")
            print(f"epochs: {self.epochs}")
            print(f"batch size: {self.batch_size}")
            print(f"image size: {self.image_size}")
            
            # Create runs directory
            runs_dir = self.output_dir / "runs"
            runs_dir.mkdir(exist_ok=True)
            
            # Train model
            results = model.train(
                data=dataset_config,
                epochs=self.epochs,
                imgsz=self.image_size,
                batch=self.batch_size,
                device=self.device,
                project=str(runs_dir),
                name=f"pore_yolo_{self.imaging_modality.lower()}_real",
                exist_ok=True,
                verbose=True
            )
            
            # Save final model
            model_save_path = self.output_dir / f"yolo_{self.imaging_modality.lower()}_real_final.pt"
            model.save(str(model_save_path))
            
            training_results = {
                'imaging_modality': self.imaging_modality,
                'model_architecture': f'YOLOv11{self.model_size}-seg',
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'image_size': self.image_size,
                'dataset_config': dataset_config,
                'model_path': str(model_save_path),
                'results': results,
                'output_directory': str(self.output_dir)
            }
            
            print(f"yolo training completed for {self.imaging_modality}")
            print(f"Model saved to: {model_save_path}")
            
            return training_results
            
        except Exception as e:
            print(f"training failed: {e}")
            import traceback
            traceback.print_exc()
            raise e

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="train yolo with ilastik data")
    parser.add_argument("--modality", type=str, required=True,
                        choices=['AFM', 'CRYO-SEM', 'STED', 'CONFOCAL'],
                        help="Imaging modality to train on")
    parser.add_argument("--model_size", type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help="YOLO model size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--image_size", type=int, default=640,
                        help="Input image size")
    
    args = parser.parse_args()
    
    print("yolo training")
    print(f"Arguments: {vars(args)}")
    
    #train model with REAL data
    try:
        trainer = RealIlastikYOLOTrainer(
            imaging_modality=args.modality,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size
        )
        
        results = trainer.train()
        
        print("training completed successfully!")
        print("you now have a YOLO model trained on real Ilastik segmentation data!")
        
    except Exception as e:
        print(f"script failed with error: {e}")
        import traceback
        traceback.print_exc()