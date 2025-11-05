"""
Quick Start Guide - Pore Detection System

This script provides a streamlined way to get started with the pore detection system.
It handles common setup tasks and provides guided workflows.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
dat_dir = os.path.join(project_root,'CODE','DL MODELS (copy)')

def check_requirements():
    """Check if all required packages are installed"""
    print(" Checking requirements...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'cv2',
        'numpy',
        'PIL',
        'albumentations',
        'tensorboard',
        'ultralytics',
        'matplotlib',
        'seaborn',
        'scipy',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"   {package}")
        except ImportError:
            print(f"   {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n Missing packages: {missing_packages}")
        print("Please install them using:")
        print("pip install torch torchvision opencv-python pillow albumentations tensorboard ultralytics matplotlib seaborn scipy scikit-learn tqdm")
        return False
    
    return True

def check_dataset_structure():
    """Check if dataset is properly structured"""
    print("\n Checking dataset structure...")
    
    dataset_dir = os.path.join(dat_dir,'Dataset')
    
    if not os.path.exists(dataset_dir):
        print(f" Dataset directory not found: {dataset_dir}")
        return False
    
    required_folders = [
        "AFM folder/AFM",
        "AFM folder/AFM Training", 
        "Confocal folder/CONFOCAL",
        "Confocal folder/CONFOCAL Training",
        "Cryo-sem Folder/CRYO-SEM",
        "Cryo-sem Folder/CRYO-SEM_Training",
        "STED Folder/STED",
        "STED Folder/STED Training"
    ]
    
    found_folders = []
    for folder in required_folders:
        full_path = os.path.join(dataset_dir, folder)
        if os.path.exists(full_path):
            # Count images in folder
            image_count = 0
            for root, dirs, files in os.walk(full_path):
                image_count += len([f for f in files if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))])
            
            print(f"   {folder} ({image_count} images)")
            found_folders.append(folder)
        else:
            print(f"   {folder} (not found)")
    
    if len(found_folders) >= 2:  # At least one modality with training data
        print(f"\n Found {len(found_folders)} valid data folders")
        return True
    else:
        print(f"\n Insufficient data folders found. Need at least one imaging modality with training data.")
        return False

def create_output_directories():
    """Create necessary output directories"""
    print("\n Creating output directories...")
    
    directories = [
        "results",
        "models", 
        "logs",
        "demo_results",
        "checkpoints",
        "visualizations"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   {directory}/")

def test_gpu_availability():
    """Test if GPU is available and working"""
    print("\n Checking GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"   GPU Available: {gpu_name}")
            print(f"   GPU Count: {gpu_count}")
            print(f"   GPU Memory: {memory:.1f} GB")
            
            # Test a simple tensor operation
            test_tensor = torch.randn(10, 10).cuda()
            result = torch.mm(test_tensor, test_tensor.t())
            
            print(f"   GPU Test: Success")
            return True
        else:
            print(f"   GPU Not Available - will use CPU")
            print(f"   For faster training, consider using a GPU-enabled environment")
            return False
            
    except Exception as e:
        print(f"   GPU Test Failed: {e}")
        return False

def run_quick_test():
    """Run a quick test to ensure everything works"""
    print("\n Running quick system test...")
    
    try:
        # Test data loading
        print("  Testing data loading...")
        from data_utils import create_dataloaders
        
        # Test with a small subset
        train_loader, val_loader = create_dataloaders(
            data_dir = os.path.join(dat_dir,'Dataset'),
            image_type="AFM",
            batch_size=2,
            image_size=(128, 128),
            num_workers=0
        )
        
        print("   Data loading successful")
        
        # Test model loading
        print("  Testing model imports...")
        from models.unet_model import UNet
        from models.pored2_model import PoreD2Model
        
        model_unet = UNet(in_channels=1, out_channels=1)
        model_pored2 = PoreD2Model(in_channels=1, num_classes=1)
        
        print("   Model imports successful")
        
        # Test inference engine
        print("  Testing inference engine...")
        from inference import PoreInferenceEngine
        
        # Just initialize, don't run inference
        engine = PoreInferenceEngine('unet', 'AFM')
        print("   Inference engine initialization successful")
        
        return True
        
    except Exception as e:
        print(f"   System test failed: {e}")
        return False

def create_quick_start_config():
    """Create a configuration file with recommended settings"""
    print("\n Creating quick start configuration...")
    
    config = {
        "dataset_path": "C:\Users\walsh\Downloads\DL MODELS (copy)\Dataset",
        "output_path": "results",
        "models_to_train": ["unet", "pored2", "yolo"],
        "image_types": ["AFM", "CONFOCAL", "CRYO-SEM", "STED"],
        "training_settings": {
            "batch_size": 8,
            "learning_rate": 1e-4,
            "num_epochs": 100,
            "image_size": [512, 512],
            "early_stopping_patience": 15,
            "checkpoint_interval": 10
        },
        "inference_settings": {
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "output_format": "binary_tif"
        }
    }
    
    with open("quickstart_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("   Configuration saved to quickstart_config.json")

def print_next_steps():
    """Print recommended next steps"""
    print("\n QUICK START COMPLETE!")
    print("Your pore detection system is ready! Here are the next steps:")
    print()
    print("1.  Run a quick demo:")
    print("   python demo.py --demo_type quick")
    print()
    print("2.  Train a single model:")
    print("   python train_unet.py --data_dir 'Dataset' --image_type AFM --epochs 50")
    print()
    print("3.  Train all models:")
    print("   python train_all_models.py --data_dir 'Dataset' --epochs 50")
    print()
    print("4.  Run inference on an image:")
    print("   python inference.py --model_type unet --image_type AFM --input_image 'path/to/image.tif'")
    print()
    print("5.  Analyze and compare models:")
    print("   python test_and_analyze.py --models unet pored2 yolo --image_types AFM CONFOCAL")
    print()
    print("6.  For detailed documentation:")
    print("   Open README.md")
    print()
    print(" Tips:")
    print("- Start with the demo to ensure everything works")
    print("- Begin training with fewer epochs (10-20) to test quickly")
    print("- Use GPU if available for significantly faster training")
    print("- Check logs/ directory for training progress")

def main():
    """Main quick start function"""
    print(" PORE DETECTION SYSTEM - QUICK START")
    print("="*60)
    print("This script will help you set up and test the pore detection system.")
    print("="*60)
    
    # Step 1: Check requirements
    if not check_requirements():
        print("\n Please install missing packages and run again.")
        return 1
    
    # Step 2: Check dataset
    if not check_dataset_structure():
        print("\n Please ensure your dataset is properly structured and run again.")
        return 1
    
    # Step 3: Create directories
    create_output_directories()
    
    # Step 4: Test GPU
    gpu_available = test_gpu_availability()
    
    # Step 5: Run quick test
    if not run_quick_test():
        print("\n System test failed. Please check your installation.")
        return 1
    
    # Step 6: Create config
    create_quick_start_config()
    
    # Step 7: Show next steps
    print_next_steps()
    
    print("\n Quick start completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
