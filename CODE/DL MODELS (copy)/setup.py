"""
Installation and Setup Script for Pore Detection System

This script helps set up the environment and test the installation
of the comprehensive pore detection system.
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("python 3.8+ is required")
        return False
    print(f"python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("installing required packages...")
    
    try:
        # Install basic requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"failed to install requirements: {e}")
        return False

def check_gpu_availability():
    """Check GPU availability for deep learning"""
    print("checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {gpu_name} ({gpu_count} device(s))")
            print(f"CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("No GPU detected, will use CPU (slower training)")
            return False
    except ImportError:
        print("PyTorch not installed, GPU check skipped")
        return False

def create_directories():
    """Create necessary directories"""
    print("creating project directories...")
    
    directories = [
        "checkpoints",
        "logs",
        "results",
        "yolo_training_afm",
        "yolo_training_confocal", 
        "yolo_training_cryo-sem",
        "yolo_training_sted"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"{directory}/")
    
    print("Directories created successfully")

def test_data_loading():
    """Test data loading functionality"""
    print("testing data loading...")
    
    try:
        from data_utils import PoreDetectionDataset
        print("data utilities imported successfully")
        return True
    except ImportError as e:
        print(f"failed to import data utilities: {e}")
        return False

def test_model_imports():
    """Test model imports"""
    print("testing model imports...")
    
    models_status = {}
    
    # Test U-Net
    try:
        from models.unet_model import UNet, create_unet_model
        models_status['U-Net'] = True
        print("U-Net model")
    except ImportError as e:
        models_status['U-Net'] = False
        print(f"U-Net model: {e}")
    
    # Test PoreD²
    try:
        from models.pored2_model import PoreD2Model, create_pored2_model
        models_status['PoreD²'] = True
        print("PoreD² model")
    except ImportError as e:
        models_status['PoreD²'] = False
        print(f"PoreD² model: {e}")
    
    # Test YOLO
    try:
        from models.yolo_model import PoreYOLOModel, create_yolo_model
        models_status['YOLO'] = True
        print("YOLO model")
    except ImportError as e:
        models_status['YOLO'] = False
        print(f"YOLO model: {e}")
    
    successful_models = sum(models_status.values())
    total_models = len(models_status)
    
    if successful_models == total_models:
        print(f"All {total_models} models imported successfully")
        return True
    else:
        print(f"{successful_models}/{total_models} models imported successfully")
        return successful_models > 0

def create_sample_config():
    """Create sample configuration files"""
    print("creating sample configuration...")
    
    # Sample training configuration
    config = {
        "data_dir": "C:\Users\walsh\Downloads\DL MODELS (copy)\Dataset",
        "models": ["unet", "pored2", "yolo"],
        "image_types": ["AFM", "CONFOCAL", "CRYO-SEM", "STED"],
        "epochs": {
            "unet": 50,
            "pored2": 75,
            "yolo": 50
        },
        "batch_sizes": {
            "unet": 8,
            "pored2": 4,
            "yolo": 16
        }
    }
    
    import json
    with open("sample_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("sample configuration created: sample_config.json")

def print_next_steps():
    """Print next steps for the user"""
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("next Steps:")
    print("1. Prepare your dataset:")
    print("   - Ensure dataset follows the required structure")
    print("   - Update paths in sample_config.json if needed")
    
    print("2. Start training (choose one):")
    print("   # Train all models")
    print("   python train_all_models.py --data_dir 'path/to/Dataset' --epochs 50")
    print("\n   # Train individual models")
    print("   python train_unet.py --image_type AFM --epochs 100")
    print("   python train_pored2.py --image_type CONFOCAL --epochs 150")
    print("   python train_yolo.py --image_type CRYO-SEM --epochs 100")
    
    print("\n3. Run inference:")
    print("   python inference.py --model_type unet --image_type AFM --input_image 'path/to/image.tif'")
    
    print("\n4. Analyze results:")
    print("   python test_and_analyze.py --test_dir 'path/to/Dataset'")
    
    print("Documentation:")
    print("   - See README.md for detailed usage instructions")
    print("   - Check sample_config.json for configuration options")
    print("   - Review model architectures in models/ directory")
    
    print("Troubleshooting:")
    print("   - Ensure GPU drivers are installed for CUDA support")
    print("   - Check dataset paths and permissions")
    print("   - Monitor GPU memory usage during training")
    
    print("="*60)

def main():
    """Main setup function"""
    print("Pore Detection System Setup")
    print("="*60)
    print("Setting up comprehensive pore detection system...")
    print("This includes U-Net, PoreD², and YOLO models")
    print("="*60)
    
    success_count = 0
    total_checks = 6
    
    # Run setup checks
    if check_python_version():
        success_count += 1
    
    if install_requirements():
        success_count += 1
    
    if check_gpu_availability():
        success_count += 1
    
    create_directories()
    success_count += 1
    
    if test_data_loading():
        success_count += 1
    
    if test_model_imports():
        success_count += 1
    
    create_sample_config()
    
    # Print results
    print(f"Setup Summary: {success_count}/{total_checks} checks passed")
    
    if success_count == total_checks:
        print_next_steps()
        return 0
    elif success_count >= 4:
        print("Setup completed with some warnings")
        print("You can proceed but may encounter issues")
        print_next_steps()
        return 0
    else:
        print("Setup failed. Please resolve the above issues.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
