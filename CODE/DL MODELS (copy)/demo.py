"""
Demo Script for Pore Detection System

This script provides a quick demonstration of the pore detection system
with sample data and simplified workflows for testing and validation.
"""

import os
import sys
import time
import argparse
from pathlib import Path

def run_quick_demo():
    """Run a quick demonstration with minimal setup"""
    print("quick Demo: Pore Detection System")
    print("="*50)
    
    # Check if we have sample data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    dataset_dir = os.path.join(project_root,'CODE','DL MODELS (copy)','Dataset')
    #dataset_dir = "C:\Users\walsh\Downloads\DL MODELS (copy)\Dataset"
    
    if not os.path.exists(dataset_dir):
        print("dataset directory not found!")
        print(f"Expected: {dataset_dir}")
        print("\nPlease ensure your dataset is properly structured:")
        print("Dataset/")
        print("├── AFM folder/")
        print("├── Confocal folder/")
        print("├── Cryo-sem Folder/")
        print("└── STED Folder/")
        return False
    
    print(f"Dataset found: {dataset_dir}")
    
    # Find a sample image for demonstration
    sample_image = find_sample_image(dataset_dir)
    if not sample_image:
        print("No sample images found for demonstration")
        return False
    
    print(f"Sample image: {sample_image}")
    
    # Run quick inference demo
    print("Running inference demonstration...")
    
    try:
        from inference import PoreInferenceEngine
        
        # Test U-Net (fastest to demo)
        print("\nTesting U-Net model...")
        engine = PoreInferenceEngine('unet', 'AFM')
        
        result = engine.process_image(sample_image, "demo_results/unet_demo")
        
        if result['success']:
            print("U-Net demo completed successfully!")
            print(f"   Processing time: {result['timing']['total_time']:.3f} seconds")
            print(f"   Pore coverage: {result['pore_statistics']['pore_percentage']:.2f}%")
            print(f"   Results saved to: {result['output_files']['binary_mask_path']}")
        else:
            print(f"U-Net demo failed: {result['error']}")
    
    except Exception as e:
        print(f"Demo failed: {e}")
        return False
    
    return True

def find_sample_image(dataset_dir: str) -> str:
    """Find a sample image for demonstration"""
    
    # Look for AFM images first (usually work well)
    afm_dir = os.path.join(dataset_dir, "AFM folder", "AFM")
    
    if os.path.exists(afm_dir):
        for root, dirs, files in os.walk(afm_dir):
            for file in files:
                if file.lower().endswith(('.tif', '.tiff')):
                    return os.path.join(root, file)
    
    # If no AFM images, try other types
    for folder in ["Confocal folder/CONFOCAL", "Cryo-sem Folder/CRYO-SEM", "STED Folder/STED"]:
        search_dir = os.path.join(dataset_dir, folder)
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.lower().endswith(('.tif', '.tiff')):
                        return os.path.join(root, file)
    
    return None

def run_training_demo():
    """Run a minimal training demonstration"""
    print("Training Demo: Quick U-Net Training")
    print("="*50)
    
    try:
        from train_unet import UNetTrainer
        
        print("Initializing U-Net trainer for AFM images...")
        print("Note: This is a minimal demo with few epochs")
        
        trainer = UNetTrainer(
            data_dir="C:\Users\walsh\Downloads\DL MODELS (copy)\Dataset",
            image_type="AFM",
            batch_size=4,  # Small batch for demo
            learning_rate=1e-4,
            num_epochs=5,  # Very few epochs for demo
            image_size=(256, 256)  # Smaller size for speed
        )
        
        print("Starting minimal training (5 epochs)...")
        results = trainer.train()
        
        print("Training demo completed!")
        print(f"training time: {results['total_training_time']:.2f} seconds")
        print(f"final loss: {results['final_train_loss']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"training demo failed: {e}")
        return False

def show_system_info():
    """Display system information relevant to deep learning"""
    print("system Information")
    print("="*40)
    
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Count: {torch.cuda.device_count()}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: Not installed")
    
    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except ImportError:
        print("OpenCV: Not installed")
    
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except ImportError:
        print("NumPy: Not installed")

def run_model_comparison_demo():
    """Run a comparison demo of all three models"""
    print("model Comparison Demo")
    print("="*50)
    
    # Find sample image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    dataset_dir =  os.path.join(project_root,'CODE','DL MODELS (copy)','Dataset')
    sample_image = find_sample_image(dataset_dir)
    
    if not sample_image:
        print("no sample image found for comparison")
        return False
    
    print(f"Testing all models on: {os.path.basename(sample_image)}")
    
    models = ['unet', 'pored2', 'yolo']
    results = {}
    
    for model in models:
        print(f"testing {model.upper()}...")
        
        try:
            from inference import PoreInferenceEngine
            
            engine = PoreInferenceEngine(model, 'AFM')
            result = engine.process_image(
                sample_image, 
                f"demo_results/comparison_{model}"
            )
            
            if result['success']:
                results[model] = result
                print(f"{model.upper()}: {result['timing']['total_time']:.3f}s, "
                      f"{result['pore_statistics']['pore_percentage']:.2f}% coverage")
            else:
                print(f"{model.upper()}: {result['error']}")
        
        except Exception as e:
            print(f"{model.upper()}: {e}")
    
    # Print comparison
    if len(results) > 1:
        print(f"model Comparison Summary:")
        print("-" * 40)
        
        for model, result in results.items():
            print(f"{model.upper():8}: {result['timing']['total_time']:.3f}s, "
                  f"{result['pore_statistics']['pore_percentage']:.2f}% coverage")
        
        # Find fastest and most thorough
        fastest = min(results.items(), key=lambda x: x[1]['timing']['total_time'])
        most_thorough = max(results.items(), key=lambda x: x[1]['pore_statistics']['pore_percentage'])
        
        print(f"fastest: {fastest[0].upper()} ({fastest[1]['timing']['total_time']:.3f}s)")
        print(f"most Thorough: {most_thorough[0].upper()} ({most_thorough[1]['pore_statistics']['pore_percentage']:.2f}%)")
    
    return len(results) > 0

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Pore Detection System Demo')
    parser.add_argument('--demo_type', type=str, default='quick',
                       choices=['quick', 'training', 'comparison', 'system_info'],
                       help='Type of demo to run')
    
    args = parser.parse_args()
    
    print("PORE DETECTION SYSTEM DEMO")
    print("This demo showcases the capabilities of our comprehensive")
    print("pore detection system with U-Net, PoreD², and YOLO models.")
    
    if args.demo_type == 'system_info':
        show_system_info()
        return 0
    
    elif args.demo_type == 'quick':
        success = run_quick_demo()
        
    elif args.demo_type == 'training':
        success = run_training_demo()
        
    elif args.demo_type == 'comparison':
        success = run_model_comparison_demo()
    
    else:
        print(f"Unknown demo type: {args.demo_type}")
        return 1
    
    if success:
        print("demo completed successfully!")
        print("\nNext steps:")
        print("1. Review the generated results in the demo_results/ folder")
        print("2. Try training models with more epochs for better performance")
        print("3. Run the full analysis script for comprehensive comparison")
        print("4. Check README.md for detailed usage instructions")
        return 0
    else:
        print("demo encountered issues")
        print("Please check your installation and dataset setup")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
