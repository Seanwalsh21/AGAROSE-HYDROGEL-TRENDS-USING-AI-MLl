"""
Usage Examples - Pore Detection System

This file contains practical examples of how to use the pore detection system
for common tasks and workflows.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def example_1_quick_inference():
    """
    Example 1: Quick inference on a single image
    This is the fastest way to get pore detection results on a single image.
    """
    print("=" * 60)
    print("EXAMPLE 1: Quick Inference on Single Image")
    print("=" * 60)
    print("Purpose: Get pore detection results quickly on one image")
    print("Time: ~10-30 seconds per image")
    print("Use case: Testing, quick analysis")
    print()
    
    # Step 1: Import the inference engine
    from inference import PoreInferenceEngine
    
    # Step 2: Choose your model and image type
    model_type = 'unet'  # Options: 'unet', 'pored2', 'yolo'
    image_type = 'AFM'   # Options: 'AFM', 'CONFOCAL', 'CRYO-SEM', 'STED'
    
    # Step 3: Initialize the inference engine
    print(f"Initializing {model_type.upper()} model for {image_type} images...")
    engine = PoreInferenceEngine(model_type, image_type)
    
    # Step 4: Process your image
    input_image = "path/to/your/image.tif"  # Replace with actual path
    output_dir = "results/quick_inference"
    
    print(f"Processing: {input_image}")
    result = engine.process_image(input_image, output_dir)
    
    # Step 5: Check results
    if result['success']:
        print("inference completed successfully!")
        print(f"Processing time: {result['timing']['total_time']:.3f} seconds")
        print(f"Pore coverage: {result['pore_statistics']['pore_percentage']:.2f}%")
        print(f"Binary mask saved: {result['output_files']['binary_mask_path']}")
        print(f"Overlay saved: {result['output_files']['overlay_path']}")
    else:
        print(f"inference failed: {result['error']}")

def example_2_train_single_model():
    """
    Example 2: Train a single model on one image type
    This shows how to train one model (U-Net) on one imaging modality.
    """

    print("EXAMPLE 2: Train Single Model")
    print("Purpose: Train U-Net model on AFM images")
    print("Time: ~30-60 minutes (depends on data size and epochs)")
    print("Use case: Learning, experimentation")
    print()
    
    # Step 1: Import the trainer
    from train_unet import UNetTrainer
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    
    # Step 2: Set up training parameters
    trainer = UNetTrainer(
        data_dir = os.path.join(project_root,'CODE','DL MODELS (copy)','Dataset'),  # Your dataset path
        image_type="AFM",                               # Type of images to train on
        batch_size=8,                                   # Adjust based on GPU memory
        learning_rate=1e-4,                             # Learning rate
        num_epochs=50,                                  # Number of training epochs
        image_size=(512, 512),                          # Input image size
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Step 3: Start training
    print("Starting training...")
    print("Note: This will take several minutes to complete")
    print("Check logs/training_*.log for progress")
    
    results = trainer.train()
    
    # Step 4: Check results
    print("training completed!")
    print(f"Training time: {results['total_training_time']:.2f} seconds")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Model saved to: {results['model_path']}")

def example_3_batch_inference():
    """
    Example 3: Process multiple images at once
    This shows how to process an entire folder of images.
    """
    print("=" * 60)
    print("EXAMPLE 3: Batch Inference")
    print("=" * 60)
    print("Purpose: Process multiple images in a folder")
    print("Time: ~10-30 seconds per image")
    print("Use case: Processing datasets, production use")
    print()
    
    from inference import PoreInferenceEngine
    import glob
    import time
    
    # Step 1: Set up parameters
    model_type = 'unet'
    image_type = 'AFM'
    input_folder = "path/to/your/images/"  # Replace with actual path
    output_folder = "results/batch_processing"
    
    # Step 2: Find all images in folder
    image_extensions = ['*.tif', '*.tiff', '*.png', '*.jpg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    print(f"Found {len(image_files)} images to process")
    
    # Step 3: Initialize engine
    engine = PoreInferenceEngine(model_type, image_type)
    
    # Step 4: Process each image
    results = []
    total_start_time = time.time()
    
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Create unique output directory for each image
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(output_folder, image_name)
        
        result = engine.process_image(image_path, output_dir)
        results.append(result)
        
        if result['success']:
            print(f"Success: {result['timing']['total_time']:.3f}s, "
                  f"{result['pore_statistics']['pore_percentage']:.2f}% pores")
        else:
            print(f"Failed: {result['error']}")
    
    # Step 5: Summary
    total_time = time.time() - total_start_time
    successful = sum(1 for r in results if r['success'])
    
    print(f"Batch Processing Summary:")
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(image_files) - successful}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/len(image_files):.2f} seconds")

def example_4_compare_models():
    """
    Example 4: Compare all three models on the same image
    This shows how to run all models and compare their results.
    """
    print("EXAMPLE 4: Model Comparison")
    print("Purpose: Compare U-Net, PoreD², and YOLO on same image")
    print("Time: ~1-2 minutes total")
    print("Use case: Research, method comparison")
    print()
    
    from inference import PoreInferenceEngine
    
    # Step 1: Set up parameters
    image_type = 'AFM'
    input_image = "path/to/your/test_image.tif"  # Replace with actual path
    models = ['unet', 'pored2', 'yolo']
    
    # Step 2: Run each model
    results = {}
    
    for model in models:
        print(f"testing {model.upper()} model...")
        
        try:
            engine = PoreInferenceEngine(model, image_type)
            output_dir = f"results/comparison/{model}"
            
            result = engine.process_image(input_image, output_dir)
            results[model] = result
            
            if result['success']:
                print(f"success: {result['timing']['total_time']:.3f}s")
                print(f"pore coverage: {result['pore_statistics']['pore_percentage']:.2f}%")
                print(f"results: {result['output_files']['binary_mask_path']}")
            else:
                print(f"failed: {result['error']}")
        
        except Exception as e:
            print(f"error: {e}")
            results[model] = {'success': False, 'error': str(e)}
    
    # Step 3: Compare results
    print(f"MODEL COMPARISON SUMMARY")

    
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if successful_results:
        # Find fastest model
        fastest = min(successful_results.items(), 
                     key=lambda x: x[1]['timing']['total_time'])
        
        # Find most thorough (highest pore percentage)
        most_thorough = max(successful_results.items(), 
                           key=lambda x: x[1]['pore_statistics']['pore_percentage'])
        
        print(f"fastest: {fastest[0].upper()} "
              f"({fastest[1]['timing']['total_time']:.3f}s)")
        print(f"most Thorough: {most_thorough[0].upper()} "
              f"({most_thorough[1]['pore_statistics']['pore_percentage']:.2f}%)")
        
        print(f"\nDetailed Results:")
        for model, result in successful_results.items():
            print(f"  {model.upper():8}: "
                  f"{result['timing']['total_time']:.3f}s, "
                  f"{result['pore_statistics']['pore_percentage']:.2f}% coverage")

def example_5_full_training_pipeline():
    """
    Example 5: Train all models on all image types
    This is the complete training pipeline for research purposes.
    """

    print("EXAMPLE 5: Full Training Pipeline")
    print("Purpose: Train all models on all image types")
    print("Time: Several hours (3-6 hours depending on hardware)")
    print("Use case: Research, comprehensive analysis")
    print()
    
    # This would run the master training script
    print("To run the full training pipeline:")
    print()
    print("Command line:")
    print("  python train_all_models.py --data_dir 'Dataset' --epochs 100")
    print()
    print("Or programmatically:")
    print("""
    from train_all_models import MasterTrainer
    
    trainer = MasterTrainer(
        data_dir="C:\Users\walsh\Downloads\DL MODELS (copy)\Dataset",
        output_dir="results/full_training",
        models=['unet', 'pored2', 'yolo'],
        image_types=['AFM', 'CONFOCAL', 'CRYO-SEM', 'STED'],
        epochs=100,
        batch_size=8
    )
    
    results = trainer.train_all()
    """)
    print()
    print("This will:")
    print("- Train 12 models total (3 models × 4 image types)")
    print("- Save checkpoints and logs for each")
    print("- Generate comprehensive comparison report")
    print("- Take several hours to complete")

def example_6_custom_analysis():
    """
    Example 6: Custom analysis and visualization
    This shows how to analyze results and create custom visualizations.
    """
    print("=" * 60)
    print("EXAMPLE 6: Custom Analysis")
    print("=" * 60)
    print("Purpose: Analyze results and create visualizations")
    print("Time: ~5-10 minutes")
    print("Use case: Research analysis, presentation")
    print()
    
    print("To run comprehensive analysis:")
    print()
    print("Command line:")
    print("  python test_and_analyze.py --models unet pored2 yolo --image_types AFM CONFOCAL")
    print()
    print("Or for custom analysis:")
    print("""
    from test_and_analyze import PoreDetectionAnalyzer
    
    analyzer = PoreDetectionAnalyzer(
        results_dir="results",
        models=['unet', 'pored2', 'yolo'],
        image_types=['AFM', 'CONFOCAL', 'CRYO-SEM', 'STED']
    )
    
    # Generate performance comparison
    comparison = analyzer.compare_models()
    
    # Create visualizations
    analyzer.create_performance_plots()
    analyzer.create_sample_comparisons()
    
    # Generate detailed report
    report = analyzer.generate_detailed_report()
    """)
    print()
    print("This will create:")
    print("- Performance comparison charts")
    print("- Sample result visualizations")
    print("- Statistical analysis report")
    print("- Timing analysis")

def main():
    """
    Main function to run examples
    """
    print("PORE DETECTION SYSTEM - USAGE EXAMPLES")
    print("=" * 60)
    print("This file contains practical examples for common workflows.")
    print("Choose an example to run:")
    print()
    print("1. Quick inference on single image (fastest)")
    print("2. Train single model (U-Net on AFM)")
    print("3. Batch inference on multiple images")
    print("4. Compare all models on same image")
    print("5. Full training pipeline (all models, all types)")
    print("6. Custom analysis and visualization")
    print()
    
    try:
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            example_1_quick_inference()
        elif choice == '2':
            example_2_train_single_model()
        elif choice == '3':
            example_3_batch_inference()
        elif choice == '4':
            example_4_compare_models()
        elif choice == '5':
            example_5_full_training_pipeline()
        elif choice == '6':
            example_6_custom_analysis()
        else:
            print("Invalid choice. Please run the script again.")
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError running example: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()
