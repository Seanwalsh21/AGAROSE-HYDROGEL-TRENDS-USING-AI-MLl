"""
Master Training Script for All Pore Detection Models

This script orchestrates the training of all three deep learning models
(U-Net, PoreD^2, YOLO) across all four imaging modalities (AFM, Confocal, Cryo-SEM, STED).

The script provides:
1. Sequential training of all models for all image types
2. Comprehensive timing and performance tracking
3. Detailed logging and progress reporting
4. Error handling and recovery
5. Final comparison and analysis
"""

import os
import time
import argparse
import json
from typing import Dict, List
import traceback
from datetime import datetime

# Import training modules
import sys
sys.path.append('.')
from train_unet import UNetTrainer
from train_pored2 import PoreD2Trainer
#from train_yolo import YOLOPoreTrainer


class MasterTrainer:
    """
    Master trainer that coordinates training of all models across all image types
    
    This class manages the complete training pipeline for the pore detection
    project, ensuring proper sequencing, logging, and result compilation.
    """
    
    def __init__(self, data_dir: str, base_epochs: int = 50, 
                 unet_batch_size: int = 8, pored2_batch_size: int = 4, 
                 yolo_batch_size: int = 16):
        """
        Initialize the master trainer
        
        Args:
            data_dir: Root directory containing the dataset
            base_epochs: Base number of epochs for training
            unet_batch_size: Batch size for U-Net training
            pored2_batch_size: Batch size for PoreD^2 training
            yolo_batch_size: Batch size for YOLO training
        """
        
        self.data_dir = data_dir
        self.base_epochs = base_epochs
        self.unet_batch_size = unet_batch_size
        self.pored2_batch_size = pored2_batch_size
        self.yolo_batch_size = yolo_batch_size
        
        # Define image types and models
        self.image_types = ['AFM', 'CONFOCAL', 'CRYO-SEM', 'STED']
        self.models = ['unet', 'pored2', 'yolo']
        
        # Adjust epochs for different models based on complexity
        self.model_epochs = {
            'unet': base_epochs,
            'pored2': int(base_epochs * 1.5),  # More epochs for complex model
            'yolo': base_epochs
        }
        
        # Results storage
        self.training_results = {}
        self.training_summary = {
            'start_time': None,
            'end_time': None,
            'total_training_time': 0,
            'successful_trainings': 0,
            'failed_trainings': 0,
            'model_results': {}
        }
        
        # Setup logging
        self._setup_logging()
        
        print(f"Master Trainer Initialized")
        print(f"Data Directory: {self.data_dir}")
        print(f"Image Types: {self.image_types}")
        print(f"Models: {self.models}")
        print(f"Base Epochs: {self.base_epochs}")
        
    def _setup_logging(self):
        """Setup logging directory and files"""
        self.log_dir = f"training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.master_log_path = os.path.join(self.log_dir, "master_training_log.txt")
        
        # Initialize master log
        with open(self.master_log_path, 'w') as f:
            f.write("Pore Detection Master Training Log\n")
            f.write("=" * 60 + "\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Directory: {self.data_dir}\n")
            f.write(f"Image Types: {', '.join(self.image_types)}\n")
            f.write(f"Models: {', '.join(self.models)}\n\n")
        
        print(f"Logging directory created: {self.log_dir}")
    
    def _log_message(self, message: str, print_message: bool = True):
        """Log message to master log file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.master_log_path, 'a') as f:
            f.write(log_entry)
        
        if print_message:
            print(message)
    
    def train_unet_model(self, image_type: str) -> Dict:
        """
        Train U-Net model for specific image type
        
        Args:
            image_type: Type of imaging data
            
        Returns:
            Training results dictionary
        """
        self._log_message(f"\nStarting U-Net training for {image_type}")
        
        try:
            trainer = UNetTrainer(
                data_dir=self.data_dir,
                image_type=image_type,
                batch_size=self.unet_batch_size,
                learning_rate=1e-4,
                num_epochs=self.model_epochs['unet'],
                image_size=(512, 512)
            )
            
            results = trainer.train()
            
            self._log_message(f"✓ U-Net training completed for {image_type}")
            self._log_message(f"  Training time: {results['total_training_time']:.2f} seconds")
            self._log_message(f"  Best validation loss: {results['best_val_loss']:.6f}")
            
            return results
            
        except Exception as e:
            error_msg = f"U-Net training failed for {image_type}: {str(e)}"
            self._log_message(error_msg)
            self._log_message(f"Error traceback:\n{traceback.format_exc()}")
            
            return {
                'status': 'failed',
                'error': str(e),
                'image_type': image_type,
                'model': 'unet'
            }
    
    def train_pored2_model(self, image_type: str) -> Dict:
        """
        train PoreD^2 model for specific image type
        
        Args:
            image_type: Type of imaging data
            
        Returns:
            Training results dictionary
        """
        self._log_message(f"\nStarting PoreD^2 training for {image_type}")
        
        try:
            trainer = PoreD2Trainer(
                data_dir=self.data_dir,
                image_type=image_type,
                batch_size=self.pored2_batch_size,
                learning_rate=5e-5,
                num_epochs=self.model_epochs['pored2'],
                image_size=(512, 512)
            )
            
            results = trainer.train()
            
            self._log_message(f"✓ PoreD^2 training completed for {image_type}")
            self._log_message(f"  Training time: {results['total_training_time']:.2f} seconds")
            self._log_message(f"  Best validation loss: {results['best_val_loss']:.6f}")
            
            return results
            
        except Exception as e:
            error_msg = f"PoreD^2 training failed for {image_type}: {str(e)}"
            self._log_message(error_msg)
            self._log_message(f"Error traceback:\n{traceback.format_exc()}")
            
            return {
                'status': 'failed',
                'error': str(e),
                'image_type': image_type,
                'model': 'pored2'
            }
    
    def train_yolo_model(self, image_type: str) -> Dict:
        """
        Train YOLO model for specific image type
        
        Args:
            image_type: Type of imaging data
            
        Returns:
            Training results dictionary
        """
        self._log_message(f"\nStarting YOLO training for {image_type}")
        
        try:
            trainer = YOLOPoreTrainer(
                data_dir=self.data_dir,
                image_type=image_type,
                model_size='n',
                epochs=self.model_epochs['yolo'],
                batch_size=self.yolo_batch_size,
                image_size=640
            )
            
            results = trainer.train()
            
            self._log_message(f"yolo training completed for {image_type}")
            self._log_message(f"Training time: {results['total_training_time']:.2f} seconds")
            
            return results
            
        except Exception as e:
            error_msg = f"YOLO training failed for {image_type}: {str(e)}"
            self._log_message(error_msg)
            self._log_message(f"Error traceback:\n{traceback.format_exc()}")
            
            return {
                'status': 'failed',
                'error': str(e),
                'image_type': image_type,
                'model': 'yolo'
            }
    
    def train_all_models(self) -> Dict:
        """
        Train all models for all image types
        
        Returns:
            Complete training results
        """
        self.training_summary['start_time'] = datetime.now()
        total_start_time = time.time()
        
        self._log_message("\n" + "="*80)
        self._log_message("STARTING COMPREHENSIVE PORE DETECTION MODEL TRAINING")
        self._log_message("="*80)
        self._log_message(f"Total combinations to train: {len(self.image_types)} × {len(self.models)} = {len(self.image_types) * len(self.models)}")
        
        # Initialize results structure
        for image_type in self.image_types:
            self.training_results[image_type] = {}
            self.training_summary['model_results'][image_type] = {}
        
        training_count = 0
        total_combinations = len(self.image_types) * len(self.models)
        
        # Train each model for each image type
        for image_type in self.image_types:
            self._log_message(f"\n{'='*60}")
            self._log_message(f"TRAINING MODELS FOR {image_type} IMAGES")
            self._log_message(f"{'='*60}")
            
            for model in self.models:
                training_count += 1
                
                self._log_message(f"\n[{training_count}/{total_combinations}] Training {model.upper()} for {image_type}")
                
                # Train the specific model
                if model == 'unet':
                    results = self.train_unet_model(image_type)
                elif model == 'pored2':
                    results = self.train_pored2_model(image_type)
                elif model == 'yolo':
                    results = self.train_yolo_model(image_type)
                
                # Store results
                self.training_results[image_type][model] = results
                
                # Update summary
                if results.get('status') == 'failed':
                    self.training_summary['failed_trainings'] += 1
                    self.training_summary['model_results'][image_type][model] = 'FAILED'
                else:
                    self.training_summary['successful_trainings'] += 1
                    self.training_summary['model_results'][image_type][model] = 'SUCCESS'
                
                # Log progress
                success_rate = (self.training_summary['successful_trainings'] / training_count) * 100
                self._log_message(f"Progress: {training_count}/{total_combinations} ({training_count/total_combinations*100:.1f}%)")
                self._log_message(f"Success Rate: {success_rate:.1f}%")
        
        # Finalize timing
        total_training_time = time.time() - total_start_time
        self.training_summary['end_time'] = datetime.now()
        self.training_summary['total_training_time'] = total_training_time
        
        # Generate final report
        self._generate_final_report()
        
        # Save results to JSON
        self._save_results_json()
        
        self._log_message(f"\n{'='*80}")
        self._log_message("MASTER TRAINING COMPLETED")
        self._log_message(f"{'='*80}")
        self._log_message(f"Total Time: {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours)")
        self._log_message(f"Successful Trainings: {self.training_summary['successful_trainings']}/{total_combinations}")
        self._log_message(f"Failed Trainings: {self.training_summary['failed_trainings']}/{total_combinations}")
        self._log_message(f"Success Rate: {(self.training_summary['successful_trainings']/total_combinations)*100:.1f}%")
        
        return {
            'training_results': self.training_results,
            'training_summary': self.training_summary,
            'log_directory': self.log_dir
        }
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        report_path = os.path.join(self.log_dir, "final_training_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE PORE DETECTION TRAINING REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Training Period: {self.training_summary['start_time'].strftime('%Y-%m-%d %H:%M:%S')} to {self.training_summary['end_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Duration: {self.training_summary['total_training_time']:.2f} seconds ({self.training_summary['total_training_time']/3600:.2f} hours)\n")
            f.write(f"Image Types Processed: {len(self.image_types)}\n")
            f.write(f"Models Trained: {len(self.models)}\n")
            f.write(f"Total Training Runs: {len(self.image_types) * len(self.models)}\n")
            f.write(f"Successful Trainings: {self.training_summary['successful_trainings']}\n")
            f.write(f"Failed Trainings: {self.training_summary['failed_trainings']}\n")
            f.write(f"Success Rate: {(self.training_summary['successful_trainings']/(len(self.image_types) * len(self.models)))*100:.1f}%\n\n")
            
            f.write("DETAILED RESULTS BY IMAGE TYPE\n")
            f.write("-" * 40 + "\n\n")
            
            for image_type in self.image_types:
                f.write(f"{image_type} IMAGING:\n")
                f.write("  " + "-" * 20 + "\n")
                
                for model in self.models:
                    result = self.training_results[image_type][model]
                    status = "SUCCESS" if result.get('status') != 'failed' else "FAILED"
                    f.write(f"  {model.upper()}: {status}")
                    
                    if status == "SUCCESS":
                        if 'total_training_time' in result:
                            f.write(f" ({result['total_training_time']:.1f}s)")
                        if 'best_val_loss' in result:
                            f.write(f" - Loss: {result['best_val_loss']:.6f}")
                    else:
                        f.write(f" - Error: {result.get('error', 'Unknown')}")
                    
                    f.write("\n")
                f.write("\n")
            
            f.write("MODEL PERFORMANCE COMPARISON\n")
            f.write("-" * 40 + "\n\n")
            
            # Calculate average training times for successful models
            model_times = {model: [] for model in self.models}
            model_success_rates = {model: 0 for model in self.models}
            
            for image_type in self.image_types:
                for model in self.models:
                    result = self.training_results[image_type][model]
                    if result.get('status') != 'failed':
                        model_success_rates[model] += 1
                        if 'total_training_time' in result:
                            model_times[model].append(result['total_training_time'])
            
            for model in self.models:
                success_rate = (model_success_rates[model] / len(self.image_types)) * 100
                avg_time = sum(model_times[model]) / len(model_times[model]) if model_times[model] else 0
                
                f.write(f"{model.upper()}:\n")
                f.write(f"  Success Rate: {success_rate:.1f}% ({model_success_rates[model]}/{len(self.image_types)})\n")
                f.write(f"  Average Training Time: {avg_time:.1f} seconds\n")
                f.write(f"  Total Training Runs: {len(model_times[model])}\n\n")
            
            f.write("TECHNICAL SPECIFICATIONS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Data Directory: {self.data_dir}\n")
            f.write(f"Base Epochs: {self.base_epochs}\n")
            f.write(f"U-Net Batch Size: {self.unet_batch_size}\n")
            f.write(f"PoreD^2 Batch Size: {self.pored2_batch_size}\n")
            f.write(f"YOLO Batch Size: {self.yolo_batch_size}\n\n")
            
            f.write("MODEL ARCHITECTURES\n")
            f.write("-" * 40 + "\n")
            f.write("1. U-Net: Semantic segmentation model based on Ronneberger et al. (2015)\n")
            f.write("   - Encoder-decoder architecture with skip connections\n")
            f.write("   - Binary segmentation for pore detection\n")
            f.write("   - Input size: 512x512 pixels\n\n")
            
            f.write("2. PoreD^2: Advanced multi-task learning model\n")
            f.write("   - Custom architecture with attention mechanisms\n")
            f.write("   - Multi-task learning: segmentation + pore statistics\n")
            f.write("   - Separable convolutions for efficiency\n")
            f.write("   - Input size: 512x512 pixels\n\n")
            
            f.write("3. YOLO: Object detection approach (YOLOv8)\n")
            f.write("   - Treats pores as individual objects\n")
            f.write("   - Provides both detection and segmentation\n")
            f.write("   - Input size: 640x640 pixels\n")
            f.write("   - Based on Ultralytics implementation\n\n")
            
            f.write("DATASET INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write("Image Types:\n")
            f.write("- AFM (Atomic Force Microscopy): Various concentrations (1%, 1.5%, 2%)\n")
            f.write("- CONFOCAL (Confocal Microscopy): Various concentrations (0.375%, 1%)\n")
            f.write("- CRYO-SEM (Scanning Electron Microscopy): Various magnifications\n")
            f.write("- STED (Stimulated Emission Depletion): Various concentrations\n\n")
            
            f.write("Training Data:\n")
            f.write("- Augmented versions of original images (rotations, flips)\n")
            f.write("- Pseudo-labels generated using image processing techniques\n")
            f.write("- Unsupervised learning approach due to lack of manual annotations\n\n")
            
            f.write("REFERENCES\n")
            f.write("-" * 40 + "\n")
            f.write("1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation.\n")
            f.write("2. Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics\n")
            f.write("3. Custom PoreD^2 architecture developed for this project\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        self._log_message(f"Final report saved: {report_path}")
    
    def _save_results_json(self):
        """Save results in JSON format for further analysis"""
        json_path = os.path.join(self.log_dir, "training_results.json")
        
        # Prepare JSON-serializable data
        json_data = {
            'summary': {
                'start_time': self.training_summary['start_time'].isoformat(),
                'end_time': self.training_summary['end_time'].isoformat(),
                'total_training_time': self.training_summary['total_training_time'],
                'successful_trainings': self.training_summary['successful_trainings'],
                'failed_trainings': self.training_summary['failed_trainings'],
                'model_results': self.training_summary['model_results']
            },
            'detailed_results': {}
        }
        
        # Add detailed results (excluding non-serializable objects)
        for image_type in self.image_types:
            json_data['detailed_results'][image_type] = {}
            for model in self.models:
                result = self.training_results[image_type][model]
                
                # Extract serializable data
                serializable_result = {}
                for key, value in result.items():
                    if isinstance(value, (str, int, float, bool, list, dict)):
                        serializable_result[key] = value
                
                json_data['detailed_results'][image_type][model] = serializable_result
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        self._log_message(f"Results JSON saved: {json_path}")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
data_dir = os.path.join(project_root,'CODE','DL MODELS (copy)')

def main():
    """Main training orchestration function"""
    parser = argparse.ArgumentParser(description='Master Training Script for Pore Detection Models')
    parser.add_argument('--data_dir', type=str, default=os.path.join(data_dir,'Dataset'),
                       help='Root directory containing the dataset')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Base number of epochs for training')
    parser.add_argument('--unet_batch_size', type=int, default=8,
                       help='Batch size for U-Net training')
    parser.add_argument('--pored2_batch_size', type=int, default=4,
                       help='Batch size for PoreD^2 training')
    parser.add_argument('--yolo_batch_size', type=int, default=16,
                       help='Batch size for YOLO training')
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['unet', 'pored2', 'yolo'], 
                       default=['unet', 'pored2', 'yolo'],
                       help='Models to train')
    parser.add_argument('--image_types', type=str, nargs='+',
                       choices=['AFM', 'CONFOCAL', 'CRYO-SEM', 'STED'],
                       default=['AFM', 'CONFOCAL', 'CRYO-SEM', 'STED'],
                       help='Image types to train on')
    
    args = parser.parse_args()
    
    print("starting Master Training for Pore Detection Models")
    print("=" * 80)
    print(f"Data Directory: {args.data_dir}")
    print(f"Models to train: {args.models}")
    print(f"Image types: {args.image_types}")
    print(f"Base epochs: {args.epochs}")
    print("=" * 80)
    
    # Create master trainer
    master_trainer = MasterTrainer(
        data_dir=args.data_dir,
        base_epochs=args.epochs,
        unet_batch_size=args.unet_batch_size,
        pored2_batch_size=args.pored2_batch_size,
        yolo_batch_size=args.yolo_batch_size
    )
    
    # Override default image types and models if specified
    master_trainer.image_types = args.image_types
    master_trainer.models = args.models
    
    # Start comprehensive training
    final_results = master_trainer.train_all_models()
    
    print("training completed!")
    print("=" * 80)
    print(f"Log Directory: {final_results['log_directory']}")
    print(f"Total Time: {final_results['training_summary']['total_training_time']:.2f} seconds")
    print(f"Success Rate: {(final_results['training_summary']['successful_trainings']/(len(args.image_types) * len(args.models)))*100:.1f}%")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
