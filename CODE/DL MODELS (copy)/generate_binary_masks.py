"""
Binary Mask Generation Script for Pore Detection Models

This script processes images from a test directory and generates binary masks
using trained U-Net and PoreD² models. Each image will produce binary masks
from both models, saved as individual TIFF files.

The script uses the existing inference.py module to ensure consistent results
with the trained models.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import json
import traceback

sys.path.append('.')
from inference import PoreInferenceEngine


class BinaryMaskGenerator:
    """
    Generates binary masks for test images using trained models
    """
    
    def __init__(self, test_dir: str, output_dir: str = None, models: List[str] = None):
        self.test_dir = test_dir
        
        if output_dir is None:
            self.output_dir = f"binary_masks_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.models = models if models else ['unet', 'pored2']
        self.image_types = ['AFM', 'CONFOCAL', 'CRYO-SEM', 'STED']
        
        # Statistics tracking
        self.processing_stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'by_model': {},
            'by_type': {}
        }
        
        print(f"Binary Mask Generator Initialized")
        print(f"Test Directory: {self.test_dir}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Models: {self.models}")
        
    def discover_images(self) -> Dict[str, Dict[str, List[str]]]:
        structure = {}
        
        print("Discovering images...")
        
        for image_type in self.image_types:
            type_path = os.path.join(self.test_dir, image_type)
            structure[image_type] = {}
            
            if not os.path.exists(type_path):
                print(f"directory not found: {image_type}")
                continue
            
            # Check for direct images
            direct_images = []
            subdirectories = []
            
            for item in os.listdir(type_path):
                item_path = os.path.join(type_path, item)
                
                if os.path.isfile(item_path) and item.lower().endswith(('.tif', '.tiff')):
                    direct_images.append(item_path)
                elif os.path.isdir(item_path):
                    subdirectories.append(item)
            
            # Add direct images
            if direct_images:
                structure[image_type]['main'] = direct_images
                print(f"{image_type}/main: {len(direct_images)} images")
            
            # Process subdirectories
            for subdir in subdirectories:
                subdir_path = os.path.join(type_path, subdir)
                subdir_images = []
                
                for root, dirs, files in os.walk(subdir_path):
                    for file in files:
                        if file.lower().endswith(('.tif', '.tiff')):
                            subdir_images.append(os.path.join(root, file))
                
                if subdir_images:
                    structure[image_type][subdir] = subdir_images
                    print(f"{image_type}/{subdir}: {len(subdir_images)} images")
        
        return structure
    
    def generate_binary_mask(self, image_path: str, image_type_key: str, 
                           base_type: str, model: str) -> Dict:
        try:
            # Initialize inference engine with base type (not the combined key)
            engine = PoreInferenceEngine(model, base_type)
            
            # Create output filename
            image_filename = Path(image_path).stem
            safe_filename = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' 
                                   for c in image_filename)
            
            # Create subdirectory for this image type and model
            model_output_dir = os.path.join(self.output_dir, image_type_key, model.upper())
            os.makedirs(model_output_dir, exist_ok=True)
            
            output_filename = f"{safe_filename}_binary_mask.tif"
            output_path = os.path.join(model_output_dir, output_filename)
            
            # Process image using inference engine
            # Set output_path to None to prevent automatic saving
            result = engine.process_image(image_path, output_path=None)
            
            if result['success']:
                # Extract binary mask from result
                binary_mask = result['binary_mask']
                
                # Ensure binary mask is in correct format (0 for pores, 255 for background)
                if binary_mask.dtype != np.uint8:
                    binary_mask = binary_mask.astype(np.uint8)
                
                # Save binary mask as TIF
                success = cv2.imwrite(output_path, binary_mask)
                
                if not success:
                    raise Exception(f"Failed to save binary mask to {output_path}")
                
                return {
                    'success': True,
                    'input_path': image_path,
                    'output_path': output_path,
                    'model': model,
                    'image_type_key': image_type_key,
                    'base_type': base_type,
                    'pore_coverage': result['pore_statistics']['pore_percentage'],
                    'pore_pixels': result['pore_statistics']['pore_pixels'],
                    'total_pixels': result['pore_statistics']['total_pixels'],
                    'processing_time': result['timing']['total_time']
                }
            else:
                return {
                    'success': False,
                    'input_path': image_path,
                    'model': model,
                    'image_type_key': image_type_key,
                    'base_type': base_type,
                    'error': result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            error_msg = str(e)
            traceback_str = traceback.format_exc()
            print(f"exception details: {traceback_str}")
            
            return {
                'success': False,
                'input_path': image_path,
                'model': model,
                'image_type_key': image_type_key,
                'base_type': base_type,
                'error': error_msg,
                'traceback': traceback_str
            }
    
    def process_all_images(self):
        """
        Process all discovered images and generate binary masks
        """
        # Discover images
        structure = self.discover_images()
        
        # Calculate total images
        total_images = sum(len(images) for type_dict in structure.values() 
                          for images in type_dict.values())
        
        if total_images == 0:
            print("No images found in test directory!")
            return
        
        self.processing_stats['total_images'] = total_images
        total_operations = total_images * len(self.models)
        

        print(f"STARTING BINARY MASK GENERATION")
        print(f"Total images: {total_images}")
        print(f"Total operations: {total_operations} ({total_images} images × {len(self.models)} models)")
        
        current_operation = 0
        results_log = []
        
        # Process each image type
        for image_type, subdirs in structure.items():
            if not subdirs:
                continue
            
            for subdir_name, image_paths in subdirs.items():
                # Create combined key
                if subdir_name == 'main':
                    image_type_key = image_type
                else:
                    image_type_key = f"{image_type}_{subdir_name}"
                
                print(f"\n{'='*60}")
                print(f"Processing {image_type_key} ({len(image_paths)} images)")
                print(f"{'='*60}")
                
                # Initialize stats for this type
                if image_type_key not in self.processing_stats['by_type']:
                    self.processing_stats['by_type'][image_type_key] = {
                        'total': len(image_paths),
                        'successful': 0,
                        'failed': 0
                    }
                
                # Process each image with each model
                for image_path in image_paths:
                    image_name = os.path.basename(image_path)
                    
                    for model in self.models:
                        current_operation += 1
                        
                        print(f"\n[{current_operation}/{total_operations}] {image_name}")
                        print(f"  Model: {model.upper()}, Type: {image_type_key}")
                        
                        # Generate binary mask using the base image_type for model initialization
                        result = self.generate_binary_mask(
                            image_path, 
                            image_type_key,  # For output organization
                            image_type,      # Base type for model loading (AFM, CONFOCAL, CRYO-SEM, STED)
                            model
                        )
                        
                        # Update statistics
                        if model not in self.processing_stats['by_model']:
                            self.processing_stats['by_model'][model] = {
                                'successful': 0,
                                'failed': 0
                            }
                        
                        if result['success']:
                            print(f"coverage: {result['pore_coverage']:.2f}% | Pores: {result['pore_pixels']:,}/{result['total_pixels']:,} pixels")
                            print(f"time: {result['processing_time']:.3f}s | Saved: {os.path.basename(result['output_path'])}")
                            self.processing_stats['successful'] += 1
                            self.processing_stats['by_model'][model]['successful'] += 1
                            self.processing_stats['by_type'][image_type_key]['successful'] += 1
                        else:
                            print(f"failed: {result['error']}")
                            self.processing_stats['failed'] += 1
                            self.processing_stats['by_model'][model]['failed'] += 1
                            self.processing_stats['by_type'][image_type_key]['failed'] += 1
                        
                        results_log.append(result)
                        
                        # Progress update
                        progress = (current_operation / total_operations) * 100
                        print(f"progress: {progress:.1f}% ({current_operation}/{total_operations})")
        
        # Save results log
        self._save_results_log(results_log)
        
        # Generate summary report
        self._generate_summary_report()
        
        # Generate CSV summary
        self._generate_csv_summary(results_log)
        
        print(f"BINARY MASK GENERATION COMPLETED")
        print(f"Total Operations: {total_operations}")
        print(f"Successful: {self.processing_stats['successful']}")
        print(f"Failed: {self.processing_stats['failed']}")
        print(f"Success Rate: {(self.processing_stats['successful']/total_operations)*100:.1f}%")
        print(f"Output Directory: {self.output_dir}")
    
    def _save_results_log(self, results: List[Dict]):
        # Save detailed results log as JSON
        log_path = os.path.join(self.output_dir, "processing_log.json")
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_result = {
                'success': result['success'],
                'input_path': result['input_path'],
                'model': result['model'],
                'image_type_key': result.get('image_type_key', 'unknown'),
                'base_type': result.get('base_type', 'unknown')
            }
            
            if result['success']:
                serializable_result.update({
                    'output_path': result['output_path'],
                    'pore_coverage': float(result['pore_coverage']),
                    'pore_pixels': int(result['pore_pixels']),
                    'total_pixels': int(result['total_pixels']),
                    'processing_time': float(result['processing_time'])
                })
            else:
                serializable_result['error'] = result.get('error', 'Unknown error')
                if 'traceback' in result:
                    serializable_result['traceback'] = result['traceback']
            
            serializable_results.append(serializable_result)
        
        with open(log_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'test_directory': self.test_dir,
                'output_directory': self.output_dir,
                'models_used': self.models,
                'statistics': self.processing_stats,
                'results': serializable_results
            }, f, indent=2)
        
        print(f"Processing log saved: {log_path}")
    
    def _generate_summary_report(self):
        #generate a summary report
        report_path = os.path.join(self.output_dir, "summary_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("BINARY MASK GENERATION SUMMARY REPORT\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Directory: {self.test_dir}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            f.write("OVERALL STATISTICS\n")
            f.write(f"Total Images Processed: {self.processing_stats['total_images']}\n")
            f.write(f"Models Used: {', '.join([m.upper() for m in self.models])}\n")
            f.write(f"Total Operations: {self.processing_stats['total_images'] * len(self.models)}\n")
            f.write(f"Successful Operations: {self.processing_stats['successful']}\n")
            f.write(f"Failed Operations: {self.processing_stats['failed']}\n")
            total_ops = self.processing_stats['successful'] + self.processing_stats['failed']
            if total_ops > 0:
                f.write(f"Success Rate: {(self.processing_stats['successful']/total_ops)*100:.1f}%\n\n")
            
            f.write("RESULTS BY MODEL\n")
            for model, stats in self.processing_stats['by_model'].items():
                f.write(f"{model.upper()}:\n")
                f.write(f"  Successful: {stats['successful']}\n")
                f.write(f"  Failed: {stats['failed']}\n")
                total = stats['successful'] + stats['failed']
                if total > 0:
                    f.write(f"  Success Rate: {(stats['successful']/total)*100:.1f}%\n")
                f.write("\n")
            
            f.write("RESULTS BY IMAGE TYPE\n")
            for image_type, stats in self.processing_stats['by_type'].items():
                f.write(f"{image_type}:\n")
                f.write(f"  Total Images: {stats['total']}\n")
                f.write(f"  Successful Operations: {stats['successful']}\n")
                f.write(f"  Failed Operations: {stats['failed']}\n")
                expected_ops = stats['total'] * len(self.models)
                if expected_ops > 0:
                    f.write(f"  Success Rate: {(stats['successful']/expected_ops)*100:.1f}%\n")
                f.write("\n")
            
            f.write("OUTPUT STRUCTURE\n")
            f.write("Binary masks are organized as:\n")
            f.write("  output_dir/\n")
            f.write("    ├── AFM/\n")
            f.write("    │   ├── UNET/\n")
            f.write("    │   │   └── [image_name]_binary_mask.tif\n")
            f.write("    │   └── PORED2/\n")
            f.write("    │       └── [image_name]_binary_mask.tif\n")
            f.write("    ├── CRYO-SEM_PPMS/\n")
            f.write("    │   ├── UNET/\n")
            f.write("    │   └── PORED2/\n")
            f.write("    ├── CONFOCAL/\n")
            f.write("    ├── STED/\n")
            f.write("    ├── processing_log.json\n")
            f.write("    └── summary_report.txt\n\n")
            
            f.write("NOTES\n")
            f.write("- Binary mask format: Pores = BLACK (0), Background = WHITE (255)\n")
            f.write("- All masks are saved as grayscale TIFF files\n")
            f.write("- Models use trained weights specific to each image type\n")
            f.write("- Processing uses the inference.py module for consistency\n")
        
        print(f"Summary report saved: {report_path}")
    
    def _generate_csv_summary(self, results: List[Dict]):
        csv_path = os.path.join(self.output_dir, "results_summary.csv")
        
        try:
            with open(csv_path, 'w') as f:
                #write header
                f.write("Image_Name,Image_Type,Base_Type,Model,Success,Pore_Coverage_%,")
                f.write("Pore_Pixels,Total_Pixels,Processing_Time_sec,Output_Path,Error\n")
                
                #write data rows
                for result in results:
                    image_name = os.path.basename(result['input_path'])
                    image_type_key = result.get('image_type_key', 'unknown')
                    base_type = result.get('base_type', 'unknown')
                    model = result['model'].upper()
                    success = 'Yes' if result['success'] else 'No'
                    
                    if result['success']:
                        pore_coverage = f"{result['pore_coverage']:.2f}"
                        pore_pixels = result['pore_pixels']
                        total_pixels = result['total_pixels']
                        processing_time = f"{result['processing_time']:.3f}"
                        output_path = result['output_path']
                        error = ""
                    else:
                        pore_coverage = ""
                        pore_pixels = ""
                        total_pixels = ""
                        processing_time = ""
                        output_path = ""
                        error = result.get('error', 'Unknown error').replace(',', ';')
                    
                    f.write(f"{image_name},{image_type_key},{base_type},{model},{success},")
                    f.write(f"{pore_coverage},{pore_pixels},{total_pixels},{processing_time},")
                    f.write(f"{output_path},{error}\n")
            
            print(f"CSV summary saved: {csv_path}")
            
        except Exception as e:
            print(f"Failed to generate CSV summary: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate binary masks for test images using trained models'
    )
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save binary masks (optional)')
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['unet', 'pored2'], 
                       default=['unet', 'pored2'],
                       help='Models to use for generating masks')
    
    args = parser.parse_args()
    
    
    #validate test directory
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory '{args.test_dir}' does not exist!")
        return 1
    
    #create generator
    generator = BinaryMaskGenerator(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        models=args.models
    )
    
    #process all images
    generator.process_all_images()
    
    print("Binary mask generation completed successfully!")
    print(f"Check the output directory for results: {generator.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)