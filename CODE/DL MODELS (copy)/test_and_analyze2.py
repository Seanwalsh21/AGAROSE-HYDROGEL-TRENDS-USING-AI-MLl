"""
Comprehensive Analysis and Testing Script for Pore Detection Models - Updated Version

This script provides comprehensive testing and analysis capabilities for all trained models
with support for analyzing different image types and subdirectories.

It includes:

1. Model performance comparison by image type and subdirectory
2. Batch inference on test images
3. Statistical analysis of results
4. Visual comparison of outputs
5. Timing benchmarks
6. Detailed reporting
"""

import os
import time
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime
import json

# Import our modules
import sys
sys.path.append('.')
from inference import PoreInferenceEngine


class PoreDetectionAnalyzer:
    """
    Comprehensive analyzer for pore detection models with flexible directory structure
    
    This class provides tools for comparing model performance,
    analyzing results by image type and subdirectory, and generating detailed reports.
    """
    
    def __init__(self, test_dir: str, models: List[str]):
        """
        Initialize the analyzer
        
        Args:
            test_dir: Directory containing test images
            models: List of models to test ('unet', 'pored2', 'yolo')
        """
        
        self.test_dir = test_dir
        self.models = models
        
        # Define image types and their potential subdirectories
        self.image_types = ['AFM', 'CONFOCAL', 'CRYO-SEM', 'STED']
        
        # Results storage
        self.test_results = {}
        self.performance_metrics = {}
        self.timing_data = {}
        self.discovered_structure = {}
        
        # Create output directory
        self.output_dir = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Pore Detection Analyzer Initialized")
        print(f"Test Directory: {self.test_dir}")
        print(f"Models: {self.models}")
        print(f"Image Types: {self.image_types}")
        print(f"Output Directory: {self.output_dir}")
    
    def discover_directory_structure(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Discover the actual directory structure and categorize images
        
        Returns:
            Dictionary mapping image types to subdirectories and their images
        """
        structure = {}
        
        for image_type in self.image_types:
            type_path = os.path.join(self.test_dir, image_type)
            structure[image_type] = {}
            
            if not os.path.exists(type_path):
                print(f"Directory not found: {type_path}")
                continue
            
            # Check if there are direct images in the type directory
            direct_images = []
            subdirectories = []
            
            for item in os.listdir(type_path):
                item_path = os.path.join(type_path, item)
                
                if os.path.isfile(item_path) and item.lower().endswith(('.tif', '.tiff')):
                    direct_images.append(item_path)
                elif os.path.isdir(item_path):
                    subdirectories.append(item)
            
            # If there are direct images, create a 'main' category
            if direct_images:
                structure[image_type]['main'] = direct_images
                print(f"Found {len(direct_images)} direct images in {image_type}")
            
            # Process subdirectories
            for subdir in subdirectories:
                subdir_path = os.path.join(type_path, subdir)
                subdir_images = []
                
                # Find all .tif files in this subdirectory (including nested)
                for root, dirs, files in os.walk(subdir_path):
                    for file in files:
                        if file.lower().endswith(('.tif', '.tiff')):
                            subdir_images.append(os.path.join(root, file))
                
                if subdir_images:
                    structure[image_type][subdir] = subdir_images
                    print(f"Found {len(subdir_images)} images in {image_type}/{subdir}")
        
        self.discovered_structure = structure
        return structure
    
    def find_test_images(self) -> Dict[str, List[str]]:
        """
        Find test images based on discovered directory structure
        
        Returns:
            Dictionary mapping combined image type keys to lists of image paths
        """
        structure = self.discover_directory_structure()
        test_images = {}
        
        for image_type, subdirs in structure.items():
            if not subdirs:
                continue
            
            for subdir_name, image_paths in subdirs.items():
                # Create combined key for image type and subdirectory
                if subdir_name == 'main':
                    combined_key = image_type
                else:
                    combined_key = f"{image_type}_{subdir_name}"
                
                test_images[combined_key] = image_paths
                print(f"Prepared {len(image_paths)} test images for {combined_key}")
        
        return test_images
    
    def run_comprehensive_analysis(self) -> Dict:
        """
        Run comprehensive analysis across all models and image types
        
        Returns:
            Complete analysis results
        """
        start_time = time.time()
        
        # Find test images
        test_images = self.find_test_images()
        
        if not test_images:
            print("No test images found in the specified directory structure!")
            return {}
        
        # Get all unique image type keys
        all_image_type_keys = list(test_images.keys())
        
        # Initialize results structure
        for image_type_key in all_image_type_keys:
            self.test_results[image_type_key] = {}
            self.performance_metrics[image_type_key] = {}
            self.timing_data[image_type_key] = {}
            
            for model in self.models:
                self.test_results[image_type_key][model] = []
                self.timing_data[image_type_key][model] = []
        
        # Run analysis for each combination
        total_images = sum(len(images) for images in test_images.values())
        print(f"Total images to process: {total_images}")
        print(f"Total combinations: {total_images * len(self.models)}")
        
        current_combination = 0
        
        for image_type_key in all_image_type_keys:
            image_paths = test_images[image_type_key]
            
            if not image_paths:
                print(f"No test images found for {image_type_key}, skipping...")
                continue
            
            # Extract base type for inference engine
            base_type = image_type_key.split('_')[0]
            
            print(f"\n{'='*60}")
            print(f"ANALYZING {image_type_key} IMAGES")
            print(f"Base Type: {base_type}")
            print(f"Number of Images: {len(image_paths)}")
            print(f"{'='*60}")
            
            # Limit to first 5 images per type for demonstration
            sample_images = image_paths[:5]
            
            for model in self.models:
                print(f"\nTesting {model.upper()} on {image_type_key} images...")
                
                # Initialize inference engine with base type
                try:
                    engine = PoreInferenceEngine(model, base_type)
                    
                    model_results = []
                    model_timings = []
                    
                    for i, image_path in enumerate(sample_images):
                        current_combination += 1
                        print(f"[{current_combination}/{len(sample_images)*len(self.models)*len(all_image_type_keys)}] Processing {os.path.basename(image_path)} with {model.upper()}")
                        
                        # Run inference
                        result = engine.process_image(
                            image_path, 
                            output_path=os.path.join(self.output_dir, f"{model}_{image_type_key}_{i}")
                        )
                        
                        if result['success']:
                            # Add type information to result
                            result['image_type_key'] = image_type_key
                            result['base_type'] = base_type
                            
                            model_results.append(result)
                            model_timings.append(result['timing']['total_time'])
                            print(f"  Success: {result['pore_statistics']['pore_percentage']:.1f}% coverage in {result['timing']['total_time']:.3f}s")
                        else:
                            print(f"  Failed: {result['error']}")
                    
                    # Store results
                    self.test_results[image_type_key][model] = model_results
                    self.timing_data[image_type_key][model] = model_timings
                    
                    # Calculate performance metrics
                    self._calculate_performance_metrics(image_type_key, model, model_results)
                    
                    print(f"  Completed {len(model_results)}/{len(sample_images)} images for {model.upper()}")
                    
                except Exception as e:
                    print(f"  Error initializing {model} for {image_type_key}: {e}")
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        self._generate_comparison_report(total_time, all_image_type_keys)
        
        # Create visualizations
        self._create_visualizations(all_image_type_keys)
        
        # Save results
        self._save_analysis_results(all_image_type_keys)
        

        print("COMPLETE ANALYSIS COMPLETED")
        print(f"Total Analysis Time: {total_time:.2f} seconds")
        print(f"Results saved to: {self.output_dir}")
        
        return {
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'timing_data': self.timing_data,
            'total_time': total_time,
            'output_dir': self.output_dir,
            'discovered_structure': self.discovered_structure
        }
    
    def _calculate_performance_metrics(self, image_type_key: str, model: str, results: List[Dict]):
        """Calculate performance metrics for a model on an image type"""
        if not results:
            self.performance_metrics[image_type_key][model] = {
                'avg_processing_time': 0,
                'avg_pore_coverage': 0,
                'std_pore_coverage': 0,
                'successful_inferences': 0,
                'total_inferences': 0
            }
            return
        
        # Extract metrics
        processing_times = [r['timing']['total_time'] for r in results]
        pore_coverages = [r['pore_statistics']['pore_percentage'] for r in results]
        
        metrics = {
            'avg_processing_time': np.mean(processing_times),
            'std_processing_time': np.std(processing_times),
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times),
            'avg_pore_coverage': np.mean(pore_coverages),
            'std_pore_coverage': np.std(pore_coverages),
            'min_pore_coverage': np.min(pore_coverages),
            'max_pore_coverage': np.max(pore_coverages),
            'successful_inferences': len(results),
            'total_inferences': len(results)
        }
        
        self.performance_metrics[image_type_key][model] = metrics
    
    def _generate_comparison_report(self, total_time: float, all_image_type_keys: List[str]):
        """Generate comprehensive comparison report"""
        report_path = os.path.join(self.output_dir, "comprehensive_analysis_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE PORE DETECTION MODEL ANALYSIS REPORT\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Analysis Time: {total_time:.2f} seconds\n")
            f.write(f"Models Tested: {', '.join(self.models)}\n")
            f.write(f"Test Directory: {self.test_dir}\n")
            f.write(f"Total Image Type Categories: {len(all_image_type_keys)}\n\n")
            
            f.write("DISCOVERED DIRECTORY STRUCTURE\n")
            f.write("-" * 35 + "\n")
            for image_type, subdirs in self.discovered_structure.items():
                f.write(f"{image_type}:\n")
                for subdir_name, images in subdirs.items():
                    f.write(f"  {subdir_name}: {len(images)} images\n")
                f.write("\n")
            
            f.write("PERFORMANCE COMPARISON BY MODEL\n")
            
            for model in self.models:
                f.write(f"{model.upper()} MODEL PERFORMANCE:\n")
                f.write("  " + "-" * 30 + "\n")
                
                model_avg_times = []
                model_avg_coverages = []
                
                for image_type_key in all_image_type_keys:
                    if image_type_key in self.performance_metrics and model in self.performance_metrics[image_type_key]:
                        metrics = self.performance_metrics[image_type_key][model]
                        
                        f.write(f"  {image_type_key}:\n")
                        f.write(f"    Processing Time: {metrics['avg_processing_time']:.4f} ± {metrics.get('std_processing_time', 0):.4f} seconds\n")
                        f.write(f"    Pore Coverage: {metrics['avg_pore_coverage']:.2f} ± {metrics.get('std_pore_coverage', 0):.2f}%\n")
                        f.write(f"    Success Rate: {metrics['successful_inferences']}/{metrics['total_inferences']}\n\n")
                        
                        model_avg_times.append(metrics['avg_processing_time'])
                        model_avg_coverages.append(metrics['avg_pore_coverage'])
                
                if model_avg_times:
                    f.write(f"  Overall Average Processing Time: {np.mean(model_avg_times):.4f} seconds\n")
                    f.write(f"  Overall Average Pore Coverage: {np.mean(model_avg_coverages):.2f}%\n\n")
            
            f.write("PERFORMANCE COMPARISON BY IMAGE TYPE\n")
            f.write("-" * 45 + "\n\n")
            
            for image_type_key in all_image_type_keys:
                f.write(f"{image_type_key}:\n")
                
                for model in self.models:
                    if image_type_key in self.performance_metrics and model in self.performance_metrics[image_type_key]:
                        metrics = self.performance_metrics[image_type_key][model]
                        f.write(f"  {model.upper()}: ")
                        f.write(f"Time: {metrics['avg_processing_time']:.4f}s, ")
                        f.write(f"Coverage: {metrics['avg_pore_coverage']:.2f}%, ")
                        f.write(f"Success: {metrics['successful_inferences']}/{metrics['total_inferences']}\n")
                    else:
                        f.write(f"  {model.upper()}: No data\n")
                
                f.write("\n")
            
            # Model rankings
            f.write("MODEL COMPARISON SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            model_rankings = self._calculate_model_rankings(all_image_type_keys)
            
            if model_rankings['speed']:
                f.write("Speed Ranking (fastest to slowest):\n")
                for i, (model, avg_time) in enumerate(model_rankings['speed'], 1):
                    f.write(f"  {i}. {model.upper()}: {avg_time:.4f} seconds\n")
            
            if model_rankings['coverage']:
                f.write("\nPore Detection Ranking (highest to lowest coverage):\n")
                for i, (model, avg_coverage) in enumerate(model_rankings['coverage'], 1):
                    f.write(f"  {i}. {model.upper()}: {avg_coverage:.2f}%\n")
            
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            self._write_recommendations(f, model_rankings)
        
        print(f"Comprehensive report saved: {report_path}")
    
    def _calculate_model_rankings(self, all_image_type_keys: List[str]) -> Dict:
        """Calculate model rankings based on performance metrics"""
        model_avg_times = {model: [] for model in self.models}
        model_avg_coverages = {model: [] for model in self.models}
        
        for image_type_key in all_image_type_keys:
            for model in self.models:
                if image_type_key in self.performance_metrics and model in self.performance_metrics[image_type_key]:
                    metrics = self.performance_metrics[image_type_key][model]
                    model_avg_times[model].append(metrics['avg_processing_time'])
                    model_avg_coverages[model].append(metrics['avg_pore_coverage'])
        
        # Calculate overall averages
        speed_ranking = []
        coverage_ranking = []
        
        for model in self.models:
            if model_avg_times[model]:
                avg_time = np.mean(model_avg_times[model])
                avg_coverage = np.mean(model_avg_coverages[model])
                speed_ranking.append((model, avg_time))
                coverage_ranking.append((model, avg_coverage))
        
        # Sort rankings
        speed_ranking.sort(key=lambda x: x[1])  # Faster is better
        coverage_ranking.sort(key=lambda x: x[1], reverse=True)  # Higher coverage is better
        
        return {
            'speed': speed_ranking,
            'coverage': coverage_ranking
        }
    
    def _write_recommendations(self, f, model_rankings: Dict):
        """Write recommendations based on analysis"""
        f.write("Based on the comprehensive analysis:\n\n")
        
        if model_rankings['speed']:
            fastest_model = model_rankings['speed'][0][0]
            f.write(f"For SPEED: {fastest_model.upper()} is the fastest model\n")
            f.write(f"  Best for real-time applications\n")
            f.write(f"  Suitable for high-throughput analysis\n\n")
        
        if model_rankings['coverage']:
            best_detection_model = model_rankings['coverage'][0][0]
            f.write(f"For ACCURACY: {best_detection_model.upper()} provides highest pore coverage detection\n")
            f.write(f"  Best for detailed pore analysis\n")
            f.write(f"  Suitable for research applications\n\n")
        
        f.write("IMAGE TYPE SPECIFIC RECOMMENDATIONS:\n")
        f.write("  Different image types may favor different models\n")
        f.write("  Consider the specific image type of your application\n")
        f.write("  Subdirectories may show varying performance patterns\n\n")
        
        f.write("MODEL-SPECIFIC RECOMMENDATIONS:\n")
        f.write("  U-Net: Best for semantic segmentation, good balance of speed and accuracy\n")
        f.write("  PoreD²: Advanced features, good for detailed analysis with statistics\n")
        f.write("  YOLO: Object detection approach, good for counting individual pores\n\n")
        
        f.write("DIRECTORY STRUCTURE CONSIDERATIONS:\n")
        f.write("  The flexible structure allows for different organization methods\n")
        f.write("  Results may vary between subdirectories of the same image type\n")
        f.write("  Consider organizing images by specific experimental conditions\n")
    
    def _create_visualizations(self, all_image_type_keys: List[str]):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Processing time comparison
        self._plot_processing_times(all_image_type_keys)
        
        # 2. Pore coverage comparison
        self._plot_pore_coverage(all_image_type_keys)
        
        # 3. Model performance heatmap
        self._plot_performance_heatmap(all_image_type_keys)
        
        # 4. Sample results comparison
        self._create_sample_comparison(all_image_type_keys)
        
        print("Visualizations created")
    
    def _plot_processing_times(self, all_image_type_keys: List[str]):
        """Create processing time comparison plot"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Prepare data
        times_data = []
        for image_type_key in all_image_type_keys:
            for model in self.models:
                if image_type_key in self.timing_data and model in self.timing_data[image_type_key]:
                    for time_val in self.timing_data[image_type_key][model]:
                        times_data.append({
                            'Model': model.upper(),
                            'Image_Type': image_type_key,
                            'Processing_Time': time_val
                        })
        
        if times_data:
            df_times = pd.DataFrame(times_data)
            sns.boxplot(data=df_times, x='Image_Type', y='Processing_Time', hue='Model', ax=ax)
            ax.set_title('Processing Time Distribution by Image Type')
            ax.set_ylabel('Processing Time (seconds)')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'processing_times_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pore_coverage(self, all_image_type_keys: List[str]):
        """Create pore coverage comparison plot"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Prepare data
        coverage_data = []
        for image_type_key in all_image_type_keys:
            for model in self.models:
                if image_type_key in self.test_results and model in self.test_results[image_type_key]:
                    for result in self.test_results[image_type_key][model]:
                        coverage_data.append({
                            'Model': model.upper(),
                            'Image_Type': image_type_key,
                            'Pore_Coverage': result['pore_statistics']['pore_percentage']
                        })
        
        if coverage_data:
            df_coverage = pd.DataFrame(coverage_data)
            sns.barplot(data=df_coverage, x='Image_Type', y='Pore_Coverage', hue='Model', ax=ax)
            ax.set_title('Average Pore Coverage by Image Type')
            ax.set_ylabel('Pore Coverage (%)')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pore_coverage_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_heatmap(self, all_image_type_keys: List[str]):
        """Create performance heatmap"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Create matrices
        n_models = len(self.models)
        n_types = len(all_image_type_keys)
        
        time_matrix = np.zeros((n_models, n_types))
        coverage_matrix = np.zeros((n_models, n_types))
        
        model_labels = [m.upper() for m in self.models]
        type_labels = [key.replace('_', '\n') for key in all_image_type_keys]
        
        for j, image_type_key in enumerate(all_image_type_keys):
            for i, model in enumerate(self.models):
                if image_type_key in self.performance_metrics and model in self.performance_metrics[image_type_key]:
                    metrics = self.performance_metrics[image_type_key][model]
                    time_matrix[i, j] = metrics['avg_processing_time']
                    coverage_matrix[i, j] = metrics['avg_pore_coverage']
        
        # Time heatmap
        sns.heatmap(time_matrix, 
                   xticklabels=type_labels,
                   yticklabels=model_labels,
                   annot=True, fmt='.3f', cmap='YlOrRd',
                   ax=axes[0])
        axes[0].set_title('Average Processing Time (seconds)')
        
        # Coverage heatmap
        sns.heatmap(coverage_matrix,
                   xticklabels=type_labels,
                   yticklabels=model_labels,
                   annot=True, fmt='.1f', cmap='YlGnBu',
                   ax=axes[1])
        axes[1].set_title('Average Pore Coverage (%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_sample_comparison(self, all_image_type_keys: List[str]):
        """Create sample comparison of model outputs"""
        print("Creating sample comparison visualization...")
        
        # Find a representative image type with data
        for image_type_key in all_image_type_keys:
            if image_type_key in self.test_results and self.test_results[image_type_key]:
                # Get the first successful result for each model
                sample_results = {}
                for model in self.models:
                    if (model in self.test_results[image_type_key] and 
                        self.test_results[image_type_key][model]):
                        sample_results[model] = self.test_results[image_type_key][model][0]
                
                if len(sample_results) >= 2:  # Need at least 2 models for comparison
                    self._create_side_by_side_comparison(image_type_key, sample_results)
                    break  # Only create one sample comparison
    
    def _create_side_by_side_comparison(self, image_type_key: str, sample_results: Dict):
        """Create side-by-side comparison for a specific image type"""
        num_models = len(sample_results)
        
        fig, axes = plt.subplots(2, num_models, figsize=(5*num_models, 10))
        
        if num_models == 1:
            axes = axes.reshape(2, 1)
        
        for i, (model, result) in enumerate(sample_results.items()):
            # Load original image
            original_path = result['input_path']
            original_image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
            
            # Get binary mask
            binary_mask = result['binary_mask']
            
            # Plot original image
            axes[0, i].imshow(original_image, cmap='gray')
            axes[0, i].set_title(f'Original {image_type_key} Image')
            axes[0, i].axis('off')
            
            # Plot binary mask
            axes[1, i].imshow(binary_mask, cmap='gray')
            axes[1, i].set_title(f'{model.upper()} Result\n'
                               f'Coverage: {result["pore_statistics"]["pore_percentage"]:.1f}%\n'
                               f'Time: {result["timing"]["total_time"]:.3f}s')
            axes[1, i].axis('off')
        
        plt.suptitle(f'Model Comparison - {image_type_key} Images', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'sample_comparison_{image_type_key.lower().replace(" ", "_")}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_analysis_results(self, all_image_type_keys: List[str]):
        """Save analysis results in JSON format"""
        results_path = os.path.join(self.output_dir, "analysis_results.json")
        
        # Prepare serializable data
        serializable_results = {
            'performance_metrics': self.performance_metrics,
            'timing_summary': {},
            'discovered_structure': self.discovered_structure,
            'analysis_metadata': {
                'models_tested': self.models,
                'image_type_keys_tested': all_image_type_keys,
                'analysis_date': datetime.now().isoformat(),
                'test_directory': self.test_dir
            }
        }
        
        # Add timing summary
        for image_type_key in all_image_type_keys:
            serializable_results['timing_summary'][image_type_key] = {}
            for model in self.models:
                if image_type_key in self.timing_data and model in self.timing_data[image_type_key]:
                    timings = self.timing_data[image_type_key][model]
                    if timings:
                        serializable_results['timing_summary'][image_type_key][model] = {
                            'mean': float(np.mean(timings)),
                            'std': float(np.std(timings)),
                            'min': float(np.min(timings)),
                            'max': float(np.max(timings)),
                            'count': len(timings)
                        }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Analysis results saved: {results_path}")
        
        # Also save a CSV summary for easy analysis
        self._save_csv_summary(all_image_type_keys)
    
    def _save_csv_summary(self, all_image_type_keys: List[str]):
        """Save a CSV summary of all results for easy analysis"""
        csv_path = os.path.join(self.output_dir, "performance_summary.csv")
        
        # Prepare data for CSV
        csv_data = []
        
        for image_type_key in all_image_type_keys:
            # Split the key to get base type and subdirectory
            parts = image_type_key.split('_', 1)
            base_type = parts[0]
            subdirectory = parts[1] if len(parts) > 1 else 'main'
            
            for model in self.models:
                if (image_type_key in self.performance_metrics and 
                    model in self.performance_metrics[image_type_key]):
                    metrics = self.performance_metrics[image_type_key][model]
                    
                    csv_data.append({
                        'Base_Type': base_type,
                        'Subdirectory': subdirectory,
                        'Full_Key': image_type_key,
                        'Model': model.upper(),
                        'Avg_Processing_Time': metrics['avg_processing_time'],
                        'Std_Processing_Time': metrics.get('std_processing_time', 0),
                        'Avg_Pore_Coverage': metrics['avg_pore_coverage'],
                        'Std_Pore_Coverage': metrics.get('std_pore_coverage', 0),
                        'Min_Pore_Coverage': metrics.get('min_pore_coverage', 0),
                        'Max_Pore_Coverage': metrics.get('max_pore_coverage', 0),
                        'Successful_Inferences': metrics['successful_inferences'],
                        'Total_Inferences': metrics['total_inferences'],
                        'Success_Rate': metrics['successful_inferences'] / max(metrics['total_inferences'], 1) * 100
                    })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            print(f"CSV summary saved: {csv_path}")
        else:
            print("No data available for CSV summary")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
dat_dir = os.path.join(project_root,'CODE','DL MODELS (copy)')

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Comprehensive Pore Detection Model Analysis')
    parser.add_argument('--test_dir', type=str, default=os.path.join(dat_dir,'Dataset'),
                       help='Directory containing test images')
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['unet', 'pored2', 'yolo'], 
                       default=['unet', 'pored2', 'yolo'],
                       help='Models to test and compare')
    
    args = parser.parse_args()
    
    print("Starting Comprehensive Pore Detection Analysis")
    print(f"Test Directory: {args.test_dir}")
    print(f"Models: {args.models}")
    print("\nExpected Directory Structure:")
    print("  /test/AFM/*.tif (direct images)")
    print("  /test/AFM/[subdirs]/*.tif (images in subdirectories)")
    print("  /test/CONFOCAL/*.tif")
    print("  /test/CRYO-SEM/[subdirs]/*.tif")
    print("  /test/STED/*.tif")
    
    # Validate test directory
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory '{args.test_dir}' does not exist!")
        print("Please ensure the directory path is correct.")
        return 1
    
    # Create analyzer
    analyzer = PoreDetectionAnalyzer(
        test_dir=args.test_dir,
        models=args.models
    )
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    if not results:
        print("analysis failed - no results generated")
        return 1
    
    print("\nANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"Results Directory: {results['output_dir']}")
    print(f"Total Analysis Time: {results['total_time']:.2f} seconds")
    print("\nGenerated Files:")
    print("  comprehensive_analysis_report.txt - Detailed text report")
    print("  performance_summary.csv - CSV data for further analysis")
    print("  Multiple visualization PNG files")
    print("  analysis_results.json - Complete results in JSON format")
    print("Discovered Structure:")
    for image_type, subdirs in results['discovered_structure'].items():
        print(f"  {image_type}:")
        for subdir_name, images in subdirs.items():
            print(f"    {subdir_name}: {len(images)} images")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)