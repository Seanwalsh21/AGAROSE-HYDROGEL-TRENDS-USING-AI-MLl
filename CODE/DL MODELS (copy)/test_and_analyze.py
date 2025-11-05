"""
Comprehensive Analysis and Testing Script for Pore Detection Models

This script provides comprehensive testing and analysis capabilities for all trained models.
It includes:

1. Model performance comparison
2. Batch inference on test images
3. Statistical analysis of results
4. Visual comparison of outputs
5. Timing benchmarks
6. Detailed reporting

Usage:
python test_and_analyze.py --test_dir "path/to/test/images" --models unet pored2 yolo
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
    complete analyzer for pore detection models
    
    This class provides tools for comparing model performance,
    analyzing results, and generating detailed reports.
    """
    
    def __init__(self, test_dir: str, models: List[str], image_types: List[str]):
        """
        Initialize the analyzer
        
        Args:
            test_dir: Directory containing test images
            models: List of models to test ('unet', 'pored2', 'yolo')
            image_types: List of image types to test
        """
        
        self.test_dir = test_dir
        self.models = models
        self.image_types = image_types
        
        # Results storage
        self.test_results = {}
        self.performance_metrics = {}
        self.timing_data = {}
        
        # Create output directory
        self.output_dir = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Pore Detection Analyzer Initialized")
        print(f"Test Directory: {self.test_dir}")
        print(f"Models: {self.models}")
        print(f"Image Types: {self.image_types}")
        print(f"Output Directory: {self.output_dir}")
    
    def find_test_images(self) -> Dict[str, List[str]]:
        """
        Find test images for each image type
        
        Returns:
            Dictionary mapping image types to lists of image paths
        """
        test_images = {image_type: [] for image_type in self.image_types}
        
        # Map image types to their folder names
        folder_mapping = {
            'AFM': 'AFM folder/AFM',
            'CONFOCAL': 'Confocal folder/CONFOCAL',
            'CRYO-SEM': 'Cryo-sem Folder/CRYO-SEM',
            'STED': 'STED Folder/STED'
        }
        
        for image_type in self.image_types:
            folder_path = os.path.join(self.test_dir, folder_mapping[image_type])
            
            if os.path.exists(folder_path):
                # Find all .tif files recursively
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        if file.lower().endswith(('.tif', '.tiff')):
                            test_images[image_type].append(os.path.join(root, file))
            
            print(f"Found {len(test_images[image_type])} test images for {image_type}")
        
        return test_images
    
    def run_comprehensive_analysis(self) -> Dict:
        """
        Run comprehensive analysis across all models and image types
        
        Returns:
            Complete analysis results
        """
        print(f"\n{'='*80}")
        print("STARTING COMPREHENSIVE PORE DETECTION ANALYSIS")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Find test images
        test_images = self.find_test_images()
        
        # Initialize results structure
        for image_type in self.image_types:
            self.test_results[image_type] = {}
            self.performance_metrics[image_type] = {}
            self.timing_data[image_type] = {}
            
            for model in self.models:
                self.test_results[image_type][model] = []
                self.timing_data[image_type][model] = []
        
        # Run analysis for each combination
        total_combinations = sum(len(images) for images in test_images.values()) * len(self.models)
        current_combination = 0
        
        for image_type in self.image_types:
            if not test_images[image_type]:
                print(f"No test images found for {image_type}, skipping...")
                continue
            
            print(f"ANALYZING {image_type} IMAGES")
            
            # Limit to first 5 images per type for demonstration
            sample_images = test_images[image_type][:5]
            
            for model in self.models:
                print(f"\nTesting {model.upper()} on {image_type} images...")
                
                # Initialize inference engine
                try:
                    engine = PoreInferenceEngine(model, image_type)
                    
                    model_results = []
                    model_timings = []
                    
                    for i, image_path in enumerate(sample_images):
                        current_combination += 1
                        print(f"[{current_combination}/{len(sample_images)*len(self.models)*len(self.image_types)}] Processing {os.path.basename(image_path)} with {model.upper()}")
                        
                        # Run inference
                        result = engine.process_image(
                            image_path, 
                            output_path=os.path.join(self.output_dir, f"{model}_{image_type}_{i}")
                        )
                        
                        if result['success']:
                            model_results.append(result)
                            model_timings.append(result['timing']['total_time'])
                        else:
                            print(f"failed: {result['error']}")
                    
                    # Store results
                    self.test_results[image_type][model] = model_results
                    self.timing_data[image_type][model] = model_timings
                    
                    # Calculate performance metrics
                    self._calculate_performance_metrics(image_type, model, model_results)
                    
                except Exception as e:
                    print(f"error initializing {model} for {image_type}: {e}")
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        self._generate_comparison_report(total_time)
        
        # Create visualizations
        self._create_visualizations()
        
        # Save results
        self._save_analysis_results()
        

        print("COMPREHENSIVE ANALYSIS COMPLETED")
        print(f"total Analysis Time: {total_time:.2f} seconds")
        print(f"results saved to: {self.output_dir}")
        
        return {
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'timing_data': self.timing_data,
            'total_time': total_time,
            'output_dir': self.output_dir
        }
    
    def _calculate_performance_metrics(self, image_type: str, model: str, results: List[Dict]):
        # calculate performance metrics for a model on an image type
        if not results:
            self.performance_metrics[image_type][model] = {
                'avg_processing_time': 0,
                'avg_pore_coverage': 0,
                'std_pore_coverage': 0,
                'successful_inferences': 0,
                'total_inferences': 0
            }
            return
        
        #extract metrics
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
        
        self.performance_metrics[image_type][model] = metrics
    
    def _generate_comparison_report(self, total_time: float):
        #generate comprehensive comparison report"""
        report_path = os.path.join(self.output_dir, "comprehensive_analysis_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE PORE DETECTION MODEL ANALYSIS REPORT\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Analysis Time: {total_time:.2f} seconds\n")
            f.write(f"Models Tested: {', '.join(self.models)}\n")
            f.write(f"Image Types: {', '.join(self.image_types)}\n\n")
            
            f.write("PERFORMANCE COMPARISON BY MODEL\n")
            f.write("-" * 40 + "\n\n")
            
            for model in self.models:
                f.write(f"{model.upper()} MODEL PERFORMANCE:\n")
                f.write("  " + "-" * 30 + "\n")
                
                model_avg_times = []
                model_avg_coverages = []
                
                for image_type in self.image_types:
                    if model in self.performance_metrics.get(image_type, {}):
                        metrics = self.performance_metrics[image_type][model]
                        
                        f.write(f"  {image_type}:\n")
                        f.write(f"    Processing Time: {metrics['avg_processing_time']:.4f} ± {metrics.get('std_processing_time', 0):.4f} seconds\n")
                        f.write(f"    Pore Coverage: {metrics['avg_pore_coverage']:.2f} ± {metrics.get('std_pore_coverage', 0):.2f}%\n")
                        f.write(f"    Successful Inferences: {metrics['successful_inferences']}/{metrics['total_inferences']}\n\n")
                        
                        model_avg_times.append(metrics['avg_processing_time'])
                        model_avg_coverages.append(metrics['avg_pore_coverage'])
                
                if model_avg_times:
                    f.write(f"  Overall Average Processing Time: {np.mean(model_avg_times):.4f} seconds\n")
                    f.write(f"  Overall Average Pore Coverage: {np.mean(model_avg_coverages):.2f}%\n\n")
            
            f.write("PERFORMANCE COMPARISON BY IMAGE TYPE\n")
            f.write("-" * 40 + "\n\n")
            
            for image_type in self.image_types:
                f.write(f"{image_type} IMAGING RESULTS:\n")
                f.write("  " + "-" * 30 + "\n")
                
                for model in self.models:
                    if model in self.performance_metrics.get(image_type, {}):
                        metrics = self.performance_metrics[image_type][model]
                        f.write(f"  {model.upper()}:\n")
                        f.write(f"    Time: {metrics['avg_processing_time']:.4f}s, ")
                        f.write(f"Coverage: {metrics['avg_pore_coverage']:.2f}%, ")
                        f.write(f"Success: {metrics['successful_inferences']}/{metrics['total_inferences']}\n")
                
                f.write("\n")
            
            f.write("MODEL COMPARISON SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            # Calculate overall model rankings
            model_rankings = self._calculate_model_rankings()
            
            f.write("Speed Ranking (fastest to slowest):\n")
            for i, (model, avg_time) in enumerate(model_rankings['speed'], 1):
                f.write(f"  {i}. {model.upper()}: {avg_time:.4f} seconds\n")
            
            f.write("\nPore Detection Ranking (highest to lowest coverage):\n")
            for i, (model, avg_coverage) in enumerate(model_rankings['coverage'], 1):
                f.write(f"  {i}. {model.upper()}: {avg_coverage:.2f}%\n")
            
            f.write("\nRECOMMendations:\n")
            f.write("-" * 20 + "\n")
            self._write_recommendations(f, model_rankings)
        
        print(f"complete report saved: {report_path}")
    
    def _calculate_model_rankings(self) -> Dict:
        """Calculate model rankings based on performance metrics"""
        model_avg_times = {model: [] for model in self.models}
        model_avg_coverages = {model: [] for model in self.models}
        
        for image_type in self.image_types:
            for model in self.models:
                if model in self.performance_metrics.get(image_type, {}):
                    metrics = self.performance_metrics[image_type][model]
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
        # Write recommendations based on analysis"""
        f.write("Based on the comprehensive analysis:\n\n")
        
        if model_rankings['speed']:
            fastest_model = model_rankings['speed'][0][0]
            f.write(f"• For SPEED: {fastest_model.upper()} is the fastest model\n")
            f.write(f"  - Best for real-time applications\n")
            f.write(f"  - Suitable for high-throughput analysis\n\n")
        
        if model_rankings['coverage']:
            best_detection_model = model_rankings['coverage'][0][0]
            f.write(f"• For ACCURACY: {best_detection_model.upper()} provides highest pore coverage detection\n")
            f.write(f"  - Best for detailed pore analysis\n")
            f.write(f"  - Suitable for research applications\n\n")
        
        f.write("MODEL-SPECIFIC RECOMMENDATIONS:\n")
        f.write("  1. U-Net: Best for semantic segmentation, good balance of speed and accuracy\n")
        f.write("  2. PoreD²: Advanced features, good for detailed analysis with statistics\n")
        f.write("  3. YOLO: Object detection approach, good for counting individual pores\n\n")
        
        f.write("IMAGE TYPE CONSIDERATIONS:\n")
        f.write("  1. Consider the specific characteristics of your imaging modality\n")
        f.write("  2. Some models may perform better on certain image types\n")
        f.write("  3. Test with your specific dataset for best results\n")
    
    def _create_visualizations(self):
        #Create comprehensive visualizations
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Processing time comparison
        self._plot_processing_times()
        
        # 2. Pore coverage comparison
        self._plot_pore_coverage()
        
        # 3. Model performance heatmap
        self._plot_performance_heatmap()
        
        # 4. Sample results comparison
        self._create_sample_comparison()
        
        print("visualizations created")
    
    def _plot_processing_times(self):
        """Create processing time comparison plot"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data
        times_data = []
        for image_type in self.image_types:
            for model in self.models:
                if model in self.timing_data.get(image_type, {}) and self.timing_data[image_type][model]:
                    for time_val in self.timing_data[image_type][model]:
                        times_data.append({
                            'Model': model.upper(),
                            'Image_Type': image_type,
                            'Processing_Time': time_val
                        })
        
        if times_data:
            df_times = pd.DataFrame(times_data)
            
            # Box plot
            sns.boxplot(data=df_times, x='Model', y='Processing_Time', ax=axes[0])
            axes[0].set_title('Processing Time Distribution by Model')
            axes[0].set_ylabel('Processing Time (seconds)')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Bar plot with error bars
            avg_times = df_times.groupby(['Model', 'Image_Type'])['Processing_Time'].agg(['mean', 'std']).reset_index()
            
            sns.barplot(data=df_times, x='Image_Type', y='Processing_Time', hue='Model', ax=axes[1])
            axes[1].set_title('Average Processing Time by Image Type')
            axes[1].set_ylabel('Processing Time (seconds)')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].legend(title='Model')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'processing_times_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pore_coverage(self):
        """Create pore coverage comparison plot"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data
        coverage_data = []
        for image_type in self.image_types:
            for model in self.models:
                if model in self.test_results.get(image_type, {}):
                    for result in self.test_results[image_type][model]:
                        coverage_data.append({
                            'Model': model.upper(),
                            'Image_Type': image_type,
                            'Pore_Coverage': result['pore_statistics']['pore_percentage']
                        })
        
        if coverage_data:
            df_coverage = pd.DataFrame(coverage_data)
            
            # Box plot
            sns.boxplot(data=df_coverage, x='Model', y='Pore_Coverage', ax=axes[0])
            axes[0].set_title('Pore Coverage Distribution by Model')
            axes[0].set_ylabel('Pore Coverage (%)')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Bar plot
            sns.barplot(data=df_coverage, x='Image_Type', y='Pore_Coverage', hue='Model', ax=axes[1])
            axes[1].set_title('Average Pore Coverage by Image Type')
            axes[1].set_ylabel('Pore Coverage (%)')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].legend(title='Model')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pore_coverage_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_heatmap(self):
        """Create performance heatmap"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Processing time heatmap
        time_matrix = np.zeros((len(self.models), len(self.image_types)))
        coverage_matrix = np.zeros((len(self.models), len(self.image_types)))
        
        for i, model in enumerate(self.models):
            for j, image_type in enumerate(self.image_types):
                if model in self.performance_metrics.get(image_type, {}):
                    metrics = self.performance_metrics[image_type][model]
                    time_matrix[i, j] = metrics['avg_processing_time']
                    coverage_matrix[i, j] = metrics['avg_pore_coverage']
        
        # Time heatmap
        sns.heatmap(time_matrix, 
                   xticklabels=self.image_types,
                   yticklabels=[m.upper() for m in self.models],
                   annot=True, fmt='.3f', cmap='YlOrRd',
                   ax=axes[0])
        axes[0].set_title('Average Processing Time (seconds)')
        
        # Coverage heatmap
        sns.heatmap(coverage_matrix,
                   xticklabels=self.image_types,
                   yticklabels=[m.upper() for m in self.models],
                   annot=True, fmt='.1f', cmap='YlGnBu',
                   ax=axes[1])
        axes[1].set_title('Average Pore Coverage (%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_sample_comparison(self):
        """Create sample comparison of model outputs"""
        print("Creating sample comparison visualization...")
        
        # Find a sample image from each type
        for image_type in self.image_types:
            if image_type in self.test_results and self.test_results[image_type]:
                # Get the first successful result for each model
                sample_results = {}
                for model in self.models:
                    if (model in self.test_results[image_type] and 
                        self.test_results[image_type][model]):
                        sample_results[model] = self.test_results[image_type][model][0]
                
                if len(sample_results) >= 2:  # Need at least 2 models for comparison
                    self._create_side_by_side_comparison(image_type, sample_results)
    
    def _create_side_by_side_comparison(self, image_type: str, sample_results: Dict):
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
            axes[0, i].set_title(f'Original {image_type} Image')
            axes[0, i].axis('off')
            
            # Plot binary mask
            axes[1, i].imshow(binary_mask, cmap='gray')
            axes[1, i].set_title(f'{model.upper()} Result\n'
                               f'Coverage: {result["pore_statistics"]["pore_percentage"]:.1f}%\n'
                               f'Time: {result["timing"]["total_time"]:.3f}s')
            axes[1, i].axis('off')
        
        plt.suptitle(f'Model Comparison - {image_type} Images', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'sample_comparison_{image_type.lower()}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_analysis_results(self):
        """Save analysis results in JSON format"""
        results_path = os.path.join(self.output_dir, "analysis_results.json")
        
        # Prepare serializable data
        serializable_results = {
            'performance_metrics': self.performance_metrics,
            'timing_summary': {},
            'analysis_metadata': {
                'models_tested': self.models,
                'image_types_tested': self.image_types,
                'analysis_date': datetime.now().isoformat(),
                'test_directory': self.test_dir
            }
        }
        
        # Add timing summary
        for image_type in self.image_types:
            serializable_results['timing_summary'][image_type] = {}
            for model in self.models:
                if model in self.timing_data.get(image_type, {}):
                    timings = self.timing_data[image_type][model]
                    if timings:
                        serializable_results['timing_summary'][image_type][model] = {
                            'mean': float(np.mean(timings)),
                            'std': float(np.std(timings)),
                            'min': float(np.min(timings)),
                            'max': float(np.max(timings)),
                            'count': len(timings)
                        }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"analysis results saved: {results_path}")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
dat_dir = os.path.join(project_root,'CODE','DL MODELS (copy)')

def main():
    #main analysis function
    parser = argparse.ArgumentParser(description='Comprehensive Pore Detection Model Analysis')
    parser.add_argument('--test_dir', type=str, default=os.path.join(dat_dir,'Dataset'),
                       help='Directory containing test images')
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['unet', 'pored2', 'yolo'], 
                       default=['unet', 'pored2', 'yolo'],
                       help='Models to test and compare')
    parser.add_argument('--image_types', type=str, nargs='+',
                       choices=['AFM', 'CONFOCAL', 'CRYO-SEM', 'STED'],
                       default=['AFM', 'CONFOCAL', 'CRYO-SEM', 'STED'],
                       help='Image types to test')
    
    args = parser.parse_args()
    
    print("complete Pore Detection Analysis")
    print(f"Test Directory: {args.test_dir}")
    print(f"Models: {args.models}")
    print(f"Image Types: {args.image_types}")
    
    # Create analyzer
    analyzer = PoreDetectionAnalyzer(
        test_dir=args.test_dir,
        models=args.models,
        image_types=args.image_types
    )
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"Results Directory: {results['output_dir']}")
    print(f"Total Analysis Time: {results['total_time']:.2f} seconds")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
