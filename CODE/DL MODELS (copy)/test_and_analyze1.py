"""
Comprehensive Analysis and Testing Script for Pore Detection Models - Updated Version

This script provides comprehensive testing and analysis capabilities for all trained models
with support for analyzing subdirectories separately (concentrations/magnifications).

It includes:

1. Model performance comparison by subdirectory
2. Batch inference on test images
3. Statistical analysis of results by concentration/magnification
4. Visual comparison of outputs
5. Timing benchmarks
6. Detailed reporting with subdirectory breakdown

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
    Comprehensive analyzer for pore detection models with subdirectory support
    
    This class provides tools for comparing model performance,
    analyzing results by concentration/magnification, and generating detailed reports.
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
        
        # Define all image types with their subdirectories
        self.image_type_subdirs = {
            'AFM': ['1%', '1.5%', '2%'],
            'CONFOCAL': ['0.375%', '1%'],
            'CRYO-SEM': ['x1000', 'x3000', 'x10000', 'x30000', 'x60000'],
            'STED': ['0.375%', '1%']
        }
        
        # Create flat list of all image types (including subdirectories)
        self.all_image_types = []
        for base_type, subdirs in self.image_type_subdirs.items():
            for subdir in subdirs:
                self.all_image_types.append(f"{base_type}_{subdir}")
        
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
        print(f"Image Types with Subdirectories:")
        for base_type, subdirs in self.image_type_subdirs.items():
            print(f"  {base_type}: {subdirs}")
        print(f"Total Analysis Types: {len(self.all_image_types)}")
        print(f"Output Directory: {self.output_dir}")
    
    def find_test_images(self) -> Dict[str, List[str]]:
        """
        Find test images for each image type and subdirectory
        
        Returns:
            Dictionary mapping image types to lists of image paths
        """
        test_images = {image_type: [] for image_type in self.all_image_types}
        
        # Base folder mapping
        base_folder_mapping = {
            'AFM': 'AFM folder/AFM',
            'CONFOCAL': 'Confocal folder/CONFOCAL',
            'CRYO-SEM': 'Cryo-sem Folder/CRYO-SEM',
            'STED': 'STED Folder/STED'
        }
        
        for base_type, subdirs in self.image_type_subdirs.items():
            base_folder = base_folder_mapping[base_type]
            base_path = os.path.join(self.test_dir, base_folder)
            
            for subdir in subdirs:
                image_type_key = f"{base_type}_{subdir}"
                subdir_path = os.path.join(base_path, subdir)
                
                if os.path.exists(subdir_path):
                    # Find all .tif files in this specific subdirectory
                    for root, dirs, files in os.walk(subdir_path):
                        for file in files:
                            if file.lower().endswith(('.tif', '.tiff')):
                                test_images[image_type_key].append(os.path.join(root, file))
                    
                    print(f"Found {len(test_images[image_type_key])} test images for {image_type_key}")
                else:
                    print(f"⚠ Directory not found: {subdir_path}")
        
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
        
        # Initialize results structure
        for image_type in self.all_image_types:
            self.test_results[image_type] = {}
            self.performance_metrics[image_type] = {}
            self.timing_data[image_type] = {}
            
            for model in self.models:
                self.test_results[image_type][model] = []
                self.timing_data[image_type][model] = []
        
        # Run analysis for each combination
        total_images = sum(len(images) for images in test_images.values())
        total_combinations = total_images * len(self.models)
        current_combination = 0
        
        for image_type in self.all_image_types:
            if not test_images[image_type]:
                print(f"No test images found for {image_type}, skipping...")
                continue
            
            # Extract base type for inference engine
            base_type = image_type.split('_')[0]
            subdir = '_'.join(image_type.split('_')[1:])
            

            print(f"ANALYZING {image_type} IMAGES")
            print(f"Base Type: {base_type}, Subdirectory: {subdir}")
            
            # Limit to first 5 images per type for demonstration
            sample_images = test_images[image_type][:5]
            
            for model in self.models:
                print(f"testing {model.upper()} on {image_type} images...")
                
                # Initialize inference engine with base type
                try:
                    engine = PoreInferenceEngine(model, base_type)
                    
                    model_results = []
                    model_timings = []
                    
                    for i, image_path in enumerate(sample_images):
                        current_combination += 1
                        print(f"[{current_combination}/{min(len(sample_images)*len(self.models)*len([t for t in self.all_image_types if test_images[t]]), total_combinations)}] Processing {os.path.basename(image_path)} with {model.upper()}")
                        
                        # Run inference
                        result = engine.process_image(
                            image_path, 
                            output_path=os.path.join(self.output_dir, f"{model}_{image_type}_{i}")
                        )
                        
                        if result['success']:
                            # Add subdirectory information to result
                            result['image_type_full'] = image_type
                            result['base_type'] = base_type
                            result['subdirectory'] = subdir
                            
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
        

        print("complete analysis completed")
        print(f"Total Analysis Time: {total_time:.2f} seconds")
        print(f"Results saved to: {self.output_dir}")
        
        return {
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'timing_data': self.timing_data,
            'total_time': total_time,
            'output_dir': self.output_dir
        }
    
    def _calculate_performance_metrics(self, image_type: str, model: str, results: List[Dict]):
        """Calculate performance metrics for a model on an image type"""
        if not results:
            self.performance_metrics[image_type][model] = {
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
        
        self.performance_metrics[image_type][model] = metrics
    
    def _generate_comparison_report(self, total_time: float):
        """Generate comprehensive comparison report with subdirectory breakdown"""
        report_path = os.path.join(self.output_dir, "comprehensive_analysis_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE PORE DETECTION MODEL ANALYSIS REPORT\n")
            f.write("WITH SUBDIRECTORY BREAKDOWN\n")
            
            f.write("EXECUTIVE SUMMARY:")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Analysis Time: {total_time:.2f} seconds\n")
            f.write(f"Models Tested: {', '.join(self.models)}\n")
            f.write(f"Total Image Types: {len(self.all_image_types)}\n")
            f.write("Image Types with Subdirectories:\n")
            for base_type, subdirs in self.image_type_subdirs.items():
                f.write(f"  {base_type}: {', '.join(subdirs)}\n")
            f.write("\n")
            
            f.write("PERFORMANCE COMPARISON BY MODEL\n")
            f.write("-" * 40 + "\n\n")
            
            for model in self.models:
                f.write(f"{model.upper()} MODEL PERFORMANCE:\n")
                f.write("  " + "-" * 30 + "\n")
                
                model_avg_times = []
                model_avg_coverages = []
                
                # Group by base type
                for base_type in self.image_type_subdirs.keys():
                    f.write(f"  {base_type}:\n")
                    
                    for subdir in self.image_type_subdirs[base_type]:
                        image_type = f"{base_type}_{subdir}"
                        
                        if image_type in self.performance_metrics and model in self.performance_metrics[image_type]:
                            metrics = self.performance_metrics[image_type][model]
                            
                            f.write(f"    {subdir}:\n")
                            f.write(f"      Processing Time: {metrics['avg_processing_time']:.4f} ± {metrics.get('std_processing_time', 0):.4f} seconds\n")
                            f.write(f"      Pore Coverage: {metrics['avg_pore_coverage']:.2f} ± {metrics.get('std_pore_coverage', 0):.2f}%\n")
                            f.write(f"      Success Rate: {metrics['successful_inferences']}/{metrics['total_inferences']}\n")
                            
                            model_avg_times.append(metrics['avg_processing_time'])
                            model_avg_coverages.append(metrics['avg_pore_coverage'])
                        else:
                            f.write(f"    {subdir}: No data\n")
                    
                    f.write("\n")
                
                if model_avg_times:
                    f.write(f"  Overall Average Processing Time: {np.mean(model_avg_times):.4f} seconds\n")
                    f.write(f"  Overall Average Pore Coverage: {np.mean(model_avg_coverages):.2f}%\n\n")
            
            f.write("PERFORMANCE COMPARISON BY IMAGE TYPE AND SUBDIRECTORY\n")
            f.write("-" * 55 + "\n\n")
            
            for base_type in self.image_type_subdirs.keys():
                f.write(f"{base_type} IMAGING RESULTS:\n")
                f.write("  " + "-" * 30 + "\n")
                
                for subdir in self.image_type_subdirs[base_type]:
                    image_type = f"{base_type}_{subdir}"
                    f.write(f"  {subdir}:\n")
                    
                    for model in self.models:
                        if image_type in self.performance_metrics and model in self.performance_metrics[image_type]:
                            metrics = self.performance_metrics[image_type][model]
                            f.write(f"    {model.upper()}: ")
                            f.write(f"Time: {metrics['avg_processing_time']:.4f}s, ")
                            f.write(f"Coverage: {metrics['avg_pore_coverage']:.2f}%, ")
                            f.write(f"Success: {metrics['successful_inferences']}/{metrics['total_inferences']}\n")
                        else:
                            f.write(f"    {model.upper()}: No data\n")
                    
                    f.write("\n")
                
                f.write("\n")
            
            # Model rankings
            f.write("MODEL COMPARISON SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            model_rankings = self._calculate_model_rankings()
            
            f.write("Speed Ranking (fastest to slowest):\n")
            for i, (model, avg_time) in enumerate(model_rankings['speed'], 1):
                f.write(f"  {i}. {model.upper()}: {avg_time:.4f} seconds\n")
            
            f.write("\nPore Detection Ranking (highest to lowest coverage):\n")
            for i, (model, avg_coverage) in enumerate(model_rankings['coverage'], 1):
                f.write(f"  {i}. {model.upper()}: {avg_coverage:.2f}%\n")
            
            f.write("\nCONCENTRATION/MAGNIFICATION ANALYSIS\n")
            f.write("-" * 40 + "\n")
            self._write_subdirectory_analysis(f)
            
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            self._write_recommendations(f, model_rankings)
        
        print(f"comprehensive report saved: {report_path}")
    
    def _write_subdirectory_analysis(self, f):
        """Write analysis specific to concentrations/magnifications"""
        f.write("Analysis by concentration/magnification levels:\n\n")
        
        for base_type in self.image_type_subdirs.keys():
            f.write(f"{base_type} Analysis:\n")
            
            # Collect data for this base type
            subdir_data = {}
            for subdir in self.image_type_subdirs[base_type]:
                image_type = f"{base_type}_{subdir}"
                subdir_data[subdir] = {}
                
                for model in self.models:
                    if image_type in self.performance_metrics and model in self.performance_metrics[image_type]:
                        metrics = self.performance_metrics[image_type][model]
                        subdir_data[subdir][model] = {
                            'coverage': metrics['avg_pore_coverage'],
                            'time': metrics['avg_processing_time']
                        }
            
            # Find trends
            if subdir_data:
                f.write("trends observed:\n")
                
                # Analyze coverage trends
                for model in self.models:
                    coverages = []
                    subdirs_with_data = []
                    
                    for subdir in self.image_type_subdirs[base_type]:
                        if subdir in subdir_data and model in subdir_data[subdir]:
                            coverages.append(subdir_data[subdir][model]['coverage'])
                            subdirs_with_data.append(subdir)
                    
                    if len(coverages) > 1:
                        if base_type in ['AFM', 'CONFOCAL', 'STED']:
                            # Concentration-based analysis
                            f.write(f"    {model.upper()}: Coverage varies across concentrations ({min(coverages):.1f}% - {max(coverages):.1f}%)\n")
                        else:
                            # Magnification-based analysis
                            f.write(f"    {model.upper()}: Coverage varies across magnifications ({min(coverages):.1f}% - {max(coverages):.1f}%)\n")
            
            f.write("\n")
    
    def _calculate_model_rankings(self) -> Dict:
        """Calculate model rankings based on performance metrics"""
        model_avg_times = {model: [] for model in self.models}
        model_avg_coverages = {model: [] for model in self.models}
        
        for image_type in self.all_image_types:
            for model in self.models:
                if image_type in self.performance_metrics and model in self.performance_metrics[image_type]:
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
        """Write recommendations based on analysis"""
        f.write("Based on the comprehensive analysis:\n\n")
        
        if model_rankings['speed']:
            fastest_model = model_rankings['speed'][0][0]
            f.write(f"For SPEED: {fastest_model.upper()} is the fastest model\n")
            f.write(f"  - Best for real-time applications\n")
            f.write(f"  - Suitable for high-throughput analysis\n\n")
        
        if model_rankings['coverage']:
            best_detection_model = model_rankings['coverage'][0][0]
            f.write(f"For ACCURACY: {best_detection_model.upper()} provides highest pore coverage detection\n")
            f.write(f"  - Best for detailed pore analysis\n")
            f.write(f"  - Suitable for research applications\n\n")
        
        f.write("SUBDIRECTORY-SPECIFIC RECOMMENDATIONS:\n")
        f.write("  - Different concentrations/magnifications may favor different models\n")
        f.write("  - Consider the specific concentration/magnification range of your application\n")
        f.write("  - Higher concentrations may require different optimization strategies\n")
        f.write("  - Magnification levels in CRYO-SEM show varying performance patterns\n\n")
        
        f.write("MODEL-SPECIFIC RECOMMENDATIONS:\n")
        f.write("  - U-Net: Best for semantic segmentation, good balance of speed and accuracy\n")
        f.write("  - PoreD²: Advanced features, good for detailed analysis with statistics\n")
        f.write("  - YOLO: Object detection approach, good for counting individual pores\n\n")
        
        f.write("IMAGE TYPE CONSIDERATIONS:\n")
        f.write("  - AFM: Test across concentration range (1%, 1.5%, 2%)\n")
        f.write("  - CONFOCAL/STED: Compare performance at 0.375% vs 1%\n")
        f.write("  - CRYO-SEM: Consider magnification effects (x1000 to x60000)\n")
        f.write("  - Validate with your specific concentration/magnification requirements\n")
    
    def _create_visualizations(self):
        """Create comprehensive visualizations with subdirectory breakdown"""
        print("Creating visualizations with subdirectory breakdown...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Processing time comparison by subdirectory
        self._plot_processing_times_by_subdirectory()
        
        # 2. Pore coverage comparison by subdirectory
        self._plot_pore_coverage_by_subdirectory()
        
        # 3. Model performance heatmap
        self._plot_performance_heatmap()
        
        # 4. Concentration/magnification trend analysis
        self._plot_trend_analysis()
        
        # 5. Sample results comparison
        self._create_sample_comparison()
        
        print("visualizations created")
    
    def _plot_processing_times_by_subdirectory(self):
        """Create processing time comparison plot with subdirectory breakdown"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()
        
        # Prepare data
        times_data = []
        for image_type in self.all_image_types:
            base_type = image_type.split('_')[0]
            subdir = '_'.join(image_type.split('_')[1:])
            
            for model in self.models:
                if image_type in self.timing_data and model in self.timing_data[image_type] and self.timing_data[image_type][model]:
                    for time_val in self.timing_data[image_type][model]:
                        times_data.append({
                            'Model': model.upper(),
                            'Base_Type': base_type,
                            'Subdirectory': subdir,
                            'Full_Type': image_type,
                            'Processing_Time': time_val
                        })
        
        if times_data:
            df_times = pd.DataFrame(times_data)
            
            # Plot for each base type
            base_types = ['AFM', 'CONFOCAL', 'CRYO-SEM', 'STED']
            
            for i, base_type in enumerate(base_types):
                if i >= len(axes):
                    break
                
                base_data = df_times[df_times['Base_Type'] == base_type]
                
                if not base_data.empty:
                    sns.boxplot(data=base_data, x='Subdirectory', y='Processing_Time', hue='Model', ax=axes[i])
                    axes[i].set_title(f'Processing Time Distribution - {base_type}')
                    axes[i].set_ylabel('Processing Time (seconds)')
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    axes[i].set_title(f'Processing Time Distribution - {base_type} (No Data)')
                    axes[i].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[i].transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'processing_times_by_subdirectory.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pore_coverage_by_subdirectory(self):
        """Create pore coverage comparison plot with subdirectory breakdown"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()
        
        # Prepare data
        coverage_data = []
        for image_type in self.all_image_types:
            base_type = image_type.split('_')[0]
            subdir = '_'.join(image_type.split('_')[1:])
            
            for model in self.models:
                if image_type in self.test_results and model in self.test_results[image_type]:
                    for result in self.test_results[image_type][model]:
                        coverage_data.append({
                            'Model': model.upper(),
                            'Base_Type': base_type,
                            'Subdirectory': subdir,
                            'Full_Type': image_type,
                            'Pore_Coverage': result['pore_statistics']['pore_percentage']
                        })
        
        if coverage_data:
            df_coverage = pd.DataFrame(coverage_data)
            
            # Plot for each base type
            base_types = ['AFM', 'CONFOCAL', 'CRYO-SEM', 'STED']
            
            for i, base_type in enumerate(base_types):
                if i >= len(axes):
                    break
                
                base_data = df_coverage[df_coverage['Base_Type'] == base_type]
                
                if not base_data.empty:
                    sns.barplot(data=base_data, x='Subdirectory', y='Pore_Coverage', hue='Model', ax=axes[i])
                    axes[i].set_title(f'Average Pore Coverage - {base_type}')
                    axes[i].set_ylabel('Pore Coverage (%)')
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    axes[i].set_title(f'Average Pore Coverage - {base_type} (No Data)')
                    axes[i].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[i].transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pore_coverage_by_subdirectory.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_heatmap(self):
        """Create performance heatmap with all subdirectories"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Create matrices
        n_models = len(self.models)
        n_types = len(self.all_image_types)
        
        time_matrix = np.zeros((n_models, n_types))
        coverage_matrix = np.zeros((n_models, n_types))
        
        model_labels = [m.upper() for m in self.models]
        type_labels = []
        
        for j, image_type in enumerate(self.all_image_types):
            type_labels.append(image_type.replace('_', '\n'))
            
            for i, model in enumerate(self.models):
                if image_type in self.performance_metrics and model in self.performance_metrics[image_type]:
                    metrics = self.performance_metrics[image_type][model]
                    time_matrix[i, j] = metrics['avg_processing_time']
                    coverage_matrix[i, j] = metrics['avg_pore_coverage']
        
        # Time heatmap
        sns.heatmap(time_matrix, 
                   xticklabels=type_labels,
                   yticklabels=model_labels,
                   annot=True, fmt='.3f', cmap='YlOrRd',
                   ax=axes[0])
        axes[0].set_title('Average Processing Time (seconds)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Coverage heatmap
        sns.heatmap(coverage_matrix,
                   xticklabels=type_labels,
                   yticklabels=model_labels,
                   annot=True, fmt='.1f', cmap='YlGnBu',
                   ax=axes[1])
        axes[1].set_title('Average Pore Coverage (%)')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_heatmap_all_subdirectories.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_trend_analysis(self):
        """Create trend analysis plots for concentration/magnification effects"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()
        
        base_types = ['AFM', 'CONFOCAL', 'CRYO-SEM', 'STED']
        
        for i, base_type in enumerate(base_types):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Collect data for this base type
            trend_data = {}
            subdirs = self.image_type_subdirs[base_type]
            
            for model in self.models:
                coverages = []
                times = []
                valid_subdirs = []
                
                for subdir in subdirs:
                    image_type = f"{base_type}_{subdir}"
                    if (image_type in self.performance_metrics and 
                        model in self.performance_metrics[image_type]):
                        metrics = self.performance_metrics[image_type][model]
                        coverages.append(metrics['avg_pore_coverage'])
                        times.append(metrics['avg_processing_time'])
                        valid_subdirs.append(subdir)
                
                if coverages:
                    trend_data[model] = {
                        'subdirs': valid_subdirs,
                        'coverages': coverages,
                        'times': times
                    }
            
            # Plot coverage trends
            if trend_data:
                for model, data in trend_data.items():
                    ax.plot(range(len(data['subdirs'])), data['coverages'], 
                           marker='o', label=f'{model.upper()}', linewidth=2, markersize=8)
                
                ax.set_xlabel('Concentration/Magnification Level')
                ax.set_ylabel('Average Pore Coverage (%)')
                ax.set_title(f'Pore Coverage Trends - {base_type}')
                ax.set_xticks(range(len(subdirs)))
                ax.set_xticklabels(subdirs, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No data available for {base_type}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Pore Coverage Trends - {base_type} (No Data)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'concentration_magnification_trends.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_sample_comparison(self):
        """Create sample comparison of model outputs for each subdirectory"""
        print("Creating sample comparison visualization for subdirectories...")
        
        # Create comparisons for each base type
        for base_type in self.image_type_subdirs.keys():
            subdirs = self.image_type_subdirs[base_type]
            
            # Find a representative subdirectory with data
            for subdir in subdirs:
                image_type = f"{base_type}_{subdir}"
                
                if image_type in self.test_results and self.test_results[image_type]:
                    # Get the first successful result for each model
                    sample_results = {}
                    for model in self.models:
                        if (model in self.test_results[image_type] and 
                            self.test_results[image_type][model]):
                            sample_results[model] = self.test_results[image_type][model][0]
                    
                    if len(sample_results) >= 2:  # Need at least 2 models for comparison
                        self._create_side_by_side_comparison(image_type, sample_results)
                        break  # Only create one sample per base type
    
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
        """Save analysis results in JSON format with subdirectory breakdown"""
        results_path = os.path.join(self.output_dir, "analysis_results.json")
        
        # Prepare serializable data
        serializable_results = {
            'performance_metrics': self.performance_metrics,
            'timing_summary': {},
            'subdirectory_breakdown': {},
            'analysis_metadata': {
                'models_tested': self.models,
                'image_types_tested': self.all_image_types,
                'image_type_subdirs': self.image_type_subdirs,
                'analysis_date': datetime.now().isoformat(),
                'test_directory': self.test_dir
            }
        }
        
        # Add timing summary
        for image_type in self.all_image_types:
            serializable_results['timing_summary'][image_type] = {}
            for model in self.models:
                if image_type in self.timing_data and model in self.timing_data[image_type]:
                    timings = self.timing_data[image_type][model]
                    if timings:
                        serializable_results['timing_summary'][image_type][model] = {
                            'mean': float(np.mean(timings)),
                            'std': float(np.std(timings)),
                            'min': float(np.min(timings)),
                            'max': float(np.max(timings)),
                            'count': len(timings)
                        }
        
        # Add subdirectory breakdown
        for base_type in self.image_type_subdirs.keys():
            serializable_results['subdirectory_breakdown'][base_type] = {}
            
            for subdir in self.image_type_subdirs[base_type]:
                image_type = f"{base_type}_{subdir}"
                serializable_results['subdirectory_breakdown'][base_type][subdir] = {}
                
                for model in self.models:
                    if (image_type in self.performance_metrics and 
                        model in self.performance_metrics[image_type]):
                        metrics = self.performance_metrics[image_type][model]
                        serializable_results['subdirectory_breakdown'][base_type][subdir][model] = {
                            'avg_processing_time': float(metrics['avg_processing_time']),
                            'avg_pore_coverage': float(metrics['avg_pore_coverage']),
                            'successful_inferences': int(metrics['successful_inferences']),
                            'total_inferences': int(metrics['total_inferences'])
                        }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"analysis results saved: {results_path}")
        
        # Also save a CSV summary for easy analysis
        self._save_csv_summary()
    
    def _save_csv_summary(self):
        """Save a CSV summary of all results for easy analysis"""
        csv_path = os.path.join(self.output_dir, "performance_summary.csv")
        
        # Prepare data for CSV
        csv_data = []
        
        for image_type in self.all_image_types:
            base_type = image_type.split('_')[0]
            subdir = '_'.join(image_type.split('_')[1:])
            
            for model in self.models:
                if (image_type in self.performance_metrics and 
                    model in self.performance_metrics[image_type]):
                    metrics = self.performance_metrics[image_type][model]
                    
                    csv_data.append({
                        'Base_Type': base_type,
                        'Subdirectory': subdir,
                        'Full_Type': image_type,
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
            print(f"csv summary saved: {csv_path}")
        else:
            print("no data available for CSV summary")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
dat_dir = os.path.join(project_root,'CODE','DL MODELS (copy)')

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Comprehensive Pore Detection Model Analysis with Subdirectory Support')
    parser.add_argument('--test_dir', type=str, default=os.path.join(dat_dir,'Dataset'),
                       help='Directory containing test images')
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['unet', 'pored2', 'yolo'], 
                       default=['unet', 'pored2', 'yolo'],
                       help='Models to test and compare')
    
    args = parser.parse_args()
    
    print("starting Comprehensive Pore Detection Analysis with Subdirectory Breakdown")
    print(f"Test Directory: {args.test_dir}")
    print(f"Models: {args.models}")
    print("subdirectory Structure:")
    print("  AFM: 1%, 1.5%, 2%")
    print("  CONFOCAL: 0.375%, 1%")
    print("  CRYO-SEM: x1000, x3000, x10000, x30000, x60000")
    print("  STED: 0.375%, 1%")
    
    # Create analyzer
    analyzer = PoreDetectionAnalyzer(
        test_dir=args.test_dir,
        models=args.models
    )
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"Results Directory: {results['output_dir']}")
    print(f"Total Analysis Time: {results['total_time']:.2f} seconds")
    print("generated Files:")
    print("comprehensive_analysis_report.txt - Detailed text report")
    print("performance_summary.csv - CSV data for further analysis")
    print("multiple visualization PNG files")
    print("analysis_results.json - Complete results in JSON format")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)