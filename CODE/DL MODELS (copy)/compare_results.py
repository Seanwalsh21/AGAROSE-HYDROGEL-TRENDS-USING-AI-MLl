"""
Comprehensive Comparison Script for Pore Detection Model Results
Compares results between different test datasets across all image modalities

This script compares:
1. External results (new dataset) vs Original results (original dataset)
2. Performance across AFM, CONFOCAL, CRYO-SEM, and STED modalities
3. Generates graphical comparisons using box plots, bar charts, and statistical analysis
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ResultsComparator:
    """
    Comprehensive comparator for pore detection analysis results
    Compares performance between different datasets and generates visualizations
    """
    
    def __init__(self, external_results_dir: str, original_results_dir: str):
        """
        Initialize the comparator
        
        Args:
            external_results_dir: Directory containing external analysis results
            original_results_dir: Directory containing original analysis results
        """
        self.external_results_dir = external_results_dir
        self.original_results_dir = original_results_dir
        
        # Create output directory
        self.output_dir = f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load results
        self.external_data = self._load_results(external_results_dir, "External Dataset")
        self.original_data = self._load_results(original_results_dir, "Original Dataset")
        
        print(f"Results Comparator Initialized")
        print(f"External Results: {external_results_dir}")
        print(f"Original Results: {original_results_dir}")
        print(f"Output Directory: {self.output_dir}")
    
    def _load_results(self, results_dir: str, dataset_name: str) -> Dict:
        """Load analysis results from directory"""
        results_file = os.path.join(results_dir, "analysis_results.json")
        csv_file = os.path.join(results_dir, "performance_summary.csv")
        
        data = {
            'dataset_name': dataset_name,
            'results_dir': results_dir,
            'json_data': None,
            'csv_data': None,
            'available_modalities': [],
            'available_models': []
        }
        
        # Load JSON results
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                data['json_data'] = json.load(f)
            print(f"Loaded JSON results for {dataset_name}")
        else:
            print(f"JSON results not found for {dataset_name}: {results_file}")
        
        # Load CSV results
        if os.path.exists(csv_file):
            data['csv_data'] = pd.read_csv(csv_file)
            if not data['csv_data'].empty:
                data['available_modalities'] = data['csv_data']['Base_Type'].unique().tolist()
                data['available_models'] = data['csv_data']['Model'].unique().tolist()
            print(f"Loaded CSV results for {dataset_name}")
            print(f"  Available modalities: {data['available_modalities']}")
            print(f"  Available models: {data['available_models']}")
        else:
            print(f"CSV results not found for {dataset_name}: {csv_file}")
        
        return data
    
    def run_comprehensive_comparison(self):
        """Run comprehensive comparison across all modalities"""
        print("STARTING COMPREHENSIVE RESULTS COMPARISON")
        print(f"{'='*80}")
        
        # Check if data is available
        if (self.external_data['csv_data'] is None or self.original_data['csv_data'] is None or
            self.external_data['csv_data'].empty or self.original_data['csv_data'].empty):
            print("Error: Missing or empty data files. Cannot proceed with comparison.")
            return
        
        # Find common modalities and models
        common_modalities = set(self.external_data['available_modalities']).intersection(
            set(self.original_data['available_modalities'])
        )
        common_models = set(self.external_data['available_models']).intersection(
            set(self.original_data['available_models'])
        )
        
        print(f"Common modalities for comparison: {list(common_modalities)}")
        print(f"Common models for comparison: {list(common_models)}")
        
        if not common_modalities or not common_models:
            print("Error: No common modalities or models found for comparison.")
            return
        
        # Prepare combined dataset
        combined_data = self._prepare_combined_dataset()
        
        # Generate comparisons
        self._generate_comparison_report(combined_data, common_modalities, common_models)
        self._create_comparison_visualizations(combined_data, common_modalities, common_models)
        self._perform_statistical_analysis(combined_data, common_modalities, common_models)
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE COMPARISON COMPLETED")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*80}")
    
    def _prepare_combined_dataset(self) -> pd.DataFrame:
        """Prepare combined dataset for comparison"""
        print("Preparing combined dataset...")
        
        # Add dataset identifier to each dataframe
        external_df = self.external_data['csv_data'].copy()
        original_df = self.original_data['csv_data'].copy()
        
        # Rename subdirectories to more descriptive names
        external_df = self._rename_subdirectories(external_df, "External")
        original_df = self._rename_subdirectories(original_df, "Original")
        
        external_df['Dataset'] = self.external_data['dataset_name']
        original_df['Dataset'] = self.original_data['dataset_name']
        
        # Combine datasets
        combined_df = pd.concat([external_df, original_df], ignore_index=True)
        
        print(f"Combined dataset created with {len(combined_df)} records")
        return combined_df
    
    def _rename_subdirectories(self, df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """Rename subdirectories to be more descriptive"""
        df = df.copy()
        
        # Create descriptive names for subdirectories
        def create_descriptive_name(row):
            base_type = row['Base_Type']
            subdir = row['Subdirectory']
            
            if subdir == 'main':
                return f"{base_type} {suffix}"
            else:
                # Keep specific subdirectory names but add suffix for clarity
                return f"{base_type} {subdir} ({suffix})"
        
        df['Descriptive_Type'] = df.apply(create_descriptive_name, axis=1)
        return df
    
    def _generate_comparison_report(self, combined_data: pd.DataFrame, 
                                  common_modalities: set, common_models: set):
        """Generate comprehensive comparison report"""
        print("Generating comparison report...")
        
        report_path = os.path.join(self.output_dir, "comparison_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE PORE DETECTION RESULTS COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Comparison Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"External Results Directory: {self.external_results_dir}\n")
            f.write(f"Original Results Directory: {self.original_results_dir}\n")
            f.write(f"Common Modalities: {', '.join(common_modalities)}\n")
            f.write(f"Common Models: {', '.join(common_models)}\n\n")
            
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 20 + "\n")
            external_counts = combined_data[combined_data['Dataset'] == self.external_data['dataset_name']].groupby('Base_Type').size()
            original_counts = combined_data[combined_data['Dataset'] == self.original_data['dataset_name']].groupby('Base_Type').size()
            
            f.write("External Dataset:\n")
            for modality, count in external_counts.items():
                f.write(f"  {modality}: {count} test configurations\n")
            
            f.write("\nOriginal Dataset:\n")
            for modality, count in original_counts.items():
                f.write(f"  {modality}: {count} test configurations\n")
            f.write("\n")
            
            # Performance comparison by modality
            f.write("PERFORMANCE COMPARISON BY MODALITY\n")
            f.write("-" * 45 + "\n\n")
            
            for modality in common_modalities:
                f.write(f"{modality} MODALITY COMPARISON:\n")
                f.write("  " + "-" * 30 + "\n")
                
                modality_data = combined_data[combined_data['Base_Type'] == modality]
                
                # Average performance comparison
                avg_stats = modality_data.groupby(['Dataset', 'Model']).agg({
                    'Avg_Processing_Time': 'mean',
                    'Avg_Pore_Coverage': 'mean',
                    'Success_Rate': 'mean'
                }).round(4)
                
                f.write("  Average Processing Time (seconds):\n")
                for dataset in modality_data['Dataset'].unique():
                    f.write(f"    {dataset}:\n")
                    dataset_stats = avg_stats.loc[dataset]
                    for model in dataset_stats.index:
                        f.write(f"      {model}: {dataset_stats.loc[model, 'Avg_Processing_Time']:.4f}\n")
                
                f.write("\n  Average Pore Coverage (%):\n")
                for dataset in modality_data['Dataset'].unique():
                    f.write(f"    {dataset}:\n")
                    dataset_stats = avg_stats.loc[dataset]
                    for model in dataset_stats.index:
                        f.write(f"      {model}: {dataset_stats.loc[model, 'Avg_Pore_Coverage']:.2f}\n")
                
                f.write("\n")
            
            # Model ranking comparison
            f.write("MODEL RANKING COMPARISON\n")
            f.write("-" * 30 + "\n")
            
            for dataset_name in [self.external_data['dataset_name'], self.original_data['dataset_name']]:
                f.write(f"\n{dataset_name} Rankings:\n")
                dataset_data = combined_data[combined_data['Dataset'] == dataset_name]
                
                # Speed ranking
                speed_ranking = dataset_data.groupby('Model')['Avg_Processing_Time'].mean().sort_values()
                f.write("  Speed Ranking (fastest to slowest):\n")
                for i, (model, avg_time) in enumerate(speed_ranking.items(), 1):
                    f.write(f"    {i}. {model}: {avg_time:.4f} seconds\n")
                
                # Coverage ranking
                coverage_ranking = dataset_data.groupby('Model')['Avg_Pore_Coverage'].mean().sort_values(ascending=False)
                f.write("  Coverage Ranking (highest to lowest):\n")
                for i, (model, avg_coverage) in enumerate(coverage_ranking.items(), 1):
                    f.write(f"    {i}. {model}: {avg_coverage:.2f}%\n")
            
            # Recommendations
            f.write("\nCOMPARISON INSIGHTS AND RECOMMENDATIONS\n")
            f.write("-" * 45 + "\n")
            self._write_comparison_insights(f, combined_data, common_modalities, common_models)
        
        print(f"Comparison report saved: {report_path}")
    
    def _write_comparison_insights(self, f, combined_data: pd.DataFrame, 
                                 common_modalities: set, common_models: set):
        """Write comparison insights and recommendations"""
        f.write("Key Insights from the Comparison:\n\n")
        
        # Dataset differences analysis
        f.write("DATASET DIFFERENCES:\n")
        
        for modality in common_modalities:
            modality_data = combined_data[combined_data['Base_Type'] == modality]
            
            # Compare average performance between datasets
            dataset_comparison = modality_data.groupby('Dataset').agg({
                'Avg_Processing_Time': 'mean',
                'Avg_Pore_Coverage': 'mean',
                'Success_Rate': 'mean'
            })
            
            if len(dataset_comparison) == 2:
                datasets = dataset_comparison.index.tolist()
                time_diff = abs(dataset_comparison.loc[datasets[0], 'Avg_Processing_Time'] - 
                               dataset_comparison.loc[datasets[1], 'Avg_Processing_Time'])
                coverage_diff = abs(dataset_comparison.loc[datasets[0], 'Avg_Pore_Coverage'] - 
                                   dataset_comparison.loc[datasets[1], 'Avg_Pore_Coverage'])
                
                f.write(f"  {modality}: Processing time differs by {time_diff:.4f}s, ")
                f.write(f"coverage differs by {coverage_diff:.1f}%\n")
        
        f.write("MODEL CONSISTENCY:\n")
        f.write("  - Compare how each model performs across different datasets\n")
        f.write("  - Identify models that show consistent performance\n")
        f.write("  - Note any significant performance variations\n\n")
        
        f.write("• RECOMMENDATIONS:\n")
        f.write("  - Use statistical tests to determine significance of differences\n")
        f.write("  - Consider dataset characteristics when choosing models\n")
        f.write("  - Validate model performance on both original and external datasets\n")
        f.write("  - Account for potential dataset-specific biases in model evaluation\n")
    
    def _create_comparison_visualizations(self, combined_data: pd.DataFrame,
                                        common_modalities: set, common_models: set):
        """Create comprehensive comparison visualizations"""
        print("Creating comparison visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("Set2")
        
        # 1. Processing time comparison by modality
        self._plot_processing_time_comparison(combined_data, common_modalities)
        
        # 2. Pore coverage comparison by modality
        self._plot_pore_coverage_comparison(combined_data, common_modalities)
        
        # 3. Model performance heatmaps
        # self._plot_performance_heatmaps(combined_data, common_modalities, common_models)
        
        # 4. Side-by-side comparisons for each modality
        for modality in common_modalities:
            self._plot_modality_specific_comparison(combined_data, modality, common_models)
        
        # 5. Overall performance comparison
        self._plot_overall_performance_comparison(combined_data, common_models)
        
        print("All visualizations created")
    
    def _plot_processing_time_comparison(self, combined_data: pd.DataFrame, common_modalities: set):
        """Create processing time comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()
        
        modalities = list(common_modalities)
        
        for i, modality in enumerate(modalities[:4]):  # Limit to 4 modalities
            if i >= len(axes):
                break
            
            modality_data = combined_data[combined_data['Base_Type'] == modality]
            
            if not modality_data.empty:
                sns.boxplot(data=modality_data, x='Model', y='Avg_Processing_Time', 
                           hue='Dataset', ax=axes[i])
                axes[i].set_title(f'Processing Time Comparison - {modality}')
                axes[i].set_ylabel('Processing Time (seconds)')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].legend(title='Dataset')
            else:
                axes[i].text(0.5, 0.5, f'No data for {modality}', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Processing Time Comparison - {modality} (No Data)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'processing_time_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pore_coverage_comparison(self, combined_data: pd.DataFrame, common_modalities: set):
        """Create pore coverage comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()
        
        modalities = list(common_modalities)
        
        for i, modality in enumerate(modalities[:4]):
            if i >= len(axes):
                break
            
            modality_data = combined_data[combined_data['Base_Type'] == modality]
            
            if not modality_data.empty:
                sns.barplot(data=modality_data, x='Model', y='Avg_Pore_Coverage', 
                           hue='Dataset', ax=axes[i])
                axes[i].set_title(f'Pore Coverage Comparison - {modality}')
                axes[i].set_ylabel('Average Pore Coverage (%)')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].legend(title='Dataset')
            else:
                axes[i].text(0.5, 0.5, f'No data for {modality}', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Pore Coverage Comparison - {modality} (No Data)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pore_coverage_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_heatmaps(self, combined_data: pd.DataFrame, 
                                 common_modalities: set, common_models: set):
        """Create performance heatmaps for comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Create separate heatmaps for each dataset
        datasets = combined_data['Dataset'].unique()
        
        for i, dataset in enumerate(datasets):
            if i >= 2:
                break
            
            dataset_data = combined_data[combined_data['Dataset'] == dataset]
            
            # Processing time heatmap
            time_pivot = dataset_data.pivot_table(
                values='Avg_Processing_Time', 
                index='Model', 
                columns='Base_Type', 
                aggfunc='mean'
            )
            
            sns.heatmap(time_pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                       ax=axes[0, i])
            axes[0, i].set_title(f'Processing Time - {dataset}')
            
            # Coverage heatmap
            coverage_pivot = dataset_data.pivot_table(
                values='Avg_Pore_Coverage', 
                index='Model', 
                columns='Base_Type', 
                aggfunc='mean'
            )
            
            sns.heatmap(coverage_pivot, annot=True, fmt='.1f', cmap='YlGnBu',
                       ax=axes[1, i])
            axes[1, i].set_title(f'Pore Coverage - {dataset}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_heatmaps_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_modality_specific_comparison(self, combined_data: pd.DataFrame, 
                                         modality: str, common_models: set):
        """Create modality-specific detailed comparison"""
        modality_data = combined_data[combined_data['Base_Type'] == modality]
        
        if modality_data.empty:
            print(f"No data available for {modality} modality")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Processing time box plot
        sns.boxplot(data=modality_data, x='Model', y='Avg_Processing_Time', 
                   hue='Dataset', ax=axes[0, 0])
        axes[0, 0].set_title(f'{modality} - Processing Time Distribution')
        axes[0, 0].set_ylabel('Processing Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Pore coverage box plot
        sns.boxplot(data=modality_data, x='Model', y='Avg_Pore_Coverage', 
                   hue='Dataset', ax=axes[0, 1])
        axes[0, 1].set_title(f'{modality} - Pore Coverage Distribution')
        axes[0, 1].set_ylabel('Pore Coverage (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Success rate comparison
        sns.barplot(data=modality_data, x='Model', y='Success_Rate', 
                   hue='Dataset', ax=axes[1, 0])
        axes[1, 0].set_title(f'{modality} - Success Rate Comparison')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Scatter plot: Processing time vs Coverage
        for dataset in modality_data['Dataset'].unique():
            dataset_subset = modality_data[modality_data['Dataset'] == dataset]
            axes[1, 1].scatter(dataset_subset['Avg_Processing_Time'], 
                              dataset_subset['Avg_Pore_Coverage'],
                              label=dataset, alpha=0.7, s=100)
        
        axes[1, 1].set_xlabel('Processing Time (seconds)')
        axes[1, 1].set_ylabel('Pore Coverage (%)')
        axes[1, 1].set_title(f'{modality} - Processing Time vs Coverage')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Detailed Analysis - {modality} Modality', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{modality.lower()}_detailed_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_overall_performance_comparison(self, combined_data: pd.DataFrame, common_models: set):
        """Create overall performance comparison across all modalities"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Overall processing time by model
        sns.boxplot(data=combined_data, x='Model', y='Avg_Processing_Time', 
                   hue='Dataset', ax=axes[0, 0])
        axes[0, 0].set_title('Overall Processing Time Comparison')
        axes[0, 0].set_ylabel('Processing Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Overall pore coverage by model
        sns.boxplot(data=combined_data, x='Model', y='Avg_Pore_Coverage', 
                   hue='Dataset', ax=axes[0, 1])
        axes[0, 1].set_title('Overall Pore Coverage Comparison')
        axes[0, 1].set_ylabel('Pore Coverage (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Performance by modality
        sns.boxplot(data=combined_data, x='Base_Type', y='Avg_Pore_Coverage', 
                   hue='Dataset', ax=axes[1, 0])
        axes[1, 0].set_title('Performance by Modality')
        axes[1, 0].set_ylabel('Pore Coverage (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Success rate comparison
        sns.barplot(data=combined_data, x='Model', y='Success_Rate', 
                   hue='Dataset', ax=axes[1, 1])
        axes[1, 1].set_title('Overall Success Rate Comparison')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Overall Performance Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'overall_performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _perform_statistical_analysis(self, combined_data: pd.DataFrame,
                                    common_modalities: set, common_models: set):
        """Perform statistical analysis of differences"""
        print("Performing statistical analysis...")
        
        stats_path = os.path.join(self.output_dir, "statistical_analysis.txt")
        
        with open(stats_path, 'w') as f:
            f.write("STATISTICAL ANALYSIS OF PERFORMANCE DIFFERENCES\n")
            
            f.write("T-TEST RESULTS FOR PERFORMANCE DIFFERENCES\n")
            
            datasets = combined_data['Dataset'].unique()
            if len(datasets) == 2:
                dataset1_data = combined_data[combined_data['Dataset'] == datasets[0]]
                dataset2_data = combined_data[combined_data['Dataset'] == datasets[1]]
                
                # T-tests for each modality and metric
                for modality in common_modalities:
                    f.write(f"{modality} MODALITY:\n")
                    f.write("  " + "-" * 20 + "\n")
                    
                    mod1_data = dataset1_data[dataset1_data['Base_Type'] == modality]
                    mod2_data = dataset2_data[dataset2_data['Base_Type'] == modality]
                    
                    if not mod1_data.empty and not mod2_data.empty:
                        # Processing time t-test
                        t_stat_time, p_val_time = stats.ttest_ind(
                            mod1_data['Avg_Processing_Time'], 
                            mod2_data['Avg_Processing_Time']
                        )
                        
                        # Coverage t-test
                        t_stat_cov, p_val_cov = stats.ttest_ind(
                            mod1_data['Avg_Pore_Coverage'], 
                            mod2_data['Avg_Pore_Coverage']
                        )
                        
                        f.write(f"  Processing Time: t={t_stat_time:.4f}, p={p_val_time:.4f}")
                        f.write(f" ({'Significant' if p_val_time < 0.05 else 'Not significant'})\n")
                        
                        f.write(f"  Pore Coverage: t={t_stat_cov:.4f}, p={p_val_cov:.4f}")
                        f.write(f" ({'Significant' if p_val_cov < 0.05 else 'Not significant'})\n")
                    else:
                        f.write("  Insufficient data for statistical testing\n")
                    
                    f.write("\n")
                
                # Overall statistical summary
                f.write("OVERALL STATISTICAL SUMMARY\n")
                f.write("-" * 30 + "\n")
                
                # Overall t-tests
                overall_time_t, overall_time_p = stats.ttest_ind(
                    dataset1_data['Avg_Processing_Time'], 
                    dataset2_data['Avg_Processing_Time']
                )
                
                overall_cov_t, overall_cov_p = stats.ttest_ind(
                    dataset1_data['Avg_Pore_Coverage'], 
                    dataset2_data['Avg_Pore_Coverage']
                )
                
                f.write(f"Overall Processing Time Difference: p={overall_time_p:.4f}\n")
                f.write(f"Overall Coverage Difference: p={overall_cov_p:.4f}\n")
                
                # Descriptive statistics
                f.write("\nDESCRIPTIVE STATISTICS\n")
                f.write("-" * 25 + "\n")
                
                desc_stats = combined_data.groupby('Dataset').agg({
                    'Avg_Processing_Time': ['mean', 'std', 'min', 'max'],
                    'Avg_Pore_Coverage': ['mean', 'std', 'min', 'max'],
                    'Success_Rate': ['mean', 'std']
                }).round(4)
                
                f.write(str(desc_stats))
                f.write("\n")
        
        print(f"Statistical analysis saved: {stats_path}")
        
        # Save detailed comparison CSV
        comparison_csv_path = os.path.join(self.output_dir, "detailed_comparison.csv")
        combined_data.to_csv(comparison_csv_path, index=False)
        print(f"Detailed comparison data saved: {comparison_csv_path}")


def main():
    """Main comparison function"""
    parser = argparse.ArgumentParser(description='Compare Pore Detection Analysis Results')
    parser.add_argument('--external_results', type=str, default='analysis_results_20250909_110405',
                       help='Directory containing external analysis results')
    parser.add_argument('--original_results', type=str, default='analysis_results_20250825_142445',
                       help='Directory containing original analysis results')
    
    args = parser.parse_args()
    
    print("Starting Comprehensive Results Comparison")
    print(f"External Results Directory: {args.external_results}")
    print(f"Original Results Directory: {args.original_results}")
    
    # Validate directories exist
    if not os.path.exists(args.external_results):
        print(f"Error: External results directory '{args.external_results}' does not exist!")
        return 1
    
    if not os.path.exists(args.original_results):
        print(f"Error: Original results directory '{args.original_results}' does not exist!")
        return 1
    
    # Create comparator
    comparator = ResultsComparator(
        external_results_dir=args.external_results,
        original_results_dir=args.original_results
    )
    
    # Run comprehensive comparison
    try:
        comparator.run_comprehensive_comparison()
        
        print("COMPARISON COMPLETED SUCCESSFULLY!")
        print(f"Results Directory: {comparator.output_dir}")
        print("\nGenerated Files:")
        print("  comparison_report.txt - Detailed comparison report")
        print("  statistical_analysis.txt - Statistical test results")
        print("  Multiple comparison visualization PNG files:")
        print("    - processing_time_comparison.png")
        print("    - pore_coverage_comparison.png")
        print("    - performance_heatmaps_comparison.png")
        print("    - [modality]_detailed_comparison.png (for each modality)")
        print("    - overall_performance_comparison.png")
        print("  detailed_comparison.csv - Combined data for further analysis")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)