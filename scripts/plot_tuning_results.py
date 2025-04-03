#!/usr/bin/env python3
"""
Script to visualize hyperparameter tuning results.
Usage: python scripts/plot_tuning_results.py --domain DOMAIN --dataset DATASET
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    """Main function to visualize hyperparameter tuning results."""
    parser = argparse.ArgumentParser(description="Visualize hyperparameter tuning results")
    parser.add_argument("--domain", choices=["entertainment", "ecommerce", "education"],
                        required=True, help="Domain of tuning results")
    parser.add_argument("--dataset", required=True, help="Dataset name of tuning results")
    parser.add_argument("--metrics", nargs='+', 
                        default=["ndcg_at_k", "precision_at_k", "recall_at_k", "map_at_k"],
                        help="Metrics to visualize")
    args = parser.parse_args()
    
    # Get results directory
    results_dir = os.path.join(project_root, "artifacts", "tuning_results", args.domain, args.dataset)
    csv_file = os.path.join(results_dir, "tuning_results.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: Results file not found at {csv_file}")
        sys.exit(1)
    
    # Load results
    results = pd.read_csv(csv_file)
    
    # Create output directory for plots
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load best parameters
    best_params_file = os.path.join(results_dir, "best_params.json")
    best_params = None
    if os.path.exists(best_params_file):
        import json
        with open(best_params_file, 'r') as f:
            best_params_data = json.load(f)
            best_params = best_params_data.get("best_params")
            best_metric = best_params_data.get("metric")
            best_score = best_params_data.get("best_score")
    
    # Create heatmap for each metric showing factors vs regularization
    for metric in args.metrics:
        if metric not in results.columns:
            print(f"Warning: Metric {metric} not found in results")
            continue
        
        # Create pivot tables for different alpha values
        for alpha in results['alpha'].unique():
            alpha_results = results[results['alpha'] == alpha]
            
            # Create pivot table
            pivot = alpha_results.pivot(index='factors', columns='regularization', values=metric)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.4f')
            plt.title(f"{metric} - Alpha: {alpha}")
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(plots_dir, f"{metric}_alpha_{alpha}.png")
            plt.savefig(plot_file)
            plt.close()
    
    # Create parameter importance plot
    plt.figure(figsize=(12, 10))
    for i, metric in enumerate(args.metrics):
        if metric not in results.columns:
            continue
        
        plt.subplot(2, 2, i+1)
        
        # Factors
        sns.boxplot(x='factors', y=metric, data=results)
        plt.title(f"Impact of Factors on {metric}")
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "factors_impact.png"))
    plt.close()
    
    plt.figure(figsize=(12, 10))
    for i, metric in enumerate(args.metrics):
        if metric not in results.columns:
            continue
        
        plt.subplot(2, 2, i+1)
        
        # Regularization
        sns.boxplot(x='regularization', y=metric, data=results)
        plt.title(f"Impact of Regularization on {metric}")
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "regularization_impact.png"))
    plt.close()
    
    plt.figure(figsize=(12, 10))
    for i, metric in enumerate(args.metrics):
        if metric not in results.columns:
            continue
        
        plt.subplot(2, 2, i+1)
        
        # Alpha
        sns.boxplot(x='alpha', y=metric, data=results)
        plt.title(f"Impact of Alpha on {metric}")
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "alpha_impact.png"))
    plt.close()
    
    # Create summary statistics table
    summary = results.groupby(['factors', 'regularization', 'alpha'])[args.metrics].mean().reset_index()
    summary = summary.sort_values(by=args.metrics[0], ascending=False)
    
    # Save top 10 combinations to CSV
    top_combinations = summary.head(10)
    top_combinations.to_csv(os.path.join(results_dir, "top_combinations.csv"), index=False)
    
    print(f"Visualization complete. Plots saved to {plots_dir}")
    
    # Print best parameters if available
    if best_params:
        print(f"\nBest parameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"Best {best_metric}: {best_score:.4f}")

if __name__ == "__main__":
    main()
