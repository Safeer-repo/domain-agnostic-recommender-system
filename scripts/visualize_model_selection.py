#!/usr/bin/env python3
"""
Script to visualize model selection process and dataset characteristics.
Usage: python scripts/visualize_model_selection.py --domain DOMAIN --dataset DATASET
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.model_selector import ModelSelector
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

def create_radar_chart(model_scores: Dict[str, float], output_path: str):
    """Create a radar chart showing model suitability scores."""
    categories = list(model_scores.keys())
    values = list(model_scores.values())
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Number of models
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Plot data
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    
    # Set title
    plt.title('Model Suitability Scores', size=16, y=1.1)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_characteristics_heatmap(characteristics: Dict[str, Any], output_path: str):
    """Create a heatmap showing dataset characteristics impact on model selection."""
    # Define characteristic groups
    groups = {
        'Scale': ['num_users', 'num_items', 'num_interactions', 'sparsity'],
        'Distribution': ['item_popularity_gini', 'user_activity_gini', 'item_popularity_entropy', 'user_activity_entropy'],
        'Temporal': ['has_temporal_data', 'temporal_span_days', 'interactions_per_day', 'avg_time_between_interactions'],
        'Features': ['num_user_features', 'num_item_features', 'user_feature_sparsity', 'item_feature_sparsity'],
        'Rating Type': ['binary_ratings', 'explicit_ratings', 'implicit_feedback'],
        'Cold Start': ['cold_start_ratio_users', 'cold_start_ratio_items'],
        'Complexity': ['avg_items_per_user', 'avg_users_per_item', 'recommendation_difficulty']
    }
    
    # Create data for heatmap
    data = []
    labels = []
    for group, features in groups.items():
        for feature in features:
            if feature in characteristics:
                value = characteristics[feature]
                # Convert boolean to numeric
                if isinstance(value, bool):
                    value = 1.0 if value else 0.0
                # Normalize numeric values
                elif isinstance(value, (int, float)) and not np.isnan(value):
                    value = float(value)
                else:
                    value = np.nan
                data.append(value)
                labels.append(f"{group}: {feature}")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    data_array = np.array(data).reshape(-1, 1)
    
    # Normalize data to 0-1 range
    if not np.all(np.isnan(data_array)):
        data_min = np.nanmin(data_array)
        data_max = np.nanmax(data_array)
        if data_max > data_min:
            data_array = (data_array - data_min) / (data_max - data_min)
    
    sns.heatmap(data_array, annot=True, cmap='YlOrRd', yticklabels=labels, xticklabels=['Value'], 
                cbar_kws={'label': 'Normalized Value'})
    
    plt.title('Dataset Characteristics', size=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_model_comparison_bar(tuning_results: Dict[str, Dict], metric: str, output_path: str):
    """Create a bar chart comparing model performances."""
    models = []
    scores = []
    
    for model_name, result in tuning_results.items():
        if result.get("metric") == metric:
            models.append(model_name)
            scores.append(result.get("best_score", 0))
    
    if not models:
        return  # No data to plot
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, scores)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(scores),
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.title(f'Model Performance Comparison - {metric}', size=16)
    plt.ylabel(metric, size=12)
    plt.xlabel('Model', size=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_decision_tree_diagram(characteristics: Dict[str, Any], selected_model: str, output_path: str):
    """Create a visual representation of the decision process."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Decision nodes and edges
    decision_flow = []
    
    # Dataset size decision
    large_dataset = characteristics["num_interactions"] > 100000
    if large_dataset:
        decision_flow.append(("Large Dataset\n(>100K interactions)", "Yes"))
    else:
        decision_flow.append(("Large Dataset\n(>100K interactions)", "No"))
    
    # Sparsity decision
    very_sparse = characteristics["sparsity"] > 0.995
    if very_sparse:
        decision_flow.append(("Extreme Sparsity\n(>99.5%)", "Yes"))
    else:
        decision_flow.append(("Extreme Sparsity\n(>99.5%)", "No"))
    
    # Feature richness decision
    feature_rich = characteristics["num_user_features"] > 0 or characteristics["num_item_features"] > 0
    if feature_rich:
        decision_flow.append(("Rich Features", "Yes"))
    else:
        decision_flow.append(("Rich Features", "No"))
    
    # Cold start decision
    cold_start = characteristics["cold_start_ratio_users"] > 0.1 or characteristics["cold_start_ratio_items"] > 0.1
    if cold_start:
        decision_flow.append(("Cold Start Issues", "Yes"))
    else:
        decision_flow.append(("Cold Start Issues", "No"))
    
    # Final decision
    decision_flow.append(("Selected Model", selected_model))
    
    # Create a simple flowchart visualization
    y_pos = 1.0
    for i, (node, answer) in enumerate(decision_flow):
        if i < len(decision_flow) - 1:
            # Decision node
            rect = plt.Rectangle((0.1, y_pos - 0.05), 0.3, 0.1, fill=True, facecolor='lightblue', edgecolor='black')
            ax.add_patch(rect)
            ax.text(0.25, y_pos, node, ha='center', va='center', fontsize=10)
            
            # Answer
            ax.text(0.45, y_pos, answer, ha='left', va='center', fontsize=10, fontweight='bold')
            
            # Arrow to next node
            if i < len(decision_flow) - 1:
                ax.arrow(0.25, y_pos - 0.05, 0, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')
            
            y_pos -= 0.15
        else:
            # Final node
            rect = plt.Rectangle((0.1, y_pos - 0.05), 0.3, 0.1, fill=True, facecolor='lightgreen', edgecolor='black')
            ax.add_patch(rect)
            ax.text(0.25, y_pos, f"{node}:\n{answer}", ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.1)
    ax.axis('off')
    plt.title('Model Selection Decision Process', size=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to visualize model selection process."""
    parser = argparse.ArgumentParser(description="Visualize model selection process")
    parser.add_argument("--domain", choices=["entertainment", "ecommerce", "education"],
                        required=True, help="Domain to analyze")
    parser.add_argument("--dataset", required=True, help="Dataset name to analyze")
    parser.add_argument("--metric", default="map_at_k", 
                        choices=["precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k", 
                                "rmse", "mae", "coverage", "novelty", "diversity"],
                        help="Metric to optimize for")
    parser.add_argument("--output-dir", default="visualizations", help="Output directory for visualizations")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Get absolute paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    artifacts_dir = os.path.join(project_root, "artifacts")
    output_dir = os.path.join(project_root, args.output_dir, args.domain, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create model selector
        selector = ModelSelector(artifacts_dir=artifacts_dir, data_dir=data_dir)
        
        # Load data for analysis
        logging.info(f"Loading data for {args.domain}/{args.dataset}")
        pipeline = PreprocessingPipeline(data_dir=data_dir)
        train_data, test_data = pipeline.preprocess(args.domain, args.dataset)
        features = pipeline.create_features(args.domain, args.dataset)
        
        # Analyze dataset characteristics
        logging.info("Analyzing dataset characteristics...")
        characteristics = selector.analyze_dataset(
            train_data, 
            test_data, 
            features.get("user_features"), 
            features.get("item_features")
        )
        
        # Calculate model scores
        model_scores = selector.calculate_model_scores(characteristics)
        
        # Select the best model
        best_model_name, _ = selector.select_best_model(
            args.domain, args.dataset, characteristics, args.metric
        )
        
        # Create visualizations
        logging.info("Creating visualizations...")
        
        # 1. Radar chart of model scores
        create_radar_chart(model_scores, os.path.join(output_dir, "model_scores_radar.png"))
        
        # 2. Heatmap of dataset characteristics
        create_characteristics_heatmap(characteristics, os.path.join(output_dir, "dataset_characteristics_heatmap.png"))
        
        # 3. Decision tree diagram
        create_decision_tree_diagram(characteristics, best_model_name, os.path.join(output_dir, "decision_tree.png"))
        
        # 4. Model comparison bar chart if tuning results exist
        tuning_results = {}
        for model_name in ["als", "svd", "lightfm", "sar", "baseline"]:
            if model_name == "als":
                model_tuning_path = os.path.join(
                    artifacts_dir, "tuning_results", args.domain, args.dataset, "best_params.json"
                )
            else:
                model_tuning_path = os.path.join(
                    artifacts_dir, "tuning_results", args.domain, args.dataset, model_name, "best_params.json"
                )
            
            if os.path.exists(model_tuning_path):
                try:
                    with open(model_tuning_path, 'r') as f:
                        tuning_data = json.load(f)
                        tuning_results[model_name] = tuning_data
                except Exception as e:
                    logging.error(f"Error loading tuning results for {model_name}: {str(e)}")
        
        if tuning_results:
            create_model_comparison_bar(tuning_results, args.metric, os.path.join(output_dir, f"model_comparison_{args.metric}.png"))
        
        # Convert numpy types to regular python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return str(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        # Save dataset characteristics to JSON for reference
        serializable_characteristics = convert_to_serializable(characteristics)
        with open(os.path.join(output_dir, "dataset_characteristics.json"), 'w') as f:
            json.dump(serializable_characteristics, f, indent=2)
        
        # Save model scores to JSON
        serializable_scores = convert_to_serializable(model_scores)
        with open(os.path.join(output_dir, "model_scores.json"), 'w') as f:
            json.dump(serializable_scores, f, indent=2)
        
        logging.info(f"Visualizations saved to {output_dir}")
        
    except Exception as e:
        logging.error(f"Error creating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()