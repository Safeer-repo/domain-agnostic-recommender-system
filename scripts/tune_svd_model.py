#!/usr/bin/env python3
"""
Script to perform hyperparameter tuning for the SVD model.
Usage: python scripts/tune_svd_model.py --domain DOMAIN --dataset DATASET
"""

import os
import sys
import argparse
import logging
import time
import json
import pandas as pd
import itertools
from typing import Dict, List, Any

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.models.model_registry import ModelRegistry

def main():
    """Main function to perform hyperparameter tuning for the SVD model."""
    parser = argparse.ArgumentParser(description="Tune hyperparameters for the SVD model")
    parser.add_argument("--domain", choices=["entertainment", "ecommerce", "education"],
                        required=True, help="Domain to tune model for")
    parser.add_argument("--dataset", required=True, help="Dataset name to tune model on")
    parser.add_argument("--metric", default="ndcg_at_k", 
                        choices=["precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k", 
                                "coverage", "novelty", "diversity", "rmse", "mae"],
                        help="Metric to optimize")
    parser.add_argument("--k", type=int, default=10, help="K value for evaluation metrics")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Get absolute paths
    data_dir = os.path.join(project_root, "data")
    artifacts_dir = os.path.join(project_root, "artifacts")
    
    # Create results directory for tuning outputs
    results_dir = os.path.join(artifacts_dir, "tuning_results", args.domain, args.dataset, "svd")
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Create preprocessing pipeline
        pipeline = PreprocessingPipeline(data_dir=data_dir)
        
        # Load preprocessed data
        logging.info(f"Loading preprocessed data for {args.domain}/{args.dataset}")
        train_data, test_data = pipeline.preprocess(
            domain=args.domain,
            dataset_name=args.dataset
        )
        logging.info(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
        
        # Create model registry
        registry = ModelRegistry(artifacts_dir=artifacts_dir)
        
        # Define parameter grid
        param_grid = {
            "n_factors": [50, 100, 150],           # Number of latent factors
            "n_epochs": [15, 20, 25],              # Number of training epochs
            "lr_all": [0.002, 0.005, 0.01],        # Learning rate
            "reg_all": [0.01, 0.02, 0.05]          # Regularization parameter
        }
        
        # Generate all combinations of parameters
        param_combinations = list(itertools.product(
            param_grid["n_factors"], 
            param_grid["n_epochs"], 
            param_grid["lr_all"],
            param_grid["reg_all"]
        ))
        
        # Log the total number of combinations
        total_combinations = len(param_combinations)
        logging.info(f"Starting grid search with {total_combinations} parameter combinations")
        
        # Initialize results tracking
        results = []
        best_score = float('-inf') if args.metric != 'rmse' and args.metric != 'mae' else float('inf')
        best_params = None
        best_model_path = None
        
        # Iterate through parameter combinations
        for i, (n_factors, n_epochs, lr_all, reg_all) in enumerate(param_combinations):
            combo_start_time = time.time()
            logging.info(f"Combination {i+1}/{total_combinations}: "
                         f"n_factors={n_factors}, n_epochs={n_epochs}, "
                         f"lr_all={lr_all}, reg_all={reg_all}")
            
            try:
                # Get SVD model
                model = registry.get_model("svd")
                
                # Set hyperparameters
                model.set_hyperparameters(
                    n_factors=n_factors,
                    n_epochs=n_epochs,
                    lr_all=lr_all,
                    reg_all=reg_all
                )
                
                # Train the model
                model.fit(train_data)
                
                # Evaluate the model - PASS train_data for beyond-accuracy metrics
                eval_metrics = model.evaluate(test_data, k=args.k, train_data=train_data)
                
                # Record current score
                current_score = eval_metrics.get(args.metric, 0)
                
                # Log results
                logging.info(f"  Evaluation results:")
                for metric, value in eval_metrics.items():
                    logging.info(f"    {metric}: {value:.4f}")
                
                # Save result
                result = {
                    "n_factors": n_factors,
                    "n_epochs": n_epochs,
                    "lr_all": lr_all,
                    "reg_all": reg_all,
                    **eval_metrics,
                    "training_time": model.metadata.get("training_time", 0),
                    "total_time": time.time() - combo_start_time
                }
                results.append(result)
                
                # Check if this is the best model so far
                is_better = False
                if args.metric in ['rmse', 'mae']:  # Lower is better
                    if current_score < best_score:
                        is_better = True
                        best_score = current_score
                else:  # Higher is better
                    if current_score > best_score:
                        is_better = True
                        best_score = current_score
                
                if is_better:
                    best_params = {
                        "n_factors": n_factors,
                        "n_epochs": n_epochs,
                        "lr_all": lr_all,
                        "reg_all": reg_all
                    }
                    # Save the best model
                    model_filename = f"svd_tuned_f{n_factors}_e{n_epochs}_lr{lr_all}_reg{reg_all}.pkl"
                    best_model_path = os.path.join(results_dir, model_filename)
                    model.save(best_model_path)
                    logging.info(f"  New best model! {args.metric}: {current_score:.4f}")
                
            except Exception as e:
                logging.error(f"Error with combination {i+1}: {str(e)}")
                result = {
                    "n_factors": n_factors,
                    "n_epochs": n_epochs,
                    "lr_all": lr_all,
                    "reg_all": reg_all,
                    "error": str(e)
                }
                results.append(result)
        
        # Save all results to a JSON file
        results_file = os.path.join(results_dir, "tuning_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save results to a CSV for easier analysis
        df_results = pd.DataFrame(results)
        csv_file = os.path.join(results_dir, "tuning_results.csv")
        df_results.to_csv(csv_file, index=False)
        
        # Print best parameters
        if best_params:
            logging.info(f"Best parameters found:")
            for param, value in best_params.items():
                logging.info(f"  {param}: {value}")
            logging.info(f"Best {args.metric}: {best_score:.4f}")
            logging.info(f"Best model saved to: {best_model_path}")
            
            # Save best parameters to a separate file
            best_params_file = os.path.join(results_dir, "best_params.json")
            with open(best_params_file, 'w') as f:
                json.dump({
                    "best_params": best_params,
                    "best_score": best_score,
                    "metric": args.metric,
                    "model_path": best_model_path
                }, f, indent=2)
        else:
            logging.warning("No best parameters found.")
        
        logging.info(f"Tuning results saved to {results_dir}")
        
    except Exception as e:
        logging.error(f"Error during hyperparameter tuning: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()