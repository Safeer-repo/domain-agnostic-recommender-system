#!/usr/bin/env python3
"""
Script to test the optimized model selector.
Usage: python scripts/test_model_selector.py --domain DOMAIN --dataset DATASET
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import time

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.model_selector import ModelSelector
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

def main():
    """Main function to test the model selector."""
    parser = argparse.ArgumentParser(description="Test the optimized model selector")
    parser.add_argument("--domain", choices=["entertainment", "ecommerce", "education"],
                        required=True, help="Domain to test")
    parser.add_argument("--dataset", required=True, help="Dataset name to test")
    parser.add_argument("--metric", default="map_at_k", 
                        choices=["precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k", 
                                "rmse", "mae", "coverage", "novelty", "diversity"],
                        help="Metric to optimize for")
    parser.add_argument("--verbose", action="store_true", help="Show detailed characteristics")
    parser.add_argument("--eval", action="store_true", help="Evaluate the selected model")
    parser.add_argument("--train", action="store_true", help="Train the selected model")
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
        
        # Print characteristics if verbose
        if args.verbose:
            logging.info("Dataset characteristics:")
            for key, value in characteristics.items():
                logging.info(f"  {key}: {value}")
        
        # Calculate model scores
        model_scores = selector.calculate_model_scores(characteristics)
        logging.info("\nModel suitability scores:")
        for model, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"  {model}: {score}")
        
        # Select the best model
        best_model_name, best_params = selector.select_best_model(
            args.domain, args.dataset, characteristics, args.metric
        )
        
        logging.info(f"\nSelected model: {best_model_name}")
        logging.info(f"Optimization metric: {args.metric}")
        logging.info(f"Model hyperparameters: {json.dumps(best_params, indent=2)}")
        
        # Train the model if requested
        if args.train:
            logging.info("\nTraining the selected model...")
            start_time = time.time()
            model = selector.get_best_model(args.domain, args.dataset, args.metric)
            model.fit(train_data, features.get("user_features"), features.get("item_features"))
            training_time = time.time() - start_time
            logging.info(f"Training completed in {training_time:.2f} seconds")
            
            # Evaluate if requested
            if args.eval:
                logging.info("\nEvaluating the model...")
                metrics = model.evaluate(test_data, k=10)
                logging.info("Evaluation results:")
                for metric, value in metrics.items():
                    logging.info(f"  {metric}: {value:.4f}")
        
        # Load best tuned model if not training
        elif args.eval:
            logging.info("\nLoading best tuned model...")
            model = selector.load_best_tuned_model(args.domain, args.dataset, args.metric)
            
            logging.info("Evaluating the model...")
            metrics = model.evaluate(test_data, k=10)
            logging.info("Evaluation results:")
            for metric, value in metrics.items():
                logging.info(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        logging.error(f"Error testing model selector: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()