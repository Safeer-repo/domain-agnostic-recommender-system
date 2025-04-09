#!/usr/bin/env python3
"""
Script to train and evaluate the SVD model.
Usage: python scripts/train_svd_model.py --domain DOMAIN --dataset DATASET [--factors FACTORS] [--epochs EPOCHS]
"""

import os
import sys
import argparse
import logging
import time
import json
import traceback

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.models.model_registry import ModelRegistry

def main():
    """Main function to train and evaluate the SVD model."""
    parser = argparse.ArgumentParser(description="Train and evaluate the SVD model")
    parser.add_argument("--domain", choices=["entertainment", "ecommerce", "education"],
                        required=True, help="Domain to train model for")
    parser.add_argument("--dataset", required=True, help="Dataset name to train model on")
    parser.add_argument("--factors", type=int, default=100, help="Number of latent factors")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--reg", type=float, default=0.02, help="Regularization parameter")
    parser.add_argument("--k", type=int, default=10, help="K value for evaluation metrics")
    parser.add_argument("--metrics", nargs="+", default=["rmse", "mae", "precision", "recall"],
                      help="Metrics to evaluate (rmse, mae, precision, recall)")
    parser.add_argument("--use-features", action="store_true", help="Use user and item features")
    parser.add_argument("--output-recommendations", action="store_true", 
                      help="Output sample recommendations to a file")
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
        # Create preprocessing pipeline
        pipeline = PreprocessingPipeline(data_dir=data_dir)
        
        # Load preprocessed data
        logging.info(f"Loading preprocessed data for {args.domain}/{args.dataset}")
        train_data, test_data = pipeline.preprocess(
            domain=args.domain,
            dataset_name=args.dataset
        )
        logging.info(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
        
        # Load features if requested
        user_features = None
        item_features = None
        if args.use_features:
            logging.info("Loading user and item features")
            features = pipeline.create_features(
                domain=args.domain,
                dataset_name=args.dataset
            )
            user_features = features.get('user_features')
            item_features = features.get('item_features')
            logging.info(f"Loaded {len(user_features) if user_features is not None else 0} user features and "
                         f"{len(item_features) if item_features is not None else 0} item features")
        
        # Create model registry
        registry = ModelRegistry(artifacts_dir=artifacts_dir)
        
        # Get SVD model
        logging.info("Creating SVD model")
        model = registry.get_model("svd")
        
        # Set hyperparameters
        model.set_hyperparameters(
            n_factors=args.factors,
            n_epochs=args.epochs,
            lr_all=args.lr,
            reg_all=args.reg
        )
        
        # Train the model
        logging.info(f"Training SVD model with {args.factors} factors, {args.epochs} epochs")
        start_time = time.time()
        model.fit(train_data, user_features, item_features)
        train_time = time.time() - start_time
        logging.info(f"Training completed in {train_time:.2f} seconds")
        
        # Evaluate the model
        logging.info(f"Evaluating model with k={args.k} and metrics {args.metrics}")
        metrics = model.evaluate(test_data, k=args.k, metrics=args.metrics)
        
        # Print evaluation results
        logging.info("Evaluation results:")
        for metric, value in metrics.items():
            logging.info(f"  {metric}: {value:.4f}")
        
        # Save the model
        model_path = registry.save_model(model, args.domain, args.dataset)
        logging.info(f"Model saved to {model_path}")
        
        # Save metadata to a separate JSON file
        metadata_path = os.path.join(
            os.path.dirname(model_path),
            f"{os.path.splitext(os.path.basename(model_path))[0]}_metadata.json"
        )
        
        with open(metadata_path, 'w') as f:
            json.dump({
                "domain": args.domain,
                "dataset": args.dataset,
                "model_type": "svd",
                "hyperparameters": model.hyperparams,
                "performance": metrics,
                "training_time": train_time,
                "n_training_samples": len(train_data),
                "n_test_samples": len(test_data)
            }, f, indent=4)
        
        logging.info(f"Model metadata saved to {metadata_path}")
        
        # Generate sample recommendations
        logging.info("Generating sample recommendations")
        # Use a small sample of users from the training data
        sample_users = train_data['user_id'].sample(min(5, len(train_data['user_id'].unique()))).unique().tolist()
        
        recommendations = {}
        for user_id in sample_users:
            try:
                user_recs = model.predict(user_id, n=5)
                
                if user_recs:
                    logging.info(f"Top 5 recommendations for user {user_id}:")
                    for item_id, score in user_recs:
                        logging.info(f"  Item {item_id}: {score:.4f}")
                    recommendations[str(user_id)] = [{"item_id": int(item_id), "score": float(score)} 
                                                  for item_id, score in user_recs]
                else:
                    logging.info(f"No recommendations generated for user {user_id}")
            except Exception as e:
                logging.error(f"Error generating recommendations for user {user_id}: {str(e)}")
                if args.debug:
                    traceback.print_exc()
        
        # Output recommendations to file if requested
        if args.output_recommendations and recommendations:
            recs_path = os.path.join(
                artifacts_dir, 
                "recommendations", 
                args.domain, 
                args.dataset,
                "svd_sample_recommendations.json"
            )
            
            os.makedirs(os.path.dirname(recs_path), exist_ok=True)
            
            with open(recs_path, 'w') as f:
                json.dump(recommendations, f, indent=4)
            
            logging.info(f"Sample recommendations saved to {recs_path}")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()