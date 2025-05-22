#!/usr/bin/env python3
"""
Script to train and evaluate the LightFM model.
Usage: python scripts/train_lightfm_model.py --domain DOMAIN --dataset DATASET [--components COMPONENTS] [--loss LOSS]
"""

import os
import sys
import argparse
import logging
import time
import traceback

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.models.model_registry import ModelRegistry

def main():
    """Main function to train and evaluate the LightFM model."""
    parser = argparse.ArgumentParser(description="Train and evaluate the LightFM model")
    parser.add_argument("--domain", choices=["entertainment", "ecommerce", "education"],
                        required=True, help="Domain to train model for")
    parser.add_argument("--dataset", required=True, help="Dataset name to train model on")
    parser.add_argument("--components", type=int, default=64, help="Number of latent components")
    parser.add_argument("--loss", choices=["warp", "bpr", "logistic"], default="warp", 
                        help="Loss function to use")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--item-alpha", type=float, default=0.0001, help="Item regularization")
    parser.add_argument("--user-alpha", type=float, default=0.0001, help="User regularization")
    parser.add_argument("--max-sampled", type=int, default=10, help="Maximum sampled items for WARP loss")
    parser.add_argument("--k", type=int, default=10, help="K value for evaluation metrics")
    parser.add_argument("--use-features", action="store_true", help="Use user and item features")
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
        
        # Get LightFM model
        logging.info("Creating LightFM model")
        model = registry.get_model("lightfm")
        
        # Set hyperparameters
        model.set_hyperparameters(
            no_components=args.components,
            loss=args.loss,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            item_alpha=args.item_alpha,
            user_alpha=args.user_alpha,
            max_sampled=args.max_sampled
        )
        
        # Train the model
        logging.info(f"Training LightFM model with {args.components} factors, {args.epochs} epochs")
        start_time = time.time()
        model.fit(train_data, user_features, item_features)
        train_time = time.time() - start_time
        logging.info(f"Training completed in {train_time:.2f} seconds")
        
        # Evaluate the model
        logging.info(f"Evaluating model with k={args.k}")
        try:
            metrics = model.evaluate(test_data, k=args.k, train_data=train_data)
        except TypeError:
            # Fallback if evaluate doesn't accept train_data
            metrics = model.evaluate(test_data, k=args.k)
        
        # Print evaluation results
        logging.info("Evaluation results:")
        for metric, value in metrics.items():
            logging.info(f"  {metric}: {value:.4f}")
        
        # Save the model
        model_path = registry.save_model(model, args.domain, args.dataset)
        logging.info(f"Model saved to {model_path}")
        
        # Generate sample recommendations
        logging.info("Generating sample recommendations")
        # Use a small sample of users from the training data
        sample_users = train_data['user_id'].sample(min(5, len(train_data['user_id'].unique()))).unique().tolist()
        
        for user_id in sample_users:
            try:
                recommendations = model.predict(user_id, n=5)
                
                if recommendations:
                    logging.info(f"Top 5 recommendations for user {user_id}:")
                    for item_id, score in recommendations:
                        logging.info(f"  Item {item_id}: {score:.4f}")
                else:
                    logging.info(f"No recommendations generated for user {user_id}")
            except Exception as e:
                logging.error(f"Error generating recommendations for user {user_id}: {str(e)}")
                if args.debug:
                    traceback.print_exc()
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
