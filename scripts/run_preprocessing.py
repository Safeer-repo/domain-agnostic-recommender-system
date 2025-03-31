#!/usr/bin/env python3
"""
Script to run the preprocessing pipeline.
Usage: python scripts/run_preprocessing.py --domain DOMAIN --dataset DATASET [--force]
"""

import os
import sys
import argparse
import logging

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

def main():
    """Main function to run the preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline")
    parser.add_argument("--domain", choices=["entertainment", "ecommerce", "education"],
                        required=True, help="Domain to preprocess data for")
    parser.add_argument("--dataset", required=True, help="Dataset name to preprocess")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if processed data exists")
    parser.add_argument("--min-ratings", type=int, default=10, help="Minimum number of ratings per item to keep")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use for testing")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Get absolute path to the data directory
    data_dir = os.path.join(project_root, "data")
    
    # Create and run the preprocessing pipeline
    pipeline = PreprocessingPipeline(
        data_dir=data_dir,
        min_ratings=args.min_ratings,
        test_size=args.test_size
    )
    
    try:
        train_data, test_data = pipeline.preprocess(
            domain=args.domain,
            dataset_name=args.dataset,
            force_reprocess=args.force
        )
        
        logging.info(f"Preprocessing complete")
        logging.info(f"Train data shape: {train_data.shape}")
        logging.info(f"Test data shape: {test_data.shape}")
        
        if 'rating' in train_data.columns:
            logging.info(f"Train data rating stats:")
            logging.info(f"  Mean: {train_data['rating'].mean():.2f}")
            logging.info(f"  Min: {train_data['rating'].min()}")
            logging.info(f"  Max: {train_data['rating'].max()}")
        
    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
