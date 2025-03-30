#!/usr/bin/env python3
"""
Script to run the preprocessing pipeline.
Usage: python scripts/run_preprocessing.py --domain DOMAIN --dataset DATASET [--force]
"""

import os
import argparse
import logging
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

def main():
    """Main function to run the preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline")
    parser.add_argument("--domain", choices=["entertainment", "ecommerce", "education"],
                        required=True, help="Domain to preprocess data for")
    parser.add_argument("--dataset", required=True, help="Dataset name to preprocess")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if processed data exists")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Setup logging
    logger = setup_logging(config, f"preprocess_{args.domain}_{args.dataset}.log")
    
    logger.info(f"Starting preprocessing for {args.domain}/{args.dataset}")
    
    # Create and run the preprocessing pipeline
    pipeline = PreprocessingPipeline(config)
    processed_data = pipeline.preprocess(args.domain, args.dataset, force_reprocess=args.force)
    
    logger.info(f"Preprocessing complete: {processed_data.shape[0]} rows, {processed_data.shape[1]} columns")
    logger.info(f"Sample data:\n{processed_data.head()}")

if __name__ == "__main__":
    main()
