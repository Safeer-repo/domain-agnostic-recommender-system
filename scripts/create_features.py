#!/usr/bin/env python3
"""
Script to run feature engineering.
Usage: python scripts/create_features.py --domain DOMAIN --dataset DATASET [--force]
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
    """Main function to run feature engineering."""
    parser = argparse.ArgumentParser(description="Run feature engineering")
    parser.add_argument("--domain", choices=["entertainment", "ecommerce", "education"],
                        required=True, help="Domain to create features for")
    parser.add_argument("--dataset", required=True, help="Dataset name to create features for")
    parser.add_argument("--force", action="store_true", help="Force feature recreation even if features exist")
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
    pipeline = PreprocessingPipeline(data_dir=data_dir)
    
    try:
        # Create features
        features = pipeline.create_features(
            domain=args.domain,
            dataset_name=args.dataset,
            force_recreate=args.force
        )
        
        # Print feature information
        logging.info(f"Feature engineering complete")
        for name, df in features.items():
            logging.info(f"  {name}: {df.shape}, Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Print sample columns and values
            if len(df) > 0:
                logging.info(f"  {name} columns: {', '.join(df.columns[:5])}...")
                
                # For numeric columns, show some statistics
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    logging.info(f"  Sample stats for {col}: min={df[col].min():.2f}, mean={df[col].mean():.2f}, max={df[col].max():.2f}")
        
    except Exception as e:
        logging.error(f"Error during feature engineering: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
