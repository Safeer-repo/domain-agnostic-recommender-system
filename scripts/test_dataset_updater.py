#!/usr/bin/env python3
"""
Script to test the DatasetUpdater functionality.
Usage: python scripts/test_dataset_updater.py
"""

import os
import sys
import logging
import time
import pandas as pd
import numpy as np
import json
import shutil
from pprint import pformat

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.ratings_storage import RatingsStorage
from src.utils.dataset_updater import DatasetUpdater

def clean_test_data():
    """Clean up test data directories to start fresh."""
    for dir_name in ["new_ratings", "datasets", "staging", "dataset_stats"]:
        dir_path = os.path.join(project_root, dir_name)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            logging.info(f"Removed test directory: {dir_path}")
    
    logging.info("Test environment cleaned")

def create_test_ratings():
    """Create test ratings for the updater to process."""
    storage = RatingsStorage(data_dir=project_root)
    
    # Create a batch of normal ratings
    normal_ratings = [
        {"user_id": 1, "item_id": 101, "rating": 4.5, "source": "explicit"},
        {"user_id": 2, "item_id": 102, "rating": 3.0, "source": "explicit"},
        {"user_id": 3, "item_id": 103, "rating": 5.0, "source": "explicit"},
        {"user_id": 1, "item_id": 104, "rating": 2.5, "source": "explicit"},
        {"user_id": 4, "item_id": 105, "rating": 3.5, "source": "explicit"},
        {"user_id": 5, "item_id": 106, "rating": 4.0, "source": "explicit"},
        {"user_id": 6, "item_id": 107, "rating": 3.0, "source": "explicit"},
        {"user_id": 7, "item_id": 108, "rating": 4.5, "source": "explicit"},
        {"user_id": 8, "item_id": 109, "rating": 2.0, "source": "explicit"},
        {"user_id": 9, "item_id": 110, "rating": 3.5, "source": "explicit"}
    ]
    
    # Add some abnormal ratings for testing validation
    abnormal_ratings = [
        {"user_id": 10, "item_id": 111, "rating": 6.0, "source": "explicit"},  # Out of range
        {"user_id": 11, "item_id": 112, "rating": -1.0, "source": "explicit"},  # Out of range
        {"user_id": 12, "item_id": 113, "rating": 3.0, "source": "explicit"},  
        {"user_id": 12, "item_id": 114, "rating": 3.0, "source": "explicit"},  
        {"user_id": 12, "item_id": 115, "rating": 3.0, "source": "explicit"},  
        {"user_id": 12, "item_id": 116, "rating": 3.0, "source": "explicit"},  
        {"user_id": 12, "item_id": 117, "rating": 3.0, "source": "explicit"}   # Many ratings from same user
    ]
    
    # Store normal ratings first
    count_normal = storage.store_ratings_batch(
        domain="entertainment",
        dataset="movielens",
        ratings=normal_ratings
    )
    
    logging.info(f"Stored {count_normal} normal ratings")
    
    # Store abnormal ratings
    count_abnormal = storage.store_ratings_batch(
        domain="entertainment",
        dataset="movielens",
        ratings=abnormal_ratings
    )
    
    logging.info(f"Stored {count_abnormal} abnormal ratings")
    
    return storage

def create_test_dataset():
    """Create a test ratings.csv in the datasets directory for merging."""
    dataset_dir = os.path.join(project_root, "datasets", "entertainment", "movielens")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create existing ratings with a good distribution
    existing_ratings = pd.DataFrame({
        "user_id": np.random.randint(1, 100, 100),
        "item_id": np.random.randint(1, 200, 100),
        "rating": np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], 100),
        "timestamp": np.full(100, int(time.time()) - 86400),  # Yesterday
        "source": np.full(100, "explicit")
    })
    
    # Save to CSV
    ratings_file = os.path.join(dataset_dir, "ratings.csv")
    existing_ratings.to_csv(ratings_file, index=False)
    
    logging.info(f"Created test dataset with {len(existing_ratings)} ratings at {ratings_file}")
    
def test_staging():
    """Test staging functionality of DatasetUpdater."""
    # Create test ratings
    storage = create_test_ratings()
    
    # Create DatasetUpdater
    updater = DatasetUpdater(data_dir=project_root, ratings_storage=storage)
    
    # Stage new ratings
    staged_ratings = updater.stage_new_ratings(
        domain="entertainment",
        dataset="movielens"
    )
    
    logging.info(f"Staged {len(staged_ratings)} ratings")
    
    # Verify staging directory structure
    staging_dir = os.path.join(project_root, "staging", "entertainment", "movielens")
    staging_files = [f for f in os.listdir(staging_dir) if f.startswith("staged_ratings_")]
    
    logging.info(f"Found {len(staging_files)} staging files: {staging_files}")
    
    return updater, staged_ratings

def test_validation(updater):
    """Test validation functionality of DatasetUpdater."""
    # Validate staged ratings
    validated_ratings, report = updater.validate_staged_ratings(
        domain="entertainment",
        dataset="movielens"
    )
    
    logging.info(f"Validation status: {report['status']}")
    logging.info(f"Validated {len(validated_ratings)} out of {report['total_ratings']} staged ratings")
    
    # Log validation checks
    for check_name, check_results in report["validation_checks"].items():
        logging.info(f"Validation check '{check_name}': {check_results}")
    
    # Verify filtered ratings
    filtered_count = report["total_ratings"] - len(validated_ratings)
    logging.info(f"Filtered out {filtered_count} ratings during validation")
    
    return validated_ratings, report

def test_merging(updater, validated_ratings):
    """Test merging functionality of DatasetUpdater."""
    # Create test dataset for merging
    create_test_dataset()
    
    # Get count before merge
    dataset_path = os.path.join(project_root, "datasets", "entertainment", "movielens", "ratings.csv")
    before_count = len(pd.read_csv(dataset_path))
    
    # Merge validated ratings
    success = updater.merge_validated_ratings(
        domain="entertainment",
        dataset="movielens",
        validated_ratings=validated_ratings,
        incremental=True
    )
    
    # Verify merge results
    if success:
        after_count = len(pd.read_csv(dataset_path))
        new_records = after_count - before_count
        
        logging.info(f"Merge successful: Added {new_records} new records to dataset")
        logging.info(f"Dataset size before: {before_count}, after: {after_count}")
        
        # Check for backup file
        backup_files = [f for f in os.listdir(os.path.dirname(dataset_path)) if f.startswith("ratings_backup_")]
        logging.info(f"Found {len(backup_files)} backup files: {backup_files}")
    else:
        logging.error("Merge failed")
    
    return success, before_count, (after_count if success else 0)

def test_full_pipeline():
    """Test the complete pipeline functionality of DatasetUpdater."""
    # Create test ratings
    storage = create_test_ratings()
    
    # Create test dataset
    create_test_dataset()
    
    # Create DatasetUpdater
    updater = DatasetUpdater(data_dir=project_root, ratings_storage=storage)
    
    # Run the complete pipeline
    report = updater.process_new_ratings(
        domain="entertainment",
        dataset="movielens",
        incremental=True
    )
    
    logging.info(f"Process report:")
    logging.info(pformat(report))
    
    # Verify overall process status
    logging.info(f"Overall process status: {report['overall_status']}")
    
    # Check if stats were generated
    stats_path = os.path.join(project_root, "dataset_stats", "entertainment", "movielens", "dataset_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        logging.info(f"Dataset stats generated:")
        logging.info(pformat(stats))
    
    return report

def main():
    """Main function to test dataset updater functionality."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    logging.info("Testing DatasetUpdater functionality")
    
    try:
        # Clean up any existing test data
        clean_test_data()
        
        # Test staging
        logging.info("\n=== Testing staging new ratings ===")
        updater, staged_ratings = test_staging()
        
        # Test validation
        logging.info("\n=== Testing validation of staged ratings ===")
        validated_ratings, validation_report = test_validation(updater)
        
        # Test merging
        logging.info("\n=== Testing merging of validated ratings ===")
        merge_success, before_count, after_count = test_merging(updater, validated_ratings)
        
        # Test full pipeline
        logging.info("\n=== Testing full processing pipeline ===")
        pipeline_report = test_full_pipeline()
        
        logging.info("\nAll tests completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()