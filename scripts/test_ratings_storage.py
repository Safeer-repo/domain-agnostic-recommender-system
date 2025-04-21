#!/usr/bin/env python3
"""
Script to test the RatingsStorage functionality.
Usage: python scripts/test_ratings_storage.py
"""

import os
import sys
import logging
import time
import pandas as pd
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.ratings_storage import RatingsStorage

def test_single_rating():
    """Test storing a single rating."""
    storage = RatingsStorage(data_dir=project_root)
    result = storage.store_rating(
        domain="entertainment",
        dataset="movielens",
        user_id=42,
        item_id=123,
        rating=4.5,
        timestamp=int(time.time()),  # Explicitly use integer timestamp
        source="test"
    )
    
    logging.info(f"Single rating stored: {result}")
    return result

def test_batch_ratings():
    """Test storing multiple ratings at once."""
    storage = RatingsStorage(data_dir=project_root)
    
    # Current timestamp for consistent testing
    current_time = int(time.time())
    
    # Create a batch of test ratings with explicit integer timestamps
    ratings = [
        {"user_id": 1, "item_id": 101, "rating": 5.0, "timestamp": current_time, "source": "explicit"},
        {"user_id": 1, "item_id": 102, "rating": 3.5, "timestamp": current_time, "source": "explicit"},
        {"user_id": 2, "item_id": 101, "rating": 4.0, "timestamp": current_time, "source": "implicit"},
        {"user_id": 3, "item_id": 103, "rating": 2.0, "timestamp": current_time, "source": "explicit"}
    ]
    
    count = storage.store_ratings_batch(
        domain="entertainment",
        dataset="movielens",
        ratings=ratings
    )
    
    logging.info(f"Batch ratings stored: {count}")
    return count

def test_retrieve_ratings():
    """Test retrieving stored ratings."""
    storage = RatingsStorage(data_dir=project_root)
    
    # Store timestamp for filtering
    timestamp_before = int(time.time())
    logging.info(f"Timestamp before: {timestamp_before}")
    
    # Add more ratings
    storage.store_rating(
        domain="entertainment",
        dataset="movielens",
        user_id=10,
        item_id=200,
        rating=3.0,
        timestamp=timestamp_before,
        source="test"
    )
    
    # Sleep briefly to ensure the timestamps are different
    time.sleep(1)
    
    timestamp_after = int(time.time())
    logging.info(f"Timestamp after: {timestamp_after}")
    
    # Add more ratings with the newer timestamp
    storage.store_rating(
        domain="entertainment",
        dataset="movielens",
        user_id=11,
        item_id=201,
        rating=4.0,
        timestamp=timestamp_after,
        source="test"
    )
    
    # Retrieve all ratings
    all_ratings = storage.get_new_ratings(
        domain="entertainment",
        dataset="movielens"
    )
    
    # Retrieve ratings after timestamp
    recent_ratings = storage.get_new_ratings(
        domain="entertainment",
        dataset="movielens",
        since_timestamp=timestamp_after
    )
    
    logging.info(f"Total ratings: {len(all_ratings)}")
    logging.info(f"Recent ratings (since {timestamp_after}): {len(recent_ratings)}")
    
    # Display ratings
    if not all_ratings.empty:
        logging.info("\nAll ratings:")
        logging.info(all_ratings)
    
    if not recent_ratings.empty:
        logging.info("\nRecent ratings:")
        logging.info(recent_ratings)
    else:
        logging.info("\nNo recent ratings found after timestamp filter")
    
    return all_ratings, recent_ratings

def test_clear_processed():
    """Test clearing processed ratings."""
    storage = RatingsStorage(data_dir=project_root)
    
    # Get current timestamp
    current_time = int(time.time())
    
    # Add some ratings with timestamps in the past (explicit integers)
    past_ratings = [
        {
            "user_id": 20, 
            "item_id": 300, 
            "rating": 4.0, 
            "timestamp": current_time - 3600,  # 1 hour ago
            "source": "test"
        },
        {
            "user_id": 21, 
            "item_id": 301, 
            "rating": 3.5, 
            "timestamp": current_time - 7200,  # 2 hours ago
            "source": "test"
        }
    ]
    
    past_count = storage.store_ratings_batch(
        domain="entertainment",
        dataset="movielens",
        ratings=past_ratings
    )
    logging.info(f"Stored {past_count} past ratings with timestamps {current_time-3600} and {current_time-7200}")
    
    # Add some ratings with current timestamp
    current_ratings = [
        {"user_id": 22, "item_id": 302, "rating": 5.0, "timestamp": current_time, "source": "test"},
        {"user_id": 23, "item_id": 303, "rating": 2.0, "timestamp": current_time, "source": "test"}
    ]
    
    current_count = storage.store_ratings_batch(
        domain="entertainment",
        dataset="movielens",
        ratings=current_ratings
    )
    logging.info(f"Stored {current_count} current ratings with timestamp {current_time}")
    
    # Get count before clearing
    count_before = storage.get_new_ratings_count(
        domain="entertainment",
        dataset="movielens"
    )
    
    # Set clear threshold to 30 minutes ago
    clear_threshold = current_time - 1800  # 30 minutes ago
    logging.info(f"Clear threshold timestamp: {clear_threshold}")
    
    # Verify ratings that should be cleared
    ratings_to_check = storage.get_new_ratings(
        domain="entertainment",
        dataset="movielens"
    )
    
    old_ratings_count = sum(ratings_to_check['timestamp'] < clear_threshold)
    logging.info(f"Ratings older than threshold: {old_ratings_count}")
    
    # Clear processed ratings (older than 30 minutes)
    removed = storage.clear_processed_ratings(
        domain="entertainment",
        dataset="movielens",
        before_timestamp=clear_threshold
    )
    
    # Get count after clearing
    count_after = storage.get_new_ratings_count(
        domain="entertainment",
        dataset="movielens"
    )
    
    # Verify what remains
    remaining_ratings = storage.get_new_ratings(
        domain="entertainment",
        dataset="movielens"
    )
    
    logging.info(f"Ratings before clearing: {count_before}")
    logging.info(f"Ratings expected to be removed: {old_ratings_count}")
    logging.info(f"Ratings actually removed: {removed}")
    logging.info(f"Ratings after clearing: {count_after}")
    logging.info(f"Verification - remaining ratings should all have timestamp >= {clear_threshold}")
    
    if not remaining_ratings.empty:
        min_remaining_timestamp = remaining_ratings['timestamp'].min()
        logging.info(f"Minimum timestamp in remaining ratings: {min_remaining_timestamp}")
        if min_remaining_timestamp < clear_threshold:
            logging.warning(f"Some ratings with timestamps before {clear_threshold} were not removed!")
        else:
            logging.info("All remaining ratings have timestamps after the threshold - correct!")
    
    return count_before, removed, count_after

def clean_test_data():
    """Clean up test data to start fresh."""
    storage_dir = os.path.join(project_root, "new_ratings", "entertainment", "movielens")
    if os.path.exists(storage_dir):
        for filename in os.listdir(storage_dir):
            if filename.startswith("ratings_") and filename.endswith(".csv"):
                file_path = os.path.join(storage_dir, filename)
                try:
                    os.remove(file_path)
                    logging.info(f"Removed test file: {file_path}")
                except Exception as e:
                    logging.error(f"Failed to remove {file_path}: {str(e)}")
    
    logging.info("Test environment cleaned")

def main():
    """Main function to test ratings storage functionality."""
    logging.basicConfig(
        level=logging.DEBUG,  # Use DEBUG level for more detailed logs
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    logging.info("Testing RatingsStorage functionality")
    
    try:
        # Clean up any existing test data
        clean_test_data()
        
        # Test storing a single rating
        logging.info("\n=== Testing single rating storage ===")
        test_single_rating()
        
        # Test storing batch ratings
        logging.info("\n=== Testing batch ratings storage ===")
        test_batch_ratings()
        
        # Test retrieving ratings
        logging.info("\n=== Testing ratings retrieval ===")
        test_retrieve_ratings()
        
        # Test clearing processed ratings
        logging.info("\n=== Testing clearing processed ratings ===")
        test_clear_processed()
        
        logging.info("\nAll tests completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()