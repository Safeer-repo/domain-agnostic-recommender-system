#!/usr/bin/env python3
"""
Script to import historical ratings for demo users from Amazon dataset
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.ratings_storage import RatingsStorage
from src.utils.user_manager import UserManager

# Initialize components
data_dir = os.path.abspath("./data")
ratings_storage = RatingsStorage(data_dir=data_dir)
user_manager = UserManager(data_dir=data_dir)

# Define demo users (same as from your create_amazon_demo_users.py)
demo_users = [
    {"original_id": "AKZTRHX9HPT4J", "username": "amazon_user_1", "password": "password123"},
    {"original_id": "AD1MRUOBFE8FG", "username": "amazon_user_2", "password": "password123"},
    {"original_id": "A12L7POEN3MQEM", "username": "amazon_user_3", "password": "password123"},
    {"original_id": "A1JNFS3IZ1XIXN", "username": "amazon_user_4", "password": "password123"},
    {"original_id": "A1NJY361HDQG5B", "username": "amazon_user_5", "password": "password123"},
]

def convert_date_to_timestamp(date_str):
    """Convert date string to Unix timestamp"""
    try:
        # Try parsing as YYYY-MM-DD format
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return int(dt.timestamp())
    except ValueError:
        try:
            # Try parsing as datetime if it includes time
            dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            return int(dt.timestamp())
        except ValueError:
            # If all else fails, return current timestamp
            return int(datetime.now().timestamp())

def import_demo_user_ratings():
    """Import historical ratings for demo users from Amazon dataset"""
    
    # Load Amazon ratings
    amazon_ratings_path = os.path.join(project_root, "data", "processed", "ecommerce", "amazon", "train.csv")
    print(f"Loading Amazon ratings from {amazon_ratings_path}")
    amazon_ratings = pd.read_csv(amazon_ratings_path)
    
    # For each demo user
    for demo_user in demo_users:
        username = demo_user["username"]
        original_id = demo_user["original_id"]
        
        # Find their system UUID
        user_id = user_manager.get_user_id(username)
        
        if not user_id:
            print(f"User {username} not found. Skipping.")
            continue
            
        print(f"\nProcessing {username} (UUID: {user_id})")
        
        # Get their original ratings
        user_ratings = amazon_ratings[amazon_ratings['user_id'] == original_id]
        print(f"Found {len(user_ratings)} ratings for original ID {original_id}")
        
        if len(user_ratings) == 0:
            print(f"No ratings found for {username}")
            continue
        
        # Convert to the format expected by ratings_storage
        ratings_to_store = []
        for _, row in user_ratings.iterrows():
            rating_data = {
                "user_id": user_id,  # Keep as string UUID
                "item_id": row["item_id"],
                "rating": float(row["rating"]),
                "timestamp": convert_date_to_timestamp(row["timestamp"]),
                "source": "historical"
            }
            ratings_to_store.append(rating_data)
        
        # Store the ratings - handle each rating individually to avoid UUID conversion issue
        stored_count = 0
        for rating in ratings_to_store:
            try:
                success = ratings_storage.store_rating(
                    domain="ecommerce",
                    dataset="amazon_electronics",
                    user_id=rating["user_id"],  # Pass UUID as string
                    item_id=rating["item_id"],
                    rating=rating["rating"],
                    timestamp=rating["timestamp"],
                    source=rating["source"]
                )
                if success:
                    stored_count += 1
            except Exception as e:
                print(f"Error storing rating: {e}")
        
        print(f"Successfully imported {stored_count} ratings for {username}")

if __name__ == "__main__":
    print("Importing historical ratings for Amazon demo users...")
    import_demo_user_ratings()
    print("\nImport completed!")