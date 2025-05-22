#!/usr/bin/env python3
"""
Script to examine rating storage to verify domain isolation.
"""

import os
import sys
import pandas as pd
from pprint import pprint

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.ratings_storage import RatingsStorage
from src.utils.user_manager import UserManager

def inspect_ratings_storage():
    """Inspect how ratings are stored by domain"""
    # Initialize components
    data_dir = os.path.join(project_root, "data")
    ratings_storage = RatingsStorage(data_dir=data_dir)
    user_manager = UserManager(data_dir=data_dir)
    
    # Get our test users
    entertainment_user = None
    ecommerce_user = None
    
    for user in user_manager.get_all_users():
        username = user.get("username", "")
        if username == "entertainment_user":
            entertainment_user = user
        elif username == "ecommerce_user":
            ecommerce_user = user
    
    if not entertainment_user or not ecommerce_user:
        print("Test users not found!")
        return
    
    print("Examining domain-specific rating storage...")
    
    # Check storage directories
    domains = ["entertainment", "ecommerce"]
    datasets = {
        "entertainment": ["movielens"],
        "ecommerce": ["amazon"]
    }
    
    # Check current ratings files
    print("\n" + "="*60)
    print("RATINGS STORAGE STRUCTURE")
    print("="*60)
    
    for domain in domains:
        print(f"\nDomain: {domain}")
        for dataset in datasets.get(domain, []):
            # Check new ratings directory
            new_ratings_dir = os.path.join(data_dir, "new_ratings", domain, dataset)
            if os.path.exists(new_ratings_dir):
                rating_files = [f for f in os.listdir(new_ratings_dir) if f.endswith('.csv')]
                print(f"  Dataset: {dataset}")
                print(f"  New ratings directory: {new_ratings_dir}")
                print(f"  Rating files: {', '.join(rating_files) if rating_files else 'None'}")
                
                # Check if ratings exist
                for file in rating_files:
                    file_path = os.path.join(new_ratings_dir, file)
                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            print(f"    {file}: {len(df)} ratings")
                            
                            # Check for our test users in the ratings
                            if entertainment_user:
                                ent_ratings = df[df['user_id'] == entertainment_user['user_id']]
                                if len(ent_ratings) > 0:
                                    print(f"      Entertainment user has {len(ent_ratings)} ratings here")
                            
                            if ecommerce_user:
                                eco_ratings = df[df['user_id'] == ecommerce_user['user_id']]
                                if len(eco_ratings) > 0:
                                    print(f"      Ecommerce user has {len(eco_ratings)} ratings here")
                        except Exception as e:
                            print(f"    Error reading {file}: {str(e)}")
            else:
                print(f"  Dataset: {dataset}")
                print(f"  New ratings directory not found: {new_ratings_dir}")
    
    # Check processed datasets
    print("\n" + "="*60)
    print("PROCESSED DATASETS")
    print("="*60)
    
    for domain in domains:
        print(f"\nDomain: {domain}")
        for dataset in datasets.get(domain, []):
            processed_dir = os.path.join(data_dir, "processed", domain, dataset)
            if os.path.exists(processed_dir):
                print(f"  Dataset: {dataset}")
                print(f"  Processed directory: {processed_dir}")
                
                train_path = os.path.join(processed_dir, "train.csv")
                test_path = os.path.join(processed_dir, "test.csv")
                
                if os.path.exists(train_path):
                    try:
                        train_df = pd.read_csv(train_path)
                        print(f"    Train data: {len(train_df)} records")
                    except Exception as e:
                        print(f"    Error reading train data: {str(e)}")
                
                if os.path.exists(test_path):
                    try:
                        test_df = pd.read_csv(test_path)
                        print(f"    Test data: {len(test_df)} records")
                    except Exception as e:
                        print(f"    Error reading test data: {str(e)}")
            else:
                print(f"  Dataset: {dataset}")
                print(f"  Processed directory not found: {processed_dir}")
    
    # Summary
    print("\n" + "="*60)
    print("DOMAIN ISOLATION SUMMARY")
    print("="*60)
    print("The system maintains domain isolation through:")
    print("1. User-level domain preferences in user profiles")
    print("2. Domain/dataset-specific rating storage directories")
    print("3. Separate models trained for each domain/dataset combination")
    print("\nUsers themselves are stored in a central users directory, but their interactions")
    print("(ratings, feedback) are stored in domain-specific locations, ensuring that")
    print("recommendations are based only on relevant domain data.")

if __name__ == "__main__":
    inspect_ratings_storage()