#!/usr/bin/env python3
"""
Script to test the data loader with the MovieLens dataset.
"""

import os
import sys
import logging

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessing.data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    """Main function to test the data loader."""
    # Use the project_root to get the data directory
    data_dir = os.path.join(project_root, "data")
    
    # Create data loader
    data_loader = DataLoader(data_dir)
    
    # Load MovieLens dataset
    try:
        ratings = data_loader.load_dataset("entertainment", "movielens")
        print(f"\nSuccessfully loaded MovieLens dataset")
        print(f"Number of ratings: {len(ratings)}")
        print(f"Number of users: {ratings['user_id'].nunique()}")
        print(f"Number of movies: {ratings['item_id'].nunique()}")
        print(f"Rating range: {ratings['rating'].min()} - {ratings['rating'].max()}")
        print("\nFirst 5 rows:")
        print(ratings.head())
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")

if __name__ == "__main__":
    main()
