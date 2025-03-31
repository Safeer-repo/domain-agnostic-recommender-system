#!/usr/bin/env python3
"""
Script to inspect the MovieLens dataset.
"""

import os
import pandas as pd

def main():
    """Main function to inspect the MovieLens dataset."""
    # Path to the MovieLens data
    data_path = os.path.join("data", "raw", "entertainment", "movielens", "ratings.csv")
    
    # Check if the file exists
    if not os.path.exists(data_path):
        print(f"Error: MovieLens data file not found at {data_path}")
        return
    
    # Load the data
    print(f"Loading MovieLens data from {data_path}")
    ratings = pd.read_csv(data_path)
    
    # Standardize column names
    ratings = ratings.rename(columns={
        'userId': 'user_id',
        'movieId': 'item_id'
    })
    
    # Print basic statistics
    print(f"\nDataset Summary:")
    print(f"Number of ratings: {len(ratings)}")
    print(f"Number of users: {ratings['user_id'].nunique()}")
    print(f"Number of movies: {ratings['item_id'].nunique()}")
    print(f"Rating range: {ratings['rating'].min()} - {ratings['rating'].max()}")
    
    # Print first few rows
    print("\nFirst 5 rows:")
    print(ratings.head())
    
    # Check data distribution
    print("\nRating distribution:")
    print(ratings['rating'].value_counts().sort_index())
    
    # Check memory usage
    print("\nMemory usage:")
    print(ratings.memory_usage(deep=True))

if __name__ == "__main__":
    main()
