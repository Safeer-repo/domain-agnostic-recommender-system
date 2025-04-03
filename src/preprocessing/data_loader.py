import os
import pandas as pd
import numpy as np
import logging
import glob
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading datasets from different domains."""
    
    def __init__(self, data_dir: str):
        """
        Initialize with data directory path.
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = data_dir
    
    def load_dataset(self, domain: str, dataset_name: str) -> pd.DataFrame:
        """
        Load a specific dataset from a domain.
        
        Args:
            domain: Domain name (entertainment, ecommerce, education)
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame with the loaded data
        """
        # Construct path to the raw data
        raw_data_path = os.path.join(self.data_dir, "raw", domain, dataset_name)
        
        if domain == "entertainment" and dataset_name == "movielens":
            return self._load_movielens(raw_data_path)
        elif domain == "ecommerce" and dataset_name == "amazon_electronics":
            return self._load_amazon(raw_data_path)
        elif domain == "education" and dataset_name == "open_university":
            return self._load_open_university(raw_data_path)
        else:
            raise ValueError(f"Unsupported domain/dataset: {domain}/{dataset_name}")
    
    def _load_movielens(self, data_path: str) -> pd.DataFrame:
        """
        Load MovieLens dataset.
        
        Args:
            data_path: Path to MovieLens data directory
            
        Returns:
            DataFrame with user-movie interactions
        """
        # Check for MovieLens 100K format (u.data file)
        ratings_path_100k = os.path.join(data_path, "u.data")
        
        if os.path.exists(ratings_path_100k):
            # Load MovieLens 100K format
            logger.info(f"Loading MovieLens 100K ratings from {ratings_path_100k}")
            
            # MovieLens 100K format: user id | item id | rating | timestamp
            ratings = pd.read_csv(
                ratings_path_100k, 
                sep='\t', 
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                engine='python'
            )
            
            logger.info(f"Loaded {len(ratings)} MovieLens ratings")
            return ratings
        
        # Check for MovieLens 32M format (ml-32m/ratings.csv file)
        ratings_path_32m = os.path.join(data_path, "ml-32m", "ratings.csv")
        
        if os.path.exists(ratings_path_32m):
            logger.info(f"Loading MovieLens 32M ratings from {ratings_path_32m}")
            
            # MovieLens 32M format: userId,movieId,rating,timestamp
            ratings = pd.read_csv(ratings_path_32m)
            
            # Standardize column names
            if 'userId' in ratings.columns and 'movieId' in ratings.columns:
                ratings = ratings.rename(columns={
                    'userId': 'user_id',
                    'movieId': 'item_id'
                })
            
            logger.info(f"Loaded {len(ratings)} MovieLens ratings")
            return ratings
        
        # Try finding any ratings.csv file recursively
        ratings_paths = glob.glob(os.path.join(data_path, "**/ratings.csv"), recursive=True)
        
        if ratings_paths:
            ratings_path = ratings_paths[0]
            logger.info(f"Loading MovieLens ratings from {ratings_path}")
            
            # MovieLens newer format: userId,movieId,rating,timestamp
            ratings = pd.read_csv(ratings_path)
            
            # Standardize column names
            if 'userId' in ratings.columns and 'movieId' in ratings.columns:
                ratings = ratings.rename(columns={
                    'userId': 'user_id',
                    'movieId': 'item_id'
                })
            
            logger.info(f"Loaded {len(ratings)} MovieLens ratings")
            return ratings
        
        # If none of the expected formats are found
        raise FileNotFoundError(f"MovieLens ratings file not found in {data_path}. "
                              f"Files available: {os.listdir(data_path)}")
    
    def _load_amazon(self, data_path: str) -> pd.DataFrame:
        """
        Load Amazon Electronics dataset.
        
        Args:
            data_path: Path to Amazon data directory
            
        Returns:
            DataFrame with user-product interactions
        """
        # Placeholder for Amazon dataset loading logic
        reviews_path = os.path.join(data_path, "reviews.csv")
        
        if not os.path.exists(reviews_path):
            raise FileNotFoundError(f"Amazon reviews file not found at {reviews_path}")
        
        logger.info(f"Loading Amazon reviews from {reviews_path}")
        
        # Adjust column names based on actual dataset format
        reviews = pd.read_csv(reviews_path)
        
        # Rename columns to standard format if needed
        if 'reviewerID' in reviews.columns and 'asin' in reviews.columns:
            reviews = reviews.rename(columns={
                'reviewerID': 'user_id',
                'asin': 'item_id',
                'overall': 'rating'
            })
        
        logger.info(f"Loaded {len(reviews)} Amazon reviews")
        return reviews
    
    def _load_open_university(self, data_path: str) -> pd.DataFrame:
        """
        Load Open University Learning Analytics dataset.
        
        Args:
            data_path: Path to Open University data directory
            
        Returns:
            DataFrame with student-course interactions
        """
        # Placeholder for Open University dataset loading logic
        interactions_path = os.path.join(data_path, "studentAssessment.csv")
        
        if not os.path.exists(interactions_path):
            raise FileNotFoundError(f"Open University interactions file not found at {interactions_path}")
        
        logger.info(f"Loading Open University interactions from {interactions_path}")
        
        # Load interactions
        interactions = pd.read_csv(interactions_path)
        
        # Rename columns to standard format
        if 'id_student' in interactions.columns and 'id_assessment' in interactions.columns:
            interactions = interactions.rename(columns={
                'id_student': 'user_id',
                'id_assessment': 'item_id',
                'score': 'rating'
            })
        
        logger.info(f"Loaded {len(interactions)} Open University interactions")
        return interactions
