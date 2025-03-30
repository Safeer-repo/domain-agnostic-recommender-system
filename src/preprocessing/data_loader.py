import os
import pandas as pd
import logging
from typing import Dict, Any, Optional

from src.utils.config import Config

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading datasets from different domains."""
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.data_dir = config.data_dir
    
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
        # For MovieLens 100K
        ratings_path = os.path.join(data_path, "u.data")
        
        # Check if file exists
        if not os.path.exists(ratings_path):
            raise FileNotFoundError(f"MovieLens ratings file not found at {ratings_path}")
        
        # Load ratings
        logger.info(f"Loading MovieLens ratings from {ratings_path}")
        
        # MovieLens 100K format: user id | item id | rating | timestamp
        ratings = pd.read_csv(
            ratings_path, 
            sep='\t', 
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        logger.info(f"Loaded {len(ratings)} MovieLens ratings")
        return ratings
    
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
