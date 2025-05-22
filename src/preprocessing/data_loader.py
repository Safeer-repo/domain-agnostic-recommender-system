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
        elif domain == "ecommerce" and dataset_name == "amazon":
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
        # Look for the Amazon reviews file
        potential_filenames = [
            "reviews.csv",
            "amazon_reviews.csv",
            "amazon_product_reviews.csv",
            "ratings_Electronics.csv"
        ]
        
        found_file = None
        for filename in potential_filenames:
            file_path = os.path.join(data_path, filename)
            if os.path.exists(file_path):
                found_file = file_path
                break
        
        # If no specific file is found, try looking for any CSV file
        if not found_file:
            csv_files = glob.glob(os.path.join(data_path, "*.csv"))
            if csv_files:
                found_file = csv_files[0]
        
        if not found_file:
            raise FileNotFoundError(f"Amazon reviews file not found in {data_path}")
        
        logger.info(f"Loading Amazon reviews from {found_file}")
        
        # Try loading with headers first, then without if needed
        try:
            # First try with headers
            reviews = pd.read_csv(found_file)
            
            # Check if columns look like data instead of headers
            if reviews.columns[0].replace('.', '').isdigit() or len(reviews.columns) < 3:
                logger.info("First row appears to be data; loading without headers")
                reviews = pd.read_csv(found_file, header=None, 
                                    names=['user_id', 'item_id', 'rating', 'timestamp'])
            else:
                # If headers exist, try to map them
                column_mapping = {
                    'userId': 'user_id',
                    'productId': 'item_id',
                    'Rating': 'rating',
                    'reviewerID': 'user_id',
                    'asin': 'item_id',
                    'overall': 'rating'
                }
                
                # Only rename columns that exist in the dataset
                rename_dict = {old: new for old, new in column_mapping.items() if old in reviews.columns}
                if rename_dict:
                    reviews = reviews.rename(columns=rename_dict)
            
            logger.info(f"Successfully loaded Amazon reviews")
        except Exception as e:
            logger.error(f"Error loading Amazon reviews: {str(e)}")
            raise
        
        # Ensure required columns exist
        required_cols = ['user_id', 'item_id', 'rating']
        missing_cols = [col for col in required_cols if col not in reviews.columns]
        
        if missing_cols:
            raise ValueError(f"Required columns {missing_cols} not found in Amazon reviews dataset. "
                            f"Available columns: {reviews.columns.tolist()}")
        
        # Ensure 'rating' column is numeric
        if reviews['rating'].dtype not in [np.float64, np.int64]:
            logger.warning("Converting 'rating' column to numeric")
            reviews['rating'] = pd.to_numeric(reviews['rating'], errors='coerce')
            
            # Remove rows with non-numeric ratings
            invalid_ratings = reviews['rating'].isna().sum()
            if invalid_ratings > 0:
                logger.warning(f"Removing {invalid_ratings} rows with invalid ratings")
                reviews = reviews.dropna(subset=['rating'])
        
        logger.info(f"Loaded {len(reviews)} Amazon reviews with {reviews['user_id'].nunique():,} unique users "
                    f"and {reviews['item_id'].nunique():,} unique items")
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
    
    def load_amazon_from_downloads(self, file_path: str) -> pd.DataFrame:
        """
        Load Amazon Electronics dataset from a specific file path.
        
        Args:
            file_path: Full path to the Amazon reviews file
            
        Returns:
            DataFrame with user-product interactions
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Amazon reviews file not found at {file_path}")
        
        logger.info(f"Loading Amazon reviews from {file_path}")
        
        # Load the CSV file without headers (since we know the column order)
        try:
            reviews = pd.read_csv(file_path, header=None, 
                                names=['user_id', 'item_id', 'rating', 'timestamp'])
            logger.info(f"Successfully loaded Amazon reviews with predefined column names")
        except Exception as e:
            logger.error(f"Error loading Amazon reviews: {str(e)}")
            raise
        
        # Ensure rating column is numeric
        if reviews['rating'].dtype not in [np.float64, np.int64]:
            logger.warning("Converting 'rating' column to numeric")
            reviews['rating'] = pd.to_numeric(reviews['rating'], errors='coerce')
            
            # Remove rows with invalid ratings
            invalid_ratings = reviews['rating'].isna().sum()
            if invalid_ratings > 0:
                logger.warning(f"Removing {invalid_ratings} rows with invalid ratings")
                reviews = reviews.dropna(subset=['rating'])
        
        logger.info(f"Loaded {len(reviews)} Amazon reviews with {reviews['user_id'].nunique():,} unique users "
                    f"and {reviews['item_id'].nunique():,} unique items")
        
        # Create target directory in project structure
        target_dir = os.path.join(self.data_dir, "raw", "ecommerce", "amazon")
        os.makedirs(target_dir, exist_ok=True)
        
        # Save to project structure
        target_path = os.path.join(target_dir, "reviews.csv")
        logger.info(f"Saving Amazon reviews to project structure at {target_path}")
        reviews.to_csv(target_path, index=False)
        
        return reviews