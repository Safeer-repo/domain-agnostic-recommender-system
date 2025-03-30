import os
import pandas as pd
import logging
from typing import Dict, Any, Optional

from src.preprocessing.data_loader import DataLoader
from src.utils.config import Config

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """Basic preprocessing pipeline for recommender system data."""
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.data_loader = DataLoader(config)
    
    def preprocess(self, domain: str, dataset_name: str, force_reprocess: bool = False) -> pd.DataFrame:
        """
        Preprocess a specific dataset.
        
        Args:
            domain: Domain name (entertainment, ecommerce, education)
            dataset_name: Name of the dataset
            force_reprocess: Whether to force reprocessing even if processed data exists
            
        Returns:
            DataFrame with preprocessed data
        """
        # Check if processed data already exists
        processed_path = os.path.join(
            self.config.data_dir, 
            "processed", 
            domain, 
            f"{dataset_name}.parquet"
        )
        
        if os.path.exists(processed_path) and not force_reprocess:
            logger.info(f"Using existing processed data from {processed_path}")
            return pd.read_parquet(processed_path)
        
        # Load raw data
        raw_data = self.data_loader.load_dataset(domain, dataset_name)
        
        # Apply basic preprocessing
        preprocessed_data = self._basic_preprocessing(raw_data, domain, dataset_name)
        
        # Save processed data
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        preprocessed_data.to_parquet(processed_path, index=False)
        logger.info(f"Saved processed data to {processed_path}")
        
        return preprocessed_data
    
    def _basic_preprocessing(self, data: pd.DataFrame, domain: str, dataset_name: str) -> pd.DataFrame:
        """
        Apply basic preprocessing steps to the data.
        
        Args:
            data: Raw data
            domain: Domain name
            dataset_name: Dataset name
            
        Returns:
            Preprocessed DataFrame
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Handle missing values
        if df.isna().sum().sum() > 0:
            logger.info(f"Found {df.isna().sum().sum()} missing values")
            df = df.dropna(subset=['user_id', 'item_id'])  # Drop rows with missing user or item IDs
        
        # Ensure user_id and item_id are integers
        if 'user_id' in df.columns:
            df['user_id'] = df['user_id'].astype(int)
        if 'item_id' in df.columns:
            df['item_id'] = df['item_id'].astype(int)
        
        # Handle timestamps if available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Add domain column
        df['domain'] = domain
        
        logger.info(f"Basic preprocessing complete: {len(df)} rows remaining")
        return df

if __name__ == "__main__":
    # Example usage
    from src.utils.config import load_config
    
    config = load_config()
    pipeline = PreprocessingPipeline(config)
    
    # Preprocess MovieLens dataset
    processed_data = pipeline.preprocess("entertainment", "movielens", force_reprocess=True)
    print(f"Processed data shape: {processed_data.shape}")
