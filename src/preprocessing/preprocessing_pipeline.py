import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.preprocessing.data_loader import DataLoader
from src.preprocessing.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """
    Domain-agnostic preprocessing pipeline for recommender system data.
    Handles data cleaning, filtering, and train-test splitting.
    """
    
    def __init__(self, data_dir: str, min_ratings: int = 10, test_size: float = 0.2):
        """
        Initialize preprocessing pipeline.
        
        Args:
            data_dir: Path to the data directory
            min_ratings: Minimum number of ratings per item to keep
            test_size: Fraction of data to use for testing
        """
        self.data_dir = data_dir
        self.data_loader = DataLoader(data_dir)
        self.feature_engineer = FeatureEngineer(data_dir)
        self.min_ratings = min_ratings
        self.test_size = test_size
        self.column_map = {}
    
    def _detect_headers(self, file_path: str) -> bool:
        """
        Detect if the file has headers.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Boolean indicating if headers are present
        """
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()
                
            # Check if first line contains headers (non-numeric) and second line is numeric
            first_numeric = all(x.replace('.', '').isdigit() for x in first_line.split(','))
            second_numeric = all(x.replace('.', '').isdigit() for x in second_line.split(','))
            
            return not first_numeric and second_numeric
        except Exception as e:
            logger.error(f"Header detection error: {e}")
            return False
    
    def _analyze_column(self, col_data: pd.Series) -> Optional[str]:
        """
        Analyze column to determine its role in the dataset.
        
        Args:
            col_data: Series containing column data
            
        Returns:
            Column role (user_id, item_id, rating, timestamp) or None
        """
        sample = col_data.dropna().sample(min(100, len(col_data)), random_state=42)
        
        # Timestamp detection
        if pd.api.types.is_numeric_dtype(col_data):
            if ((sample >= 1e9) & (sample <= 2e9)).all():
                return 'timestamp'
            if ((sample >= 1e12) & (sample <= 2e12)).all():
                return 'timestamp'
        
        # Rating detection
        if pd.api.types.is_numeric_dtype(col_data):
            if np.all((sample >= 0) & (sample <= 5)):
                return 'rating'
        
        # ID detection
        if col_data.apply(lambda x: str(x).isalnum()).all():
            total = len(col_data)
            unique = col_data.nunique()
            
            if total == 0:
                return None
                
            uniqueness_ratio = unique / total
            if 0.05 < uniqueness_ratio < 0.2:
                return 'user_id'
            elif uniqueness_ratio < 0.05:
                return 'item_id'
        
        return None
    
    def _auto_map_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically map columns to standard roles based on name patterns and data analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping role names to column names
        """
        column_roles = {}
        name_patterns = {
            'user_id': ['user', 'member', 'customer', 'uid', 'u_id'],
            'item_id': ['item', 'product', 'movie', 'book', 'i_id', 'pid'],
            'rating': ['rating', 'score', 'preference'],
            'timestamp': ['time', 'date', 'timestamp']
        }

        # Pattern matching first
        for col in df.columns:
            col_lower = col.lower()
            for role, patterns in name_patterns.items():
                if any(p in col_lower for p in patterns):
                    if role not in column_roles:
                        column_roles[role] = col
                        break

        # Data analysis for remaining columns
        remaining = [c for c in df.columns if c not in column_roles.values()]
        for col in remaining:
            role = self._analyze_column(df[col])
            if role and role not in column_roles:
                column_roles[role] = col

        # Final fallback assignment
        remaining = [c for c in df.columns if c not in column_roles.values()]
        if not column_roles.get('user_id') and remaining:
            column_roles['user_id'] = remaining.pop(0)
            logger.warning(f"Auto-assigned '{column_roles['user_id']}' as user_id")
        if not column_roles.get('item_id') and remaining:
            column_roles['item_id'] = remaining.pop(0)
            logger.warning(f"Auto-assigned '{column_roles['item_id']}' as item_id")

        return column_roles
    
    def _process_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process timestamp columns to datetime format.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with processed timestamps
        """
        if 'timestamp' in df.columns and pd.api.types.is_numeric_dtype(df['timestamp']):
            unit = 'ms' if df['timestamp'].max() > 1e12 else 's'
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit=unit, errors='coerce')
        return df
    
    def preprocess(self, domain: str, dataset_name: str, force_reprocess: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess a dataset for a specific domain.
        
        Args:
            domain: Domain name (entertainment, ecommerce, education)
            dataset_name: Name of the dataset
            force_reprocess: Whether to force reprocessing
            
        Returns:
            Tuple of (train_data, test_data)
        """
        # Check if processed data already exists
        processed_dir = os.path.join(self.data_dir, "processed", domain, dataset_name)
        train_path = os.path.join(processed_dir, "train.csv")
        test_path = os.path.join(processed_dir, "test.csv")
        
        if os.path.exists(train_path) and os.path.exists(test_path) and not force_reprocess:
            logger.info(f"Using existing processed data from {processed_dir}")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            return train_data, test_data
        
        # Load raw data
        logger.info(f"Loading raw data for {domain}/{dataset_name}")
        raw_data = self.data_loader.load_dataset(domain, dataset_name)
        logger.info(f"Loaded raw data: {len(raw_data):,} rows")
        
        # Standardize columns if needed
        if not all(col in raw_data.columns for col in ['user_id', 'item_id', 'rating']):
            self.column_map = self._auto_map_columns(raw_data)
            raw_data = raw_data.rename(columns={v:k for k,v in self.column_map.items()})
            logger.info(f"Column mapping: {self.column_map}")
        
        # Process timestamps
        raw_data = self._process_timestamps(raw_data)
        if 'timestamp' in raw_data.columns:
            logger.info(f"Processed timestamps: {raw_data['timestamp'].dtype}")
        
        # Filter items with too few ratings
        original_count = len(raw_data)
        item_counts = raw_data['item_id'].value_counts()
        valid_items = item_counts[item_counts >= self.min_ratings].index
        filtered_data = raw_data[raw_data['item_id'].isin(valid_items)]
        
        logger.info(f"Filtering items with < {self.min_ratings} ratings:")
        logger.info(f"  Original: {original_count:,}")
        logger.info(f"  Remaining: {len(filtered_data):,}")
        logger.info(f"  Removed: {original_count - len(filtered_data):,} ({(original_count - len(filtered_data))/original_count:.1%})")
        
        # Create train-test split
        if 'timestamp' in filtered_data.columns:
            # Time-based split
            filtered_data = filtered_data.sort_values('timestamp')
            split_idx = int((1 - self.test_size) * len(filtered_data))
            train_data = filtered_data.iloc[:split_idx].copy()
            test_data = filtered_data.iloc[split_idx:].copy()
            logger.info(f"Time-based split: {len(train_data):,} train, {len(test_data):,} test")
        else:
            # Random split
            train_data, test_data = train_test_split(filtered_data, test_size=self.test_size, random_state=42)
            logger.info(f"Random split: {len(train_data):,} train, {len(test_data):,} test")
        
        # Add domain column
        train_data['domain'] = domain
        test_data['domain'] = domain
        
        # Save processed data
        os.makedirs(processed_dir, exist_ok=True)
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logger.info(f"Saved processed data to {processed_dir}")
        
        return train_data, test_data
    
    def create_features(self, domain: str, dataset_name: str, force_recreate: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Create features for a preprocessed dataset.
        
        Args:
            domain: Domain name
            dataset_name: Dataset name
            force_recreate: Whether to force feature recreation
            
        Returns:
            Dictionary with features
        """
        # Check if features already exist
        features_dir = os.path.join(self.data_dir, "features", domain, dataset_name)
        
        if os.path.exists(features_dir) and not force_recreate and len(os.listdir(features_dir)) > 0:
            logger.info(f"Loading existing features from {features_dir}")
            return self.feature_engineer.load_features(domain, dataset_name)
        
        # Load preprocessed data
        train_data, test_data = self.preprocess(domain, dataset_name)
        
        # Create features
        logger.info(f"Creating features for {domain}/{dataset_name}")
        features = self.feature_engineer.create_features(
            train_data=train_data,
            test_data=test_data,
            domain=domain,
            dataset_name=dataset_name
        )
        
        return features

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    data_dir = "./data"
    pipeline = PreprocessingPipeline(data_dir)
    
    # Preprocess data
    train_data, test_data = pipeline.preprocess("entertainment", "movielens")
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Create features
    features = pipeline.create_features("entertainment", "movielens")
    for name, df in features.items():
        print(f"{name}: {df.shape}")
