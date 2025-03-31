import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering for recommender systems.
    Extracts user, item, and interaction features from preprocessed data.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize feature engineer.
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = data_dir
        self.features = {}
    
    def _create_user_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-related features.
        
        Args:
            data: Preprocessed DataFrame
            
        Returns:
            DataFrame with user features
        """
        logger.info("Creating user features")
        
        # Ensure data types are correct
        if 'timestamp' in data.columns:
            # Convert timestamp to datetime if it's not already
            if isinstance(data['timestamp'].iloc[0], str):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Group by user_id
        user_groups = data.groupby('user_id')
        
        # Create basic features
        user_features = pd.DataFrame({
            'user_id': list(user_groups.groups.keys()),
            'user_rating_count': user_groups.size(),
            'user_avg_rating': user_groups['rating'].mean(),
            'user_rating_std': user_groups['rating'].std().fillna(0),
            'user_max_rating': user_groups['rating'].max(),
            'user_min_rating': user_groups['rating'].min()
        })
        
        # Add rating behavior features
        if 'timestamp' in data.columns:
            try:
                # Calculate time-based features safely
                logger.info("Adding time-based user features")
                
                # Get the last rating timestamp for each user
                last_rating_times = user_groups['timestamp'].max()
                
                # Get the latest timestamp in the dataset
                latest_timestamp = data['timestamp'].max()
                
                # Calculate days since last rating
                if pd.api.types.is_datetime64_dtype(last_rating_times):
                    time_diff = latest_timestamp - last_rating_times
                    user_features['days_since_last_rating'] = time_diff.dt.total_seconds() / (24 * 3600)
                    
                    # Calculate rating frequency
                    first_rating_times = user_groups['timestamp'].min()
                    time_span = (last_rating_times - first_rating_times).dt.total_seconds() / (24 * 3600)
                    # Avoid division by zero
                    time_span = time_span.clip(lower=1)
                    user_features['rating_frequency'] = user_features['user_rating_count'] / time_span
                else:
                    logger.warning("Timestamp column not in datetime format, skipping time-based features")
            except Exception as e:
                logger.warning(f"Error creating time-based user features: {str(e)}")
                logger.warning("Continuing without time-based user features")
        
        logger.info(f"Created {len(user_features)} user features")
        return user_features
    
    def _create_item_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create item-related features.
        
        Args:
            data: Preprocessed DataFrame
            
        Returns:
            DataFrame with item features
        """
        logger.info("Creating item features")
        
        # Ensure data types are correct
        if 'timestamp' in data.columns:
            # Convert timestamp to datetime if it's not already
            if isinstance(data['timestamp'].iloc[0], str):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Group by item_id
        item_groups = data.groupby('item_id')
        
        # Create features
        item_features = pd.DataFrame({
            'item_id': list(item_groups.groups.keys()),
            'item_rating_count': item_groups.size(),
            'item_avg_rating': item_groups['rating'].mean(),
            'item_rating_std': item_groups['rating'].std().fillna(0),
            'popularity': item_groups.size()  # Simple popularity metric based on rating count
        })
        
        # Add time-based popularity
        if 'timestamp' in data.columns:
            try:
                if pd.api.types.is_datetime64_dtype(data['timestamp']):
                    # Recent popularity (last 90 days)
                    latest_timestamp = data['timestamp'].max()
                    cutoff_date = latest_timestamp - pd.Timedelta(days=90)
                    recent_data = data[data['timestamp'] >= cutoff_date]
                    
                    if len(recent_data) > 0:
                        recent_groups = recent_data.groupby('item_id')
                        
                        # Create a mapping of item_id to recent_count
                        recent_counts = recent_groups.size()
                        
                        # Add to item_features, defaulting to 0 for items not in recent_counts
                        item_features['recent_popularity'] = item_features['item_id'].map(
                            lambda x: recent_counts.get(x, 0)
                        )
                        
                        # Calculate popularity trend (recent vs overall)
                        item_features['popularity_trend'] = item_features['recent_popularity'] / \
                                                        item_features['item_rating_count'].clip(lower=1)
                    else:
                        item_features['recent_popularity'] = 0
                        item_features['popularity_trend'] = 0
                else:
                    logger.warning("Timestamp column not in datetime format, skipping time-based features")
            except Exception as e:
                logger.warning(f"Error creating time-based item features: {str(e)}")
                item_features['recent_popularity'] = 0
                item_features['popularity_trend'] = 0
        
        logger.info(f"Created {len(item_features)} item features")
        return item_features
    
    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features (user-item pairs).
        
        Args:
            data: Preprocessed DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features")
        
        interaction_features = data[['user_id', 'item_id']].copy()
        
        # Add basic interaction features
        if 'timestamp' in data.columns:
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_dtype(data['timestamp']):
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                
                # Time-based features
                interaction_features['hour_of_day'] = data['timestamp'].dt.hour
                interaction_features['day_of_week'] = data['timestamp'].dt.dayofweek
                interaction_features['weekend'] = interaction_features['day_of_week'].isin([5, 6]).astype(int)
                interaction_features['month'] = data['timestamp'].dt.month
            except Exception as e:
                logger.warning(f"Error creating time-based interaction features: {str(e)}")
                logger.warning("Continuing without time-based interaction features")
        
        logger.info(f"Created interaction features for {len(interaction_features)} user-item pairs")
        return interaction_features
    
    def create_features(self, 
                        train_data: pd.DataFrame, 
                        test_data: Optional[pd.DataFrame] = None, 
                        domain: str = None, 
                        dataset_name: str = None) -> Dict[str, pd.DataFrame]:
        """
        Create features from train and test data.
        
        Args:
            train_data: Training data
            test_data: Test data (optional)
            domain: Domain name (for storage)
            dataset_name: Dataset name (for storage)
            
        Returns:
            Dictionary with user, item, and interaction features
        """
        # Combine train and test data for feature creation to ensure consistency
        if test_data is not None:
            combined_data = pd.concat([train_data, test_data], ignore_index=True)
        else:
            combined_data = train_data.copy()
        
        # Create features
        user_features = self._create_user_features(combined_data)
        item_features = self._create_item_features(combined_data)
        train_interaction_features = self._create_interaction_features(train_data)
        
        # Create test interaction features if test data is provided
        test_interaction_features = None
        if test_data is not None:
            test_interaction_features = self._create_interaction_features(test_data)
        
        # Store features
        features = {
            'user_features': user_features,
            'item_features': item_features,
            'train_interaction_features': train_interaction_features
        }
        
        if test_interaction_features is not None:
            features['test_interaction_features'] = test_interaction_features
        
        # Save features if domain and dataset_name are provided
        if domain and dataset_name:
            self._save_features(features, domain, dataset_name)
        
        return features
    
    def _save_features(self, features: Dict[str, pd.DataFrame], domain: str, dataset_name: str) -> None:
        """
        Save features to disk.
        
        Args:
            features: Dictionary with feature DataFrames
            domain: Domain name
            dataset_name: Dataset name
        """
        features_dir = os.path.join(self.data_dir, "features", domain, dataset_name)
        os.makedirs(features_dir, exist_ok=True)
        
        for feature_name, feature_df in features.items():
            feature_path = os.path.join(features_dir, f"{feature_name}.parquet")
            feature_df.to_parquet(feature_path, index=False)
            logger.info(f"Saved {feature_name} to {feature_path}")
    
    def load_features(self, domain: str, dataset_name: str) -> Dict[str, pd.DataFrame]:
        """
        Load features from disk.
        
        Args:
            domain: Domain name
            dataset_name: Dataset name
            
        Returns:
            Dictionary with feature DataFrames
        """
        features_dir = os.path.join(self.data_dir, "features", domain, dataset_name)
        features = {}
        
        if not os.path.exists(features_dir):
            logger.warning(f"Features directory not found: {features_dir}")
            return features
        
        for file_name in os.listdir(features_dir):
            if file_name.endswith(".parquet"):
                feature_name = file_name.split(".")[0]
                feature_path = os.path.join(features_dir, file_name)
                features[feature_name] = pd.read_parquet(feature_path)
                logger.info(f"Loaded {feature_name} from {feature_path}")
        
        return features

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
    
    data_dir = "./data"
    pipeline = PreprocessingPipeline(data_dir)
    engineer = FeatureEngineer(data_dir)
    
    # Preprocess data
    train_data, test_data = pipeline.preprocess("entertainment", "movielens")
    
    # Create features
    features = engineer.create_features(
        train_data=train_data,
        test_data=test_data,
        domain="entertainment",
        dataset_name="movielens"
    )
    
    # Print feature info
    for name, df in features.items():
        print(f"{name}: {df.shape}")
        if len(df) > 0:
            print(df.head())
            print()
