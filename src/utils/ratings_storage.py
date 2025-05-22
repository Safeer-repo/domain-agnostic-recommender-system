import os
import pandas as pd
import logging
import time
from datetime import datetime
from typing import Dict, Optional, Union, List

logger = logging.getLogger(__name__)

class RatingsStorage:
    """
    Handler for storing and retrieving new user ratings.
    This class manages the storage of user feedback that will be used
    to update the recommender system.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the ratings storage handler.
        
        Args:
            data_dir: Root directory for data storage
        """
        self.data_dir = data_dir
        self.new_ratings_dir = os.path.join(data_dir, "new_ratings")
        self._ensure_directory_structure()
        
    def _ensure_directory_structure(self) -> None:
        """Create necessary directory structure if it doesn't exist."""
        if not os.path.exists(self.new_ratings_dir):
            os.makedirs(self.new_ratings_dir, exist_ok=True)
            logger.info(f"Created new ratings directory at {self.new_ratings_dir}")
    
    def _get_domain_dataset_path(self, domain: str, dataset: str) -> str:
        """
        Get the path for a specific domain and dataset.
        
        Args:
            domain: Domain name (e.g., entertainment, ecommerce)
            dataset: Dataset name (e.g., movielens, amazon_electronics)
            
        Returns:
            Path to the domain/dataset directory
        """
        domain_dataset_dir = os.path.join(self.new_ratings_dir, domain, dataset)
        if not os.path.exists(domain_dataset_dir):
            os.makedirs(domain_dataset_dir, exist_ok=True)
            logger.info(f"Created directory for {domain}/{dataset} at {domain_dataset_dir}")
        return domain_dataset_dir
    
    def _get_current_ratings_file(self, domain: str, dataset: str) -> str:
        """
        Get the path to the current ratings file for a domain/dataset.
        Uses a date-based naming scheme to allow for file rotation.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            
        Returns:
            Path to the current ratings file
        """
        domain_dataset_dir = self._get_domain_dataset_path(domain, dataset)
        current_date = datetime.now().strftime("%Y%m%d")
        ratings_file = os.path.join(domain_dataset_dir, f"ratings_{current_date}.csv")
        return ratings_file
    
    def _read_ratings_file(self, file_path: str) -> pd.DataFrame:
        """
        Read a ratings file with proper data types.
        
        Args:
            file_path: Path to the ratings file
            
        Returns:
            DataFrame with correct data types
        """
        try:
            # Read the CSV file first
            ratings = pd.read_csv(file_path)
            
            # Convert user_id and item_id to strings (works for both numeric and alphanumeric IDs)
            if 'user_id' in ratings.columns:
                ratings['user_id'] = ratings['user_id'].astype(str)
            
            if 'item_id' in ratings.columns:
                ratings['item_id'] = ratings['item_id'].astype(str)
            
            # Convert rating to float
            if 'rating' in ratings.columns:
                ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce').fillna(0.0).astype(float)
            
            # Convert timestamp column to integer explicitly
            if 'timestamp' in ratings.columns:
                ratings['timestamp'] = pd.to_numeric(ratings['timestamp'], errors='coerce')
                # Fill NaN values with current timestamp if any conversion failed
                if ratings['timestamp'].isna().any():
                    logger.warning(f"Some timestamp values in {file_path} couldn't be converted to integers. Using current time instead.")
                    ratings['timestamp'] = ratings['timestamp'].fillna(int(time.time()))
            
            return ratings
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return pd.DataFrame(columns=["user_id", "item_id", "rating", "timestamp", "source"])
    
    def store_rating(self, domain: str, dataset: str, user_id: Union[str, int], 
                     item_id: int, rating: float, 
                     timestamp: Optional[int] = None,
                     source: str = "explicit") -> bool:
        """
        Store a single rating in the appropriate file.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            user_id: ID of the user (can be string UUID or integer)
            item_id: ID of the item
            rating: Rating value
            timestamp: Optional timestamp (uses current time if None)
            source: Source of the rating (explicit, implicit, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        if timestamp is None:
            timestamp = int(time.time())
        
        # Create a DataFrame for the new rating with explicit types
        rating_df = pd.DataFrame({
            "user_id": [str(user_id)],  # Always store as string
            "item_id": [str(item_id)],  # Always store as string (both numeric and alphanumeric)
            "rating": [float(rating)],
            "timestamp": [int(timestamp)],
            "source": [str(source)]
        })
        
        try:
            ratings_file = self._get_current_ratings_file(domain, dataset)
            
            # If file exists, append to it; otherwise create new file
            if os.path.exists(ratings_file):
                # Check if we should rotate the file
                file_size = os.path.getsize(ratings_file)
                # Rotate if file size exceeds 10MB
                if file_size > 10 * 1024 * 1024:  # 10MB
                    # Create new file with timestamp in name
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    domain_dataset_dir = self._get_domain_dataset_path(domain, dataset)
                    ratings_file = os.path.join(domain_dataset_dir, f"ratings_{timestamp_str}.csv")
                
                # Append to existing file
                rating_df.to_csv(ratings_file, mode='a', header=False, index=False)
                logger.debug(f"Appended rating to {ratings_file}")
            else:
                # Create new file with header
                rating_df.to_csv(ratings_file, index=False)
                logger.info(f"Created new ratings file {ratings_file}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to store rating: {str(e)}")
            return False
    
    def store_ratings_batch(self, domain: str, dataset: str, 
                           ratings: List[Dict[str, Union[int, float, str]]]) -> int:
        """
        Store multiple ratings at once.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            ratings: List of rating dictionaries, each containing user_id, item_id, rating,
                    and optionally timestamp and source
            
        Returns:
            Number of successfully stored ratings
        """
        # Validate and prepare ratings
        current_time = int(time.time())
        valid_ratings = []
        
        for rating_data in ratings:
            if 'user_id' not in rating_data or 'item_id' not in rating_data or 'rating' not in rating_data:
                logger.warning(f"Skipping invalid rating: {rating_data}")
                continue
            
            # Set default values if missing
            if 'timestamp' not in rating_data or rating_data['timestamp'] is None:
                rating_data['timestamp'] = current_time
            if 'source' not in rating_data:
                rating_data['source'] = 'explicit'
                
            # Ensure proper types
            processed_rating = {
                'user_id': str(rating_data['user_id']),  # Always store as string
                'item_id': str(rating_data['item_id']),  # Always store as string (both numeric and alphanumeric)
                'rating': float(rating_data['rating']),
                'timestamp': int(rating_data['timestamp']),
                'source': str(rating_data['source'])
            }
            valid_ratings.append(processed_rating)
        
        if not valid_ratings:
            logger.warning("No valid ratings to store")
            return 0
        
        try:
            # Convert to DataFrame
            ratings_df = pd.DataFrame(valid_ratings)
            
            # Get current file path
            ratings_file = self._get_current_ratings_file(domain, dataset)
            
            # If file exists, append to it; otherwise create new file
            if os.path.exists(ratings_file):
                # Check if we should rotate the file based on size
                file_size = os.path.getsize(ratings_file)
                if file_size > 10 * 1024 * 1024:  # 10MB
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    domain_dataset_dir = self._get_domain_dataset_path(domain, dataset)
                    ratings_file = os.path.join(domain_dataset_dir, f"ratings_{timestamp_str}.csv")
                
                # Append to existing file
                ratings_df.to_csv(ratings_file, mode='a', header=False, index=False)
            else:
                # Create new file with header
                ratings_df.to_csv(ratings_file, index=False)
            
            logger.info(f"Stored {len(valid_ratings)} ratings in {ratings_file}")
            return len(valid_ratings)
        except Exception as e:
            logger.error(f"Failed to store ratings batch: {str(e)}")
            return 0
    
    def get_new_ratings(self, domain: str, dataset: str, 
                       since_timestamp: Optional[int] = None) -> pd.DataFrame:
        """
        Get all new ratings for a domain/dataset since a specific timestamp.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            since_timestamp: Optional timestamp to filter ratings (None for all)
            
        Returns:
            DataFrame containing the ratings
        """
        domain_dataset_dir = self._get_domain_dataset_path(domain, dataset)
        all_ratings = []
        
        # Get all ratings files in the directory
        for filename in os.listdir(domain_dataset_dir):
            if filename.startswith("ratings_") and filename.endswith(".csv"):
                file_path = os.path.join(domain_dataset_dir, filename)
                try:
                    # Read the file with proper data types
                    ratings = self._read_ratings_file(file_path)
                    
                    # Skip empty dataframes
                    if ratings.empty:
                        continue
                        
                    # Filter by timestamp if specified
                    if since_timestamp is not None:
                        logger.debug(f"Filtering ratings since timestamp: {since_timestamp}")
                        logger.debug(f"First few timestamps in file: {ratings['timestamp'].head().tolist()}")
                        logger.debug(f"Timestamp type: {ratings['timestamp'].dtype}")
                        
                        # Ensure timestamp is numeric
                        ratings = ratings[ratings['timestamp'] >= since_timestamp]
                        
                    all_ratings.append(ratings)
                except Exception as e:
                    logger.error(f"Error processing ratings from {file_path}: {str(e)}")
        
        # Combine all ratings
        if not all_ratings:
            return pd.DataFrame(columns=["user_id", "item_id", "rating", "timestamp", "source"])
            
        combined_ratings = pd.concat(all_ratings, ignore_index=True)
        return combined_ratings
    
    def get_new_ratings_count(self, domain: str, dataset: str) -> int:
        """
        Get the total count of new ratings for a domain/dataset.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            
        Returns:
            Count of new ratings
        """
        try:
            ratings = self.get_new_ratings(domain, dataset)
            return len(ratings)
        except Exception as e:
            logger.error(f"Error counting new ratings: {str(e)}")
            return 0
    
    def clear_processed_ratings(self, domain: str, dataset: str, 
                               before_timestamp: int) -> int:
        """
        Remove ratings that have been processed (merged into the main dataset).
        
        Args:
            domain: Domain name
            dataset: Dataset name
            before_timestamp: Timestamp indicating which ratings have been processed
            
        Returns:
            Number of removed ratings
        """
        domain_dataset_dir = self._get_domain_dataset_path(domain, dataset)
        total_removed = 0
        
        logger.debug(f"Clearing processed ratings before timestamp: {before_timestamp}")
        
        # Get all ratings files in the directory
        for filename in os.listdir(domain_dataset_dir):
            if filename.startswith("ratings_") and filename.endswith(".csv"):
                file_path = os.path.join(domain_dataset_dir, filename)
                try:
                    # Read the file with proper data types
                    ratings = self._read_ratings_file(file_path)
                    
                    # Skip empty dataframes
                    if ratings.empty:
                        continue
                    
                    # Log for debugging
                    logger.debug(f"File: {file_path}, Shape: {ratings.shape}")
                    logger.debug(f"First few timestamp values: {ratings['timestamp'].head().tolist()}")
                    
                    # Count ratings to remove
                    to_remove_mask = ratings['timestamp'] < before_timestamp
                    to_remove = to_remove_mask.sum()
                    
                    if to_remove > 0:
                        logger.debug(f"Found {to_remove} ratings to remove in {file_path}")
                    
                    total_removed += to_remove
                    
                    # Keep only ratings that haven't been processed
                    ratings_to_keep = ratings[~to_remove_mask]
                    
                    if len(ratings_to_keep) > 0:
                        # Write back the unprocessed ratings
                        ratings_to_keep.to_csv(file_path, index=False)
                        logger.debug(f"Kept {len(ratings_to_keep)} ratings in {file_path}")
                    else:
                        # Remove the file if all ratings have been processed
                        os.remove(file_path)
                        logger.info(f"Removed empty ratings file {file_path}")
                        
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
        
        logger.info(f"Removed {total_removed} processed ratings for {domain}/{dataset}")
        return total_removed
    
    def has_user_ratings(self, domain: str, dataset: str, user_id: Union[str, int]) -> bool:
        """
        Check if a user has any ratings
        
        Args:
            domain: Domain name
            dataset: Dataset name
            user_id: User ID (string UUID or integer)
            
        Returns:
            Boolean indicating if the user has ratings
        """
        # Convert user_id to string for consistent comparison
        user_id_str = str(user_id)
        
        # First check in the main dataset
        dataset_path = os.path.join(self.data_dir, "datasets", domain, dataset, "ratings.csv")
        if os.path.exists(dataset_path):
            try:
                import pandas as pd
                ratings_df = pd.read_csv(dataset_path)
                if 'user_id' in ratings_df.columns and user_id_str in ratings_df['user_id'].astype(str).values:
                    return True
            except Exception as e:
                logger.error(f"Error checking user ratings in dataset: {str(e)}")
        
        # Then check in new ratings
        new_ratings_path = os.path.join(self.data_dir, "new_ratings", domain, dataset)
        if os.path.exists(new_ratings_path):
            rating_files = [f for f in os.listdir(new_ratings_path) if f.endswith('.csv')]
            for file in rating_files:
                try:
                    import pandas as pd
                    ratings_df = pd.read_csv(os.path.join(new_ratings_path, file))
                    if 'user_id' in ratings_df.columns and user_id_str in ratings_df['user_id'].astype(str).values:
                        return True
                except Exception as e:
                    logger.error(f"Error checking user ratings in {file}: {str(e)}")
        
        return False