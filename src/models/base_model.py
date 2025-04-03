from abc import ABC, abstractmethod
import os
import pandas as pd
import numpy as np
import logging
import pickle
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """
    Abstract base class for all recommendation models.
    Defines the interface that all models must implement.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.fitted = False
        self.model = None
        self.user_index = {}  # Maps user IDs to internal indices
        self.item_index = {}  # Maps item IDs to internal indices
        self.reverse_user_index = {}  # Maps internal indices to user IDs
        self.reverse_item_index = {}  # Maps internal indices to item IDs
        self.metadata = {
            "model_name": model_name,
            "version": "0.1.0",
            "performance": {}
        }
    
    @abstractmethod
    def fit(self, train_data: pd.DataFrame, user_features: Optional[pd.DataFrame] = None, 
            item_features: Optional[pd.DataFrame] = None) -> None:
        """
        Train the model on the provided data.
        
        Args:
            train_data: Training data with user_id, item_id, and rating columns
            user_features: Optional DataFrame with user features
            item_features: Optional DataFrame with item features
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: int, item_ids: Optional[List[int]] = None, n: int = 10) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: ID of the user
            item_ids: Optional list of item IDs to rank
            n: Number of recommendations to return
            
        Returns:
            List of (item_id, score) tuples
        """
        pass
    
    @abstractmethod
    def predict_batch(self, user_ids: List[int], n: int = 10) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n: Number of recommendations per user
            
        Returns:
            Dictionary mapping user IDs to lists of (item_id, score) tuples
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: pd.DataFrame, k: int = 10, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data with user_id, item_id, and rating columns
            k: Number of recommendations to consider for ranking metrics
            metrics: List of metrics to compute
            
        Returns:
            Dictionary mapping metric names to values
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model to
        """
        if not self.fitted:
            raise ValueError("Cannot save model that has not been fitted")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare data to save
        data = {
            "model": self.model,
            "user_index": self.user_index,
            "item_index": self.item_index,
            "reverse_user_index": self.reverse_user_index,
            "reverse_item_index": self.reverse_item_index,
            "metadata": self.metadata,
            "fitted": self.fitted
        }
        
        # Save to file
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load from file
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Restore model state
        self.model = data["model"]
        self.user_index = data["user_index"]
        self.item_index = data["item_index"]
        self.reverse_user_index = data["reverse_user_index"]
        self.reverse_item_index = data["reverse_item_index"]
        self.metadata = data["metadata"]
        self.fitted = data["fitted"]
        
        logger.info(f"Model loaded from {path}")
    
    def _create_indices(self, train_data: pd.DataFrame) -> None:
        """
        Create user and item indices for mapping between IDs and internal indices.
        
        Args:
            train_data: Training data with user_id and item_id columns
        """
        # Create user index
        unique_users = train_data['user_id'].unique()
        self.user_index = {user_id: i for i, user_id in enumerate(unique_users)}
        self.reverse_user_index = {i: user_id for user_id, i in self.user_index.items()}
        
        # Create item index
        unique_items = train_data['item_id'].unique()
        self.item_index = {item_id: i for i, item_id in enumerate(unique_items)}
        self.reverse_item_index = {i: item_id for item_id, i in self.item_index.items()}
        
        logger.info(f"Created indices for {len(self.user_index)} users and {len(self.item_index)} items")
    
    def _calculate_ranking_metrics(self, test_data: pd.DataFrame, k: int = 10) -> Dict[str, float]:
        """
        Calculate ranking metrics like Precision@k, Recall@k, NDCG@k, MAP@k.
        
        This is a generic implementation that can be overridden by subclasses.
        
        Args:
            test_data: Test data with user_id, item_id, and rating columns
            k: Number of recommendations to consider
            
        Returns:
            Dictionary with metric values
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")
        
        results = {}
        
        # Get unique users in test data
        test_users = test_data['user_id'].unique()
        
        # Calculate metrics for each user
        precision_at_k = []
        recall_at_k = []
        ndcg_at_k = []
        map_at_k = []
        
        for user_id in test_users:
            # Get ground truth items for this user
            true_items = set(test_data[test_data['user_id'] == user_id]['item_id'].tolist())
            
            if not true_items:
                continue
            
            # Get recommendations for this user
            try:
                recs = self.predict(user_id, n=k)
                rec_items = [item_id for item_id, _ in recs]
                
                # Precision@k
                hits = len(set(rec_items) & true_items)
                precision = hits / min(k, len(rec_items)) if rec_items else 0
                precision_at_k.append(precision)
                
                # Recall@k
                recall = hits / len(true_items) if true_items else 0
                recall_at_k.append(recall)
                
                # NDCG@k
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(true_items))))
                dcg = 0
                for i, item_id in enumerate(rec_items):
                    if item_id in true_items:
                        dcg += 1.0 / np.log2(i + 2)
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_at_k.append(ndcg)
                
                # MAP@k
                ap = 0
                hits = 0
                for i, item_id in enumerate(rec_items):
                    if item_id in true_items:
                        hits += 1
                        ap += hits / (i + 1)
                ap = ap / min(k, len(true_items)) if true_items else 0
                map_at_k.append(ap)
                
            except Exception as e:
                logger.warning(f"Error calculating metrics for user {user_id}: {str(e)}")
        
        # Calculate average metrics
        results['precision_at_k'] = np.mean(precision_at_k) if precision_at_k else 0
        results['recall_at_k'] = np.mean(recall_at_k) if recall_at_k else 0
        results['ndcg_at_k'] = np.mean(ndcg_at_k) if ndcg_at_k else 0
        results['map_at_k'] = np.mean(map_at_k) if map_at_k else 0
        
        return results
    
    def _calculate_rating_metrics(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate rating metrics like RMSE, MAE.
        
        Args:
            test_data: Test data with user_id, item_id, and rating columns
            
        Returns:
            Dictionary with metric values
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")
        
        results = {}
        
        # Predict ratings for test data
        predictions = []
        actuals = []
        
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']
            
            try:
                # Get prediction for this user-item pair
                pred = self.predict(user_id, [item_id], n=1)
                if pred:
                    _, pred_rating = pred[0]
                    predictions.append(pred_rating)
                    actuals.append(actual_rating)
            except Exception as e:
                logger.warning(f"Error predicting for user {user_id}, item {item_id}: {str(e)}")
        
        if predictions:
            # Calculate RMSE
            rmse = np.sqrt(np.mean(np.square(np.array(predictions) - np.array(actuals))))
            results['rmse'] = rmse
            
            # Calculate MAE
            mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
            results['mae'] = mae
        
        return results
