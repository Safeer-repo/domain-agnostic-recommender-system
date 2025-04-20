import os
import pandas as pd
import numpy as np
import logging
import time
import pickle
from typing import Dict, List, Any, Optional, Tuple
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class ALSModel(BaseModel):
    """
    Alternating Least Squares model for collaborative filtering.
    Uses the implicit library for efficient ALS implementation.
    """
    
    def __init__(self, model_name: str = "als"):
        """
        Initialize the ALS model.
        
        Args:
            model_name: Name of the model
        """
        super().__init__(model_name)
        
        # Default hyperparameters
        self.hyperparams = {
            "factors": 100,       # Latent factors (rank)
            "regularization": 0.01,  # Regularization parameter
            "iterations": 15,     # Max iterations
            "alpha": 1.0,         # Confidence scaling for implicit feedback
            "calculate_training_loss": True,
            "num_threads": 4,     # Number of parallel threads
            "use_native": True,   # Use native C++ implementation
            "use_cg": True,       # Use Conjugate Gradient for faster solving
            "use_gpu": False      # Use GPU for computation (if available)
        }
        
        self.user_features = None
        self.item_features = None
        self.user_items_csr = None  # Store the user-item matrix for recommendations
    
    def fit(self, train_data: pd.DataFrame, user_features: Optional[pd.DataFrame] = None, 
            item_features: Optional[pd.DataFrame] = None) -> None:
        """
        Train the ALS model on the provided data.
        
        Args:
            train_data: Training data with user_id, item_id, and rating columns
            user_features: Optional DataFrame with user features
            item_features: Optional DataFrame with item features
        """
        start_time = time.time()
        logger.info(f"Fitting ALS model with hyperparameters: {self.hyperparams}")
        
        # Store features for later use
        self.user_features = user_features
        self.item_features = item_features
        
        # Create user and item indices
        self._create_indices(train_data)
        
        # Convert ratings to a sparse matrix
        user_indices = [self.user_index[user_id] for user_id in train_data['user_id']]
        item_indices = [self.item_index[item_id] for item_id in train_data['item_id']]
        
        # Use rating as the confidence value
        ratings = train_data['rating'].values
        
        # Create sparse matrix (users x items)
        sparse_matrix = sparse.coo_matrix((ratings, (user_indices, item_indices)), 
                                         shape=(len(self.user_index), len(self.item_index)))
        
        # Convert to CSR format for efficient matrix operations
        self.user_items_csr = sparse_matrix.tocsr()
        
        # Initialize and fit the ALS model
        self.model = AlternatingLeastSquares(**self.hyperparams)
        self.model.fit(self.user_items_csr)
        
        # Update metadata
        self.metadata.update({
            "num_users": len(self.user_index),
            "num_items": len(self.item_index),
            "hyperparameters": self.hyperparams,
            "training_time": time.time() - start_time
        })
        
        self.fitted = True
        logger.info(f"ALS model fitting completed in {self.metadata['training_time']:.2f} seconds")
    
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
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")
        
        # Check if user exists in the training data
        if user_id not in self.user_index:
            logger.warning(f"User {user_id} not in training data, returning empty recommendations")
            return []
        
        # Get internal user index
        user_idx = self.user_index[user_id]
        
        if item_ids is not None:
            # Filter to only valid items that exist in the training data
            valid_items = [item_id for item_id in item_ids if item_id in self.item_index]
            if not valid_items:
                return []
            
            # Get internal item indices
            item_indices = [self.item_index[item_id] for item_id in valid_items]
            
            # Get scores for specific items
            scores = self.model.user_factors[user_idx].dot(self.model.item_factors[item_indices].T)
            
            # Create (item_id, score) tuples and sort by score
            recommendations = [(valid_items[i], float(scores[i])) for i in range(len(valid_items))]
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            # Return top n
            return recommendations[:n]
        else:
            # Get recommendations directly from the model
            # Note: recommend() needs the user-item matrix
            ids, scores = self.model.recommend(
                user_idx, 
                self.user_items_csr[user_idx], 
                N=n,
                filter_already_liked_items=True
            )
            
            # Map internal indices back to original item IDs
            recommendations = [(self.reverse_item_index[item_idx], float(score)) 
                              for item_idx, score in zip(ids, scores)]
            
            return recommendations
    
    def predict_batch(self, user_ids: List[int], n: int = 10) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n: Number of recommendations per user
            
        Returns:
            Dictionary mapping user IDs to lists of (item_id, score) tuples
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")
        
        recommendations = {}
        
        # Filter to users that exist in the training data
        valid_users = [user_id for user_id in user_ids if user_id in self.user_index]
        
        # Get recommendations for each valid user
        for user_id in valid_users:
            recommendations[user_id] = self.predict(user_id, n=n)
        
        # For invalid users, return empty recommendations
        for user_id in set(user_ids) - set(valid_users):
            recommendations[user_id] = []
        
        return recommendations
    
    def evaluate(self, test_data: pd.DataFrame, k: int = 10, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate the model's performance.
        
        Args:
            test_data: Test data with user_id, item_id, and rating columns
            k: Number of recommendations to consider
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")
        
        if metrics is None:
            metrics = ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'map_at_k', 
                    'coverage', 'novelty', 'diversity', 'rmse', 'mae']
        
        results = {}
        
        # Compute standard accuracy metrics (this code already exists in your models)
        # ...
        
        # Compute beyond-accuracy metrics (coverage, novelty, diversity)
        if any(m in metrics for m in ['coverage', 'novelty', 'diversity']):
            # Get a sample of users from the test data
            test_users = test_data['user_id'].unique()
            sample_size = min(100, len(test_users))  # Limit to 100 users for efficiency
            sampled_users = list(np.random.choice(test_users, size=sample_size, replace=False))
            
            # Generate recommendations for these users
            recommendations = self.predict_batch(sampled_users, n=k)
            
            # Import beyond-accuracy metrics function
            from src.utils.evaluation_metrics import calculate_all_beyond_accuracy_metrics
            
            # Get item features for diversity calculation if available
            item_features = None
            if hasattr(self, 'item_features') and self.item_features is not None:
                # Convert from matrix to dictionary for diversity calculation
                # ... model-specific code to extract item features ...
            
            # Calculate beyond-accuracy metrics
                beyond_accuracy_metrics = calculate_all_beyond_accuracy_metrics(
                    recommendations, self.train_data, item_features
                )
            
            # Update results with beyond-accuracy metrics
            results.update(beyond_accuracy_metrics)
        
        # Update metadata with performance results
        self.metadata['performance'] = results
        
        return results
    
    def set_hyperparameters(self, **kwargs: Any) -> None:
        """
        Set hyperparameters for the model.
        
        Args:
            **kwargs: Hyperparameters to set
        """
        self.hyperparams.update(kwargs)
        
        # If model is already fitted, it needs to be retrained
        if self.fitted:
            logger.warning("Hyperparameters changed, model needs to be retrained")
            self.fitted = False
    
    def get_similar_items(self, item_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        Get items similar to a given item.
        
        Args:
            item_id: ID of the item
            n: Number of similar items to return
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")
        
        # Check if item exists in the training data
        if item_id not in self.item_index:
            logger.warning(f"Item {item_id} not in training data, returning empty list")
            return []
        
        # Get internal item index
        item_idx = self.item_index[item_id]
        
        # Get similar items from the model
        similar_indices, similarity_scores = self.model.similar_items(item_idx, N=n+1)
        
        # Skip the first item (it's the query item itself)
        similar_indices = similar_indices[1:]
        similarity_scores = similarity_scores[1:]
        
        # Map internal indices back to original item IDs
        similar_items = [(self.reverse_item_index[idx], float(score)) 
                         for idx, score in zip(similar_indices, similarity_scores)]
        
        return similar_items
    
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
        
        # Prepare data to save - we need special handling for sparse matrices
        data = {
            "model": self.model,
            "user_index": self.user_index,
            "item_index": self.item_index,
            "reverse_user_index": self.reverse_user_index,
            "reverse_item_index": self.reverse_item_index,
            "metadata": self.metadata,
            "fitted": self.fitted,
            "hyperparams": self.hyperparams,
            # Store sparse matrix in a serializable format
            "user_items_csr_data": self.user_items_csr.data if self.user_items_csr is not None else None,
            "user_items_csr_indices": self.user_items_csr.indices if self.user_items_csr is not None else None,
            "user_items_csr_indptr": self.user_items_csr.indptr if self.user_items_csr is not None else None,
            "user_items_csr_shape": self.user_items_csr.shape if self.user_items_csr is not None else None
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
        self.hyperparams = data["hyperparams"]
        
        # Rebuild sparse matrix if available
        if all(k in data for k in ["user_items_csr_data", "user_items_csr_indices", "user_items_csr_indptr", "user_items_csr_shape"]):
            if data["user_items_csr_data"] is not None:
                self.user_items_csr = sparse.csr_matrix(
                    (data["user_items_csr_data"], data["user_items_csr_indices"], data["user_items_csr_indptr"]),
                    shape=data["user_items_csr_shape"]
                )
        
        logger.info(f"Model loaded from {path}")
