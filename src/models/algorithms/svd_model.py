import os
import pickle
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class SVDModel(BaseModel):
    """
    SVD-based recommender system inspired by Surprise's implementation.
    This model follows the approach in the Netflix Prize competition:
    - Matrix factorization with explicit user and item biases
    - Stochastic Gradient Descent for optimization
    """
    
    def __init__(self, model_name: str = "svd"):
        """
        Initialize the SVD model.
        
        Args:
            model_name: Name of the model
        """
        super().__init__(model_name)
        
        # Default hyperparameters
        self.hyperparams = {
            "n_factors": 100,        # Number of latent factors
            "n_epochs": 20,          # Number of training epochs
            "lr_all": 0.005,         # Learning rate for all parameters
            "reg_all": 0.02,         # Regularization parameter for all parameters
            "random_state": 42       # Random seed for reproducibility
        }
        
        self.user_features = None
        self.item_features = None
        self.user_factors = None     # User latent factors (p_u)
        self.item_factors = None     # Item latent factors (q_i)
        self.user_biases = None      # User biases (b_u)
        self.item_biases = None      # Item biases (b_i)
        self.global_mean = None      # Global mean of all ratings (mu)
        self.train_data = None
        self.user_id_mapping = {}    # Maps user IDs to matrix indices
        self.item_id_mapping = {}    # Maps item IDs to matrix indices
        self.items = None            # All unique items in the training data
        self.users = None            # All unique users in the training data
        self.rng = np.random.RandomState(self.hyperparams["random_state"])
    
    def _prepare_data(self, train_data: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Prepare data for training.
        
        Args:
            train_data: Training data with user_id, item_id, and rating columns
        
        Returns:
            Tuple of (interactions array, global mean)
        """
        # Get unique users and items
        self.users = train_data['user_id'].unique()
        self.items = train_data['item_id'].unique()
        
        # Create mappings for faster lookups
        self.user_id_mapping = {user_id: idx for idx, user_id in enumerate(self.users)}
        self.item_id_mapping = {item_id: idx for idx, item_id in enumerate(self.items)}
        
        # Create array of (user_idx, item_idx, rating) tuples
        interactions = []
        for _, row in train_data.iterrows():
            user_id, item_id, rating = row['user_id'], row['item_id'], row['rating']
            user_idx = self.user_id_mapping[user_id]
            item_idx = self.item_id_mapping[item_id]
            interactions.append((user_idx, item_idx, rating))
        
        interactions_array = np.array(interactions)
        
        # Compute global mean
        global_mean = np.mean(interactions_array[:, 2])
        
        return interactions_array, global_mean
    
    def _init_parameters(self, n_users: int, n_items: int) -> None:
        """
        Initialize model parameters.
        
        Args:
            n_users: Number of users
            n_items: Number of items
        """
        n_factors = self.hyperparams["n_factors"]
        
        # Initialize latent factors with small random values
        self.user_factors = self.rng.normal(0, 0.1, (n_users, n_factors))
        self.item_factors = self.rng.normal(0, 0.1, (n_items, n_factors))
        
        # Initialize biases to zeros
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
    
    def _sgd_update(self, 
                   u: int, 
                   i: int, 
                   r: float, 
                   lr: float, 
                   reg: float) -> float:
        """
        Update parameters using stochastic gradient descent.
        
        Args:
            u: User index
            i: Item index
            r: Actual rating
            lr: Learning rate
            reg: Regularization parameter
        
        Returns:
            Error (difference between predicted and actual rating)
        """
        # Compute prediction
        pred = self.global_mean + self.user_biases[u] + self.item_biases[i] + np.dot(self.user_factors[u], self.item_factors[i])
        
        # Compute error
        err = r - pred
        
        # Update biases
        self.user_biases[u] += lr * (err - reg * self.user_biases[u])
        self.item_biases[i] += lr * (err - reg * self.item_biases[i])
        
        # Update latent factors
        u_factors_old = self.user_factors[u].copy()
        self.user_factors[u] += lr * (err * self.item_factors[i] - reg * self.user_factors[u])
        self.item_factors[i] += lr * (err * u_factors_old - reg * self.item_factors[i])
        
        return err
    
    def fit(self, 
            train_data: pd.DataFrame, 
            user_features: Optional[pd.DataFrame] = None, 
            item_features: Optional[pd.DataFrame] = None) -> None:
        """
        Train the SVD model on the provided data.
        
        Args:
            train_data: Training data with user_id, item_id, and rating columns
            user_features: Optional DataFrame with user features
            item_features: Optional DataFrame with item features
        """
        start_time = time.time()
        logger.info(f"Fitting SVD model with hyperparameters: {self.hyperparams}")
        
        # Store optional features
        self.user_features = user_features
        self.item_features = item_features
        
        # Prepare data
        interactions, self.global_mean = self._prepare_data(train_data)
        n_users = len(self.users)
        n_items = len(self.items)
        
        # Initialize parameters
        self._init_parameters(n_users, n_items)
        
        # Get hyperparameters
        n_epochs = self.hyperparams["n_epochs"]
        lr = self.hyperparams["lr_all"]
        reg = self.hyperparams["reg_all"]
        
        # Perform SGD training
        for epoch in range(n_epochs):
            # Shuffle data for SGD
            self.rng.shuffle(interactions)
            
            # Train for one epoch
            epoch_rmse = 0
            for u_idx, i_idx, rating in interactions:
                u_idx = int(u_idx)
                i_idx = int(i_idx)
                # Update parameters and accumulate error
                err = self._sgd_update(u_idx, i_idx, rating, lr, reg)
                epoch_rmse += err ** 2
            
            # Compute RMSE for this epoch
            epoch_rmse = np.sqrt(epoch_rmse / len(interactions))
            logger.info(f"Epoch {epoch+1}/{n_epochs} - RMSE: {epoch_rmse:.4f}")
        
        # Store training data for later use
        self.train_data = train_data
        
        # Update metadata
        self.metadata.update({
            "training_time": time.time() - start_time,
            "hyperparameters": self.hyperparams,
            "n_users": n_users,
            "n_items": n_items,
            "global_mean": self.global_mean,
            "final_train_rmse": epoch_rmse
        })
        
        self.fitted = True
        logger.info(f"SVD model training completed in {self.metadata['training_time']:.2f} seconds")
    
    def predict(self, 
                user_id: int, 
                item_ids: Optional[List[int]] = None, 
                n: int = 10) -> List[Tuple[int, float]]:
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
        
        # Find user index
        if user_id not in self.user_id_mapping:
            logger.warning(f"User {user_id} not found in training data")
            # Cold start approach: use average user factors and bias
            user_factors = np.mean(self.user_factors, axis=0)
            user_bias = np.mean(self.user_biases)
        else:
            user_idx = self.user_id_mapping[user_id]
            user_factors = self.user_factors[user_idx]
            user_bias = self.user_biases[user_idx]
        
        # Compute recommendations
        if item_ids is not None:
            # Filter to only valid items
            valid_items = []
            scores = []
            
            for item in item_ids:
                if item in self.item_id_mapping:
                    item_idx = self.item_id_mapping[item]
                    # Compute predicted rating
                    pred = self.global_mean + user_bias + self.item_biases[item_idx] + np.dot(user_factors, self.item_factors[item_idx])
                    valid_items.append(item)
                    scores.append(pred)
            
            if not valid_items:
                return []
            
            # Create and sort recommendations
            recommendations = list(zip(valid_items, scores))
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:n]
        
        # If no specific items provided, recommend from all items
        # Get items not already rated by the user
        if user_id in self.user_id_mapping:
            user_rated_items_df = self.train_data[self.train_data['user_id'] == user_id]
            user_rated_items = set(user_rated_items_df['item_id'].values)
            candidate_items = [item for item in self.items if item not in user_rated_items]
        else:
            candidate_items = self.items
        
        # Calculate scores for all candidate items
        candidate_scores = []
        for item in candidate_items:
            item_idx = self.item_id_mapping[item]
            # Compute predicted rating
            pred = self.global_mean + user_bias + self.item_biases[item_idx] + np.dot(user_factors, self.item_factors[item_idx])
            candidate_scores.append((item, pred))
        
        # Sort and return top n
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return candidate_scores[:n]
    
    def predict_batch(self, 
                      user_ids: List[int], 
                      n: int = 10) -> Dict[int, List[Tuple[int, float]]]:
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
        for user_id in user_ids:
            try:
                user_recs = self.predict(user_id, n=n)
                recommendations[user_id] = user_recs
            except Exception as e:
                logger.error(f"Error generating recommendations for user {user_id}: {str(e)}")
                recommendations[user_id] = []
        
        return recommendations
    
    def _calculate_ndcg_at_k(self, test_data: pd.DataFrame, k: int) -> float:
        """
        Calculate NDCG@k for the test data.
        
        Args:
            test_data: Test data with user_id, item_id, and rating columns
            k: Number of recommendations to consider
            
        Returns:
            NDCG@k score
        """
        # Group test data by user
        user_groups = test_data.groupby('user_id')
        
        ndcg_scores = []
        for user_id, group in user_groups:
            if user_id not in self.user_id_mapping:
                continue
                
            # Get recommendations for the user
            recommendations = self.predict(user_id, n=k)
            if not recommendations:
                continue
                
            # Create relevance array for recommended items
            recommended_items = [item_id for item_id, _ in recommendations]
            
            # Get relevant items from test data (items the user actually interacted with)
            relevant_items = set(group['item_id'].values)
            
            # Calculate relevance scores
            relevance = np.zeros(len(recommended_items))
            for i, item_id in enumerate(recommended_items):
                if item_id in relevant_items:
                    relevance[i] = 1.0
            
            # If there are no relevant items in the recommendations, skip this user
            if np.sum(relevance) == 0:
                continue
                
            # Calculate NDCG@k manually
            # DCG = sum(rel_i / log2(i+1))
            dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))
            
            # Ideal DCG = DCG with perfect ranking (all relevant items at the top)
            ideal_relevance = np.zeros(len(recommended_items))
            ideal_relevance[:min(len(relevant_items), k)] = 1.0
            idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
            
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def _calculate_map_at_k(self, test_data: pd.DataFrame, k: int) -> float:
        """
        Calculate MAP@k for the test data.
        
        Args:
            test_data: Test data with user_id, item_id, and rating columns
            k: Number of recommendations to consider
            
        Returns:
            MAP@k score
        """
        # Group test data by user
        user_groups = test_data.groupby('user_id')
        
        ap_scores = []
        for user_id, group in user_groups:
            if user_id not in self.user_id_mapping:
                continue
                
            # Get recommendations for the user
            recommendations = self.predict(user_id, n=k)
            if not recommendations:
                continue
                
            # Create list of recommended items
            recommended_items = [item_id for item_id, _ in recommendations]
            
            # Get relevant items from test data (items the user actually interacted with)
            relevant_items = set(group['item_id'].values)
            
            # Calculate precision at each position where a relevant item is found
            precision_values = []
            num_relevant = 0
            
            for i, item_id in enumerate(recommended_items):
                if item_id in relevant_items:
                    num_relevant += 1
                    precision_values.append(num_relevant / (i + 1))
            
            # Average precision = mean of precision values
            if precision_values:
                ap_scores.append(np.mean(precision_values))
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
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
        
        # Update random state if it was changed
        if "random_state" in kwargs:
            self.rng = np.random.RandomState(self.hyperparams["random_state"])
        
        # If model is already fitted, it needs to be retrained
        if self.fitted:
            logger.warning("Hyperparameters changed, model needs to be retrained")
            self.fitted = False
    
    def save(self, path: str) -> None:
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        if not self.fitted:
            raise ValueError("Cannot save model that has not been fitted")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare data to save
        save_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_biases': self.user_biases,
            'item_biases': self.item_biases,
            'global_mean': self.global_mean,
            'train_data': self.train_data,
            'hyperparams': self.hyperparams,
            'metadata': self.metadata,
            'user_features': self.user_features,
            'item_features': self.item_features,
            'user_id_mapping': self.user_id_mapping,
            'item_id_mapping': self.item_id_mapping,
            'users': self.users,
            'items': self.items
        }
        
        # Save to file
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SVDModel':
        """
        Load a previously saved model.
        
        Args:
            path: Path to load the model from
        
        Returns:
            Loaded SVDModel instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load from file
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create model instance
        model = cls()
        model.user_factors = save_data['user_factors']
        model.item_factors = save_data['item_factors']
        model.user_biases = save_data['user_biases']
        model.item_biases = save_data['item_biases']
        model.global_mean = save_data['global_mean']
        model.train_data = save_data['train_data']
        model.hyperparams = save_data['hyperparams']
        model.metadata = save_data['metadata']
        model.user_features = save_data['user_features']
        model.item_features = save_data['item_features']
        model.user_id_mapping = save_data.get('user_id_mapping', {})
        model.item_id_mapping = save_data.get('item_id_mapping', {})
        model.users = save_data.get('users')
        model.items = save_data.get('items')
        model.rng = np.random.RandomState(model.hyperparams["random_state"])
        model.fitted = True
        
        return model