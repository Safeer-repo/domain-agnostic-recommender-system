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
    
    def evaluate(self, 
                 test_data: pd.DataFrame, 
                 k: int = 10, 
                 metrics: Optional[List[str]] = None) -> Dict[str, float]:
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
        
        # Prepare metrics
        if metrics is None:
            metrics = ['rmse', 'mae', 'precision', 'recall']
        
        results = {}
        
        # Compute RMSE and MAE if required
        if any(m in metrics for m in ['rmse', 'mae']):
            predictions = []
            actual = []
            
            for _, row in test_data.iterrows():
                user_id, item_id, true_rating = row['user_id'], row['item_id'], row['rating']
                
                try:
                    # Use predict method to get predicted rating
                    pred_items = self.predict(user_id, [item_id], n=1)
                    if pred_items:
                        pred_rating = pred_items[0][1]
                        predictions.append(pred_rating)
                        actual.append(true_rating)
                except Exception as e:
                    # Skip if prediction fails
                    logger.debug(f"Prediction failed for user {user_id}, item {item_id}: {str(e)}")
                    continue
            
            # Compute regression metrics
            if predictions and 'rmse' in metrics:
                results['rmse'] = np.sqrt(mean_squared_error(actual, predictions))
            
            if predictions and 'mae' in metrics:
                results['mae'] = mean_absolute_error(actual, predictions)
        
        # Compute precision and recall at k if required
        if any(m in metrics for m in ['precision', 'recall']):
            # Group test data by user
            user_test_items = test_data.groupby('user_id')['item_id'].apply(list).to_dict()
            
            # Get list of users in test data
            test_users = list(user_test_items.keys())
            
            # Compute recommendations for each user
            total_precision = 0.0
            total_recall = 0.0
            num_users = 0
            
            for user_id in test_users:
                if user_id not in self.user_id_mapping:
                    continue  # Skip users not in training data
                
                # Get ground truth items for this user
                ground_truth = set(user_test_items[user_id])
                
                # Get recommendations
                try:
                    recs = self.predict(user_id, n=k)
                    recommended_items = set([item_id for item_id, _ in recs])
                    
                    # Compute precision and recall
                    if recommended_items:
                        relevant = recommended_items.intersection(ground_truth)
                        precision = len(relevant) / len(recommended_items)
                        recall = len(relevant) / len(ground_truth) if ground_truth else 0
                        
                        total_precision += precision
                        total_recall += recall
                        num_users += 1
                except Exception as e:
                    logger.debug(f"Error computing recommendations for user {user_id}: {str(e)}")
                    continue
            
            # Compute average precision and recall
            if num_users > 0:
                if 'precision' in metrics:
                    results['precision'] = total_precision / num_users
                
                if 'recall' in metrics:
                    results['recall'] = total_recall / num_users
        
        # Update metadata
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