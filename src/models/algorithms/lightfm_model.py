import os
import pandas as pd
import numpy as np
import logging
import time
import pickle
from typing import Dict, List, Any, Optional, Tuple
import scipy.sparse as sparse
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class LightFMModel(BaseModel):
    """
    LightFM model for hybrid collaborative filtering.
    Handles both user-item interactions and features.
    """
    
    def __init__(self, model_name: str = "lightfm"):
        super().__init__(model_name)
        
        # Default hyperparameters
        self.hyperparams = {
            "loss": "warp",
            "learning_rate": 0.05,
            "epochs": 20,
            "item_alpha": 0.0001,
            "user_alpha": 0.0001,
            "no_components": 64,
            "max_sampled": 10,
            "random_state": 42
        }
        
        self.dataset = Dataset()
        self.model = None
        self.user_features = None
        self.item_features = None
        self.interactions_matrix = None
        self.user_index = {}
        self.item_index = {}
        self.reverse_user_index = {}
        self.reverse_item_index = {}
        self.fitted = False
    
    def _create_indices(self, train_data: pd.DataFrame) -> None:
        """Create user and item indices for mapping IDs."""
        unique_users = train_data['user_id'].unique()
        unique_items = train_data['item_id'].unique()
        
        self.user_index = {user_id: i for i, user_id in enumerate(unique_users)}
        self.item_index = {item_id: i for i, item_id in enumerate(unique_items)}
        self.reverse_user_index = {i: user_id for user_id, i in self.user_index.items()}
        self.reverse_item_index = {i: item_id for item_id, i in self.item_index.items()}
        
        logger.info(f"Created indices for {len(self.user_index)} users and {len(self.item_index)} items")

    def fit(self, train_data: pd.DataFrame, user_features: Optional[pd.DataFrame] = None,
            item_features: Optional[pd.DataFrame] = None) -> None:
        """
        Train the LightFM model on the provided data.
        
        Args:
            train_data: Training data with user_id, item_id, and rating columns
            user_features: Optional DataFrame with user features
            item_features: Optional DataFrame with item features
        """
        start_time = time.time()
        logger.info(f"Fitting LightFM model with hyperparameters: {self.hyperparams}")
        
        # Create indices
        self._create_indices(train_data)
        
        # Initialize dataset
        self.dataset = Dataset()
        
        # Fit dataset with users and items
        self.dataset.fit(
            users=list(self.user_index.keys()),
            items=list(self.item_index.keys())
        )
        
        # Fit user and item features if provided
        user_features_matrix = None
        item_features_matrix = None
        
        if user_features is not None:
            logger.info("Processing user features")
            user_features_matrix = self._process_user_features(user_features)
            
        if item_features is not None:
            logger.info("Processing item features")
            item_features_matrix = self._process_item_features(item_features)
        
        # Build interactions matrix
        logger.info("Building interaction matrix")
        interactions_list = [(row['user_id'], row['item_id'], row['rating']) 
                            for _, row in train_data.iterrows()]
        
        self.interactions_matrix, weights = self.dataset.build_interactions(interactions_list)
        
        # Initialize and train model
        logger.info("Training LightFM model")
        self.model = LightFM(
            loss=self.hyperparams["loss"],
            learning_rate=self.hyperparams["learning_rate"],
            no_components=self.hyperparams["no_components"],
            item_alpha=self.hyperparams["item_alpha"],
            user_alpha=self.hyperparams["user_alpha"],
            max_sampled=self.hyperparams["max_sampled"],
            random_state=self.hyperparams["random_state"]
        )
        
        self.model.fit(
            interactions=self.interactions_matrix,
            user_features=user_features_matrix,
            item_features=item_features_matrix,
            epochs=self.hyperparams["epochs"],
            verbose=True
        )
        
        # Store feature matrices for later use
        self.user_features = user_features_matrix
        self.item_features = item_features_matrix
        
        # Update metadata
        training_time = time.time() - start_time
        self.metadata.update({
            "num_users": len(self.user_index),
            "num_items": len(self.item_index),
            "has_user_features": user_features is not None,
            "has_item_features": item_features is not None,
            "hyperparameters": self.hyperparams,
            "training_time": training_time
        })
        
        self.fitted = True
        logger.info(f"LightFM model training completed in {training_time:.2f} seconds")

    def _process_user_features(self, user_features: pd.DataFrame) -> sparse.csr_matrix:
        """
        Process user features into a format suitable for LightFM.
        
        Args:
            user_features: DataFrame with user features
        
        Returns:
            Sparse matrix of user features
        """
        # Identify feature columns (all columns except user_id)
        feature_cols = [col for col in user_features.columns if col != 'user_id']
        
        # Fit feature columns to dataset
        for col in feature_cols:
            unique_values = user_features[col].unique()
            for val in unique_values:
                # Use column_value format for features
                self.dataset.fit_partial(user_features=[f"{col}_{val}"])
        
        # Build feature tuples for each user
        user_features_list = []
        for _, row in user_features.iterrows():
            user_id = row['user_id']
            if user_id in self.user_index:  # Only include users in our index
                features = []
                for col in feature_cols:
                    val = row[col]
                    features.append(f"{col}_{val}")
                user_features_list.append((user_id, features))
        
        # Build user features matrix
        user_features_matrix, _ = self.dataset.build_user_features(user_features_list)
        return user_features_matrix

    def _process_item_features(self, item_features: pd.DataFrame) -> sparse.csr_matrix:
        """
        Process item features into a format suitable for LightFM.
        
        Args:
            item_features: DataFrame with item features
        
        Returns:
            Sparse matrix of item features
        """
        # Identify feature columns (all columns except item_id)
        feature_cols = [col for col in item_features.columns if col != 'item_id']
        
        # Fit feature columns to dataset
        for col in feature_cols:
            unique_values = item_features[col].unique()
            for val in unique_values:
                # Use column_value format for features
                self.dataset.fit_partial(item_features=[f"{col}_{val}"])
        
        # Build feature tuples for each item
        item_features_list = []
        for _, row in item_features.iterrows():
            item_id = row['item_id']
            if item_id in self.item_index:  # Only include items in our index
                features = []
                for col in feature_cols:
                    val = row[col]
                    features.append(f"{col}_{val}")
                item_features_list.append((item_id, features))
        
        # Build item features matrix
        item_features_matrix, _ = self.dataset.build_item_features(item_features_list)
        return item_features_matrix

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
        
        # Handle cold-start users
        if user_id not in self.user_index:
            logger.warning(f"User {user_id} not in training data, using average user representation")
            # Return empty list for cold-start users
            return []
        
        user_idx = self.user_index[user_id]
        
        # Determine which items to score
        if item_ids is not None:
            # Filter valid items
            valid_items = [i for i in item_ids if i in self.item_index]
            if not valid_items:
                logger.warning(f"None of the requested items are in the training data")
                return []
            
            item_indices = [self.item_index[i] for i in valid_items]
        else:
            # Use all items
            valid_items = list(self.item_index.keys())
            item_indices = list(range(len(self.item_index)))
        
        # Create arrays of user indices repeated for each item
        # For LightFM, we need to provide arrays of equal length
        user_indices = [user_idx] * len(item_indices)
        
        # Get predictions
        scores = self.model.predict(
            user_ids=user_indices,  # Array of the same user_idx repeated
            item_ids=item_indices,  # Array of all item indices
            user_features=self.user_features,
            item_features=self.item_features
        )
        
        # Convert item indices back to item IDs and create recommendations
        recommendations = []
        for i, score in enumerate(scores):
            item_id = valid_items[i]
            recommendations.append((item_id, float(score)))
        
        # Sort by score and return top n
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out items the user has already interacted with
        if self.interactions_matrix is not None:
            # Get the items the user has interacted with
            user_interactions = self.interactions_matrix.tocsr()[user_idx].indices
            interacted_items = {self.reverse_item_index[idx] for idx in user_interactions}
            
            # Filter recommendations
            recommendations = [(item_id, score) for item_id, score in recommendations 
                              if item_id not in interacted_items]
        
        return recommendations[:n]

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
        
        results = {}
        for user_id in user_ids:
            try:
                results[user_id] = self.predict(user_id, n=n)
            except Exception as e:
                logger.error(f"Error generating predictions for user {user_id}: {str(e)}")
                results[user_id] = []
        
        return results

    def evaluate(self, test_data: pd.DataFrame, k: int = 10, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
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
            metrics = ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'map_at_k', 'rmse', 'mae']
        
        results = {}
        
        # Create test interactions for LightFM evaluation
        test_interactions_list = []
        for _, row in test_data.iterrows():
            user_id, item_id = row['user_id'], row['item_id']
            if user_id in self.user_index and item_id in self.item_index:
                test_interactions_list.append((user_id, item_id, 1.0))  # Binary interaction
        
        if not test_interactions_list:
            logger.warning("No valid test interactions found")
            return results
            
        test_interactions, _ = self.dataset.build_interactions(test_interactions_list)
        
        # LightFM native metrics
        if 'precision_at_k' in metrics:
            try:
                precision = precision_at_k(
                    self.model, 
                    test_interactions, 
                    k=k,
                    user_features=self.user_features,
                    item_features=self.item_features
                ).mean()
                results['precision_at_k'] = float(precision)
            except Exception as e:
                logger.error(f"Error calculating precision@k: {str(e)}")
        
        if 'recall_at_k' in metrics:
            try:
                recall = recall_at_k(
                    self.model, 
                    test_interactions, 
                    k=k,
                    user_features=self.user_features,
                    item_features=self.item_features
                ).mean()
                results['recall_at_k'] = float(recall)
            except Exception as e:
                logger.error(f"Error calculating recall@k: {str(e)}")
        
        if 'auc' in metrics:
            try:
                auc = auc_score(
                    self.model, 
                    test_interactions,
                    user_features=self.user_features,
                    item_features=self.item_features
                ).mean()
                results['auc'] = float(auc)
            except Exception as e:
                logger.error(f"Error calculating AUC: {str(e)}")
        
        # Custom implementation for NDCG@k
        if 'ndcg_at_k' in metrics:
            try:
                ndcg = self._calculate_ndcg_at_k(test_data, k)
                results['ndcg_at_k'] = float(ndcg)
            except Exception as e:
                logger.error(f"Error calculating NDCG@k: {str(e)}")
        
        # Custom implementation for MAP@k
        if 'map_at_k' in metrics:
            try:
                map_score = self._calculate_map_at_k(test_data, k)
                results['map_at_k'] = float(map_score)
            except Exception as e:
                logger.error(f"Error calculating MAP@k: {str(e)}")
        
        # Rating prediction metrics (RMSE, MAE)
        if any(m in metrics for m in ['rmse', 'mae']):
            predictions = []
            actuals = []
            
            for _, row in test_data.iterrows():
                user_id, item_id, rating = row['user_id'], row['item_id'], row['rating']
                if user_id in self.user_index and item_id in self.item_index:
                    try:
                        user_idx = self.user_index[user_id]
                        item_idx = self.item_index[item_id]
                        
                        pred = self.model.predict(
                            user_ids=[user_idx],
                            item_ids=[item_idx],
                            user_features=self.user_features,
                            item_features=self.item_features
                        )[0]
                        
                        predictions.append(pred)
                        actuals.append(rating)
                    except Exception as e:
                        logger.debug(f"Error predicting for user {user_id}, item {item_id}: {str(e)}")
            
            if predictions:
                if 'rmse' in metrics:
                    rmse = np.sqrt(mean_squared_error(actuals, predictions))
                    results['rmse'] = float(rmse)
                
                if 'mae' in metrics:
                    mae = mean_absolute_error(actuals, predictions)
                    results['mae'] = float(mae)
        
        # Update metadata
        self.metadata['performance'] = results
        
        return results
    
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
            if user_id not in self.user_index:
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
            if user_id not in self.user_index:
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
            "model": self.model,
            "dataset": self.dataset,
            "user_index": self.user_index,
            "item_index": self.item_index,
            "reverse_user_index": self.reverse_user_index,
            "reverse_item_index": self.reverse_item_index,
            "user_features": self.user_features,
            "item_features": self.item_features,
            "interactions_matrix": self.interactions_matrix,
            "metadata": self.metadata,
            "hyperparams": self.hyperparams,
            "fitted": self.fitted
        }
        
        # Save to file
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'LightFMModel':
        """
        Load a previously saved model.
        
        Args:
            path: Path to load the model from
        
        Returns:
            Loaded LightFMModel instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load from file
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create model instance
        model = cls()
        model.model = save_data["model"]
        model.dataset = save_data["dataset"]
        model.user_index = save_data["user_index"]
        model.item_index = save_data["item_index"]
        model.reverse_user_index = save_data["reverse_user_index"]
        model.reverse_item_index = save_data["reverse_item_index"]
        model.user_features = save_data["user_features"]
        model.item_features = save_data["item_features"]
        model.interactions_matrix = save_data["interactions_matrix"]
        model.metadata = save_data["metadata"]
        model.hyperparams = save_data["hyperparams"]
        model.fitted = save_data["fitted"]
        
        logger.info(f"Model loaded from {path}")
        return model
