import os
import pandas as pd
import numpy as np
import logging
import time
import pickle
from typing import Dict, List, Any, Optional, Tuple
import itertools

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class BaselineModel(BaseModel):
    """
    Baseline recommender model implementing two basic strategies:
    1. Rating prediction: User's mean rating
    2. Top-K recommendations: Most popular items
    """
    
    def __init__(self, model_name: str = "baseline"):
        """
        Initialize the Baseline model.
        
        Args:
            model_name: Name of the model
        """
        super().__init__(model_name)
        
        # Default hyperparameters
        self.hyperparams = {
            "recommendation_strategy": "popularity",  # 'popularity' or 'user_mean'
            "remove_seen": True                       # Whether to remove seen items from recommendations
        }
        
        self.user_means = None
        self.item_popularity = None
        self.col_user = "user_id"
        self.col_item = "item_id"
        self.col_rating = "rating"
        self.col_timestamp = "timestamp"
        self.col_prediction = "prediction"
        
        # Training data reference for coverage calculation
        self.train_data = None
    
    def _calculate_user_means(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mean ratings for each user.
        
        Args:
            train_data: Training data with user_id, item_id, and rating columns
            
        Returns:
            DataFrame with user means
        """
        user_means = train_data.groupby(self.col_user)[self.col_rating].mean()
        user_means = user_means.to_frame().reset_index()
        user_means.rename(columns={self.col_rating: "mean_rating"}, inplace=True)
        return user_means
    
    def _calculate_item_popularity(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate popularity scores for each item (based on number of ratings).
        
        Args:
            train_data: Training data with user_id, item_id, and rating columns
            
        Returns:
            DataFrame with item popularity scores
        """
        item_counts = train_data[self.col_item].value_counts().to_frame().reset_index()
        item_counts.columns = [self.col_item, "popularity_score"]
        return item_counts
    
    def fit(self, train_data: pd.DataFrame, user_features: Optional[pd.DataFrame] = None, 
            item_features: Optional[pd.DataFrame] = None) -> None:
        """
        Train the Baseline model on the provided data.
        
        Args:
            train_data: Training data with user_id, item_id, and rating columns
            user_features: Optional DataFrame with user features (not used in baseline)
            item_features: Optional DataFrame with item features (not used in baseline)
        """
        start_time = time.time()
        logger.info(f"Fitting Baseline model with hyperparameters: {self.hyperparams}")
        
        # Store training data for later use
        self.train_data = train_data
        
        # Create user and item indices
        self._create_indices(train_data)
        
        # Calculate user means for rating prediction
        logger.info("Calculating user means")
        self.user_means = self._calculate_user_means(train_data)
        
        # Calculate item popularity for top-k recommendations
        logger.info("Calculating item popularity")
        self.item_popularity = self._calculate_item_popularity(train_data)
        
        # Update metadata
        self.metadata.update({
            "num_users": len(self.user_index),
            "num_items": len(self.item_index),
            "hyperparameters": self.hyperparams,
            "training_time": time.time() - start_time
        })
        
        self.fitted = True
        logger.info(f"Baseline model fitting completed in {self.metadata['training_time']:.2f} seconds")
    
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
        
        if self.hyperparams["recommendation_strategy"] == "user_mean":
            # For rating prediction, use user's mean rating
            if user_id not in self.user_index:
                # For new users, use global average
                global_mean = self.train_data[self.col_rating].mean()
                score = global_mean
            else:
                user_mean = self.user_means[self.user_means[self.col_user] == user_id]["mean_rating"].values[0]
                score = user_mean
            
            if item_ids is not None:
                # Return same score for all items
                recommendations = [(item_id, score) for item_id in item_ids if item_id in self.item_index]
            else:
                # For general recommendations, return top popular items
                recommendations = [(row[self.col_item], score) 
                                   for _, row in self.item_popularity.head(n).iterrows()]
        
        else:  # popularity strategy
            if self.hyperparams["remove_seen"] and user_id in self.user_index:
                # Remove items seen by the user in training data
                seen_items = set(self.train_data[self.train_data[self.col_user] == user_id][self.col_item])
            else:
                seen_items = set()
            
            if item_ids is not None:
                # Rank specified items by popularity
                item_scores = self.item_popularity.set_index(self.col_item)
                recommendations = []
                for item_id in item_ids:
                    if item_id in item_scores.index and item_id not in seen_items:
                        score = item_scores.loc[item_id, "popularity_score"]
                        recommendations.append((item_id, float(score)))
            else:
                # Return top n popular items
                recommendations = []
                for _, row in self.item_popularity.iterrows():
                    item_id = row[self.col_item]
                    if item_id not in seen_items:
                        recommendations.append((item_id, float(row["popularity_score"])))
                    if len(recommendations) >= n:
                        break
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x[1], reverse=True)
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
        
        recommendations = {}
        
        for user_id in user_ids:
            recommendations[user_id] = self.predict(user_id, n=n)
        
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
        
        # Get unique users from test data
        test_users = test_data[self.col_user].unique()
        
        # Generate recommendations for each user
        all_recommendations = self.predict_batch(test_users, n=k)
        
        # Convert test data to a dictionary for easier lookup
        test_dict = test_data.groupby(self.col_user)[self.col_item].apply(list).to_dict()
        
        # Calculate ranking metrics
        precision_scores = []
        recall_scores = []
        ndcg_scores = []
        ap_scores = []
        
        for user_id in test_users:
            if user_id not in test_dict:
                continue
            
            actual_items = set(test_dict[user_id])
            recommended_items = [item_id for item_id, _ in all_recommendations[user_id]]
            
            # Precision@k
            hits = len(set(recommended_items) & actual_items)
            precision_scores.append(hits / k)
            
            # Recall@k
            recall_scores.append(hits / len(actual_items) if actual_items else 0)
            
            # NDCG@k
            dcg = 0.0
            for i, item_id in enumerate(recommended_items):
                if item_id in actual_items:
                    dcg += 1.0 / np.log2(i + 2)
            
            idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(actual_items), k))])
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0)
            
            # Average Precision
            if not recommended_items:
                ap_scores.append(0.0)
                continue
            
            hits_at_k = 0
            precision_at_k = 0.0
            for i, item_id in enumerate(recommended_items):
                if item_id in actual_items:
                    hits_at_k += 1
                    precision_at_k += hits_at_k / (i + 1)
            
            ap_scores.append(precision_at_k / min(len(actual_items), k) if actual_items else 0)
        
        # Store accuracy metrics
        if 'precision_at_k' in metrics:
            results['precision_at_k'] = np.mean(precision_scores) if precision_scores else 0.0
        if 'recall_at_k' in metrics:
            results['recall_at_k'] = np.mean(recall_scores) if recall_scores else 0.0
        if 'ndcg_at_k' in metrics:
            results['ndcg_at_k'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        if 'map_at_k' in metrics:
            results['map_at_k'] = np.mean(ap_scores) if ap_scores else 0.0
        
        # Calculate rating prediction metrics (only for user_mean strategy)
        if any(m in metrics for m in ['rmse', 'mae']) and self.hyperparams["recommendation_strategy"] == "user_mean":
            predictions = []
            actuals = []
            
            for _, row in test_data.iterrows():
                user_id = row[self.col_user]
                item_id = row[self.col_item]
                actual_rating = row[self.col_rating]
                
                # Get prediction
                pred_items = self.predict(user_id, [item_id], n=1)
                if pred_items:
                    predicted_rating = pred_items[0][1]
                else:
                    # For new items, use global average
                    predicted_rating = self.train_data[self.col_rating].mean()
                
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
            
            # Calculate RMSE and MAE
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            if 'rmse' in metrics:
                results['rmse'] = np.sqrt(np.mean((predictions - actuals) ** 2))
            if 'mae' in metrics:
                results['mae'] = np.mean(np.abs(predictions - actuals))
        
        # Calculate beyond-accuracy metrics
        if any(m in metrics for m in ['coverage', 'novelty', 'diversity']):
            from src.utils.evaluation_metrics import calculate_all_beyond_accuracy_metrics
            
            # Calculate beyond-accuracy metrics using the same set of recommendations
            beyond_accuracy_metrics = calculate_all_beyond_accuracy_metrics(
                all_recommendations, self.train_data, None
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
        For baseline model, return most popular items (excluding the query item).
        
        Args:
            item_id: ID of the item
            n: Number of similar items to return
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")
        
        similar_items = []
        
        # Return most popular items (excluding the query item)
        for _, row in self.item_popularity.iterrows():
            if row[self.col_item] != item_id:
                similar_items.append((row[self.col_item], float(row["popularity_score"])))
            if len(similar_items) >= n:
                break
        
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
        
        # Prepare data to save
        data = {
            "model": None,  # Baseline doesn't have a complex model object
            "user_index": self.user_index,
            "item_index": self.item_index,
            "reverse_user_index": self.reverse_user_index,
            "reverse_item_index": self.reverse_item_index,
            "metadata": self.metadata,
            "fitted": self.fitted,
            "hyperparams": self.hyperparams,
            "user_means": self.user_means,
            "item_popularity": self.item_popularity
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
        self.user_index = data["user_index"]
        self.item_index = data["item_index"]
        self.reverse_user_index = data["reverse_user_index"]
        self.reverse_item_index = data["reverse_item_index"]
        self.metadata = data["metadata"]
        self.fitted = data["fitted"]
        self.hyperparams = data["hyperparams"]
        self.user_means = data["user_means"]
        self.item_popularity = data["item_popularity"]
        
        logger.info(f"Model loaded from {path}")