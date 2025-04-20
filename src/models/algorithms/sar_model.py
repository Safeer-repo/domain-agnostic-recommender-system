import os
import pandas as pd
import numpy as np
import logging
import time
import pickle
from typing import Dict, List, Any, Optional, Tuple
import scipy.sparse as sparse

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class SARModel(BaseModel):
    """
    Simple Algorithm for Recommendation (SAR) model.
    Implements collaborative filtering using item similarity and user affinity matrices.
    """
    
    def __init__(self, model_name: str = "sar"):
        """
        Initialize the SAR model.
        
        Args:
            model_name: Name of the model
        """
        super().__init__(model_name)
        
        # Default hyperparameters
        self.hyperparams = {
            "similarity_type": "jaccard",    # Type of similarity metric (jaccard, lift, or counts)
            "time_decay_coefficient": 30,    # Time decay half-life in days
            "time_now": None,               # Reference time for time decay
            "use_timedecay": True,          # Whether to use time decay
            "remove_seen": True             # Whether to remove seen items from recommendations
        }
        
        self.item_similarity = None
        self.user_affinity = None
        self.item_popularity = None
        self.item_support = None
        self.col_user = "user_id"
        self.col_item = "item_id"
        self.col_rating = "rating"
        self.col_timestamp = "timestamp"
        self.col_prediction = "prediction"
        
        # Training data reference for coverage calculation
        self.train_data = None
    
    def _calculate_item_similarity(self, train_data: pd.DataFrame) -> sparse.csr_matrix:
        """
        Calculate item-item similarity matrix using co-occurrence and the specified similarity metric.
        
        Args:
            train_data: Training data with user_id, item_id columns
            
        Returns:
            Sparse item similarity matrix
        """
        # Create user-item matrix
        user_item = sparse.csr_matrix(
            (np.ones(len(train_data)), 
             (train_data[self.col_user].values, train_data[self.col_item].values))
        )
        
        # Calculate co-occurrence matrix (C = A^T * A)
        item_cooccurrence = user_item.transpose().dot(user_item).tocsr()
        
        # Get diagonal elements (item counts)
        item_counts = item_cooccurrence.diagonal()
        self.item_support = np.asarray(item_counts).squeeze()
        
        # Calculate similarity based on specified metric
        if self.hyperparams["similarity_type"] == "jaccard":
            # Jaccard: s_ij = c_ij / (c_ii + c_jj - c_ij)
            diag = item_cooccurrence.diagonal()
            diag_coo = sparse.diags(diag).tocsr()
            denominator = diag_coo + diag_coo.transpose() - item_cooccurrence
            # Avoid division by zero
            denominator.data[denominator.data == 0] = 1e-10
            # Create sparse matrix with reciprocal values
            recip_denominator = denominator.copy()
            recip_denominator.data = 1.0 / recip_denominator.data
            # Perform element-wise multiplication
            similarity = item_cooccurrence.multiply(recip_denominator).tocsr()
        
        elif self.hyperparams["similarity_type"] == "lift":
            # Lift: s_ij = c_ij / (c_ii * c_jj)
            counts_sqrt = np.sqrt(item_counts)
            counts_matrix = sparse.diags(1.0 / counts_sqrt).tocsr()
            similarity = counts_matrix.dot(item_cooccurrence).dot(counts_matrix)
        
        else:  # counts
            # Counts: s_ij = c_ij
            similarity = item_cooccurrence.tocsr()
        
        # Remove self-similarities by setting diagonal to 0
        if not sparse.issparse(similarity):
            # If similarity is not sparse, convert it to CSR first
            similarity = sparse.csr_matrix(similarity)
        
        # Clear diagonal entries
        # For CSR matrices, we can directly manipulate the diagonal
        similarity = similarity.tolil()  # Convert to LIL for efficient modification
        similarity.setdiag(0)            # Set diagonal to 0
        similarity = similarity.tocsr()  # Convert back to CSR
        similarity.eliminate_zeros()     # Remove explicit zeros
        
        return similarity
    
    def _calculate_user_affinity(self, train_data: pd.DataFrame) -> sparse.csr_matrix:
        """
        Calculate user-item affinity matrix with optional time decay.
        
        Args:
            train_data: Training data with user_id, item_id, rating, and timestamp columns
            
        Returns:
            Sparse user affinity matrix
        """
        if self.hyperparams["use_timedecay"] and self.col_timestamp in train_data.columns:
            # Convert timestamp to numeric values (if it's not already)
            if not np.issubdtype(train_data[self.col_timestamp].dtype, np.number):
                # If timestamp is string or datetime, convert to numeric (seconds since epoch)
                try:
                    timestamps = pd.to_datetime(train_data[self.col_timestamp]).astype('int64') / 1e9
                except:
                    # If already numeric, just convert to float
                    timestamps = train_data[self.col_timestamp].astype(float)
            else:
                timestamps = train_data[self.col_timestamp].astype(float)
            
            # Set the reference time
            if self.hyperparams["time_now"] is None:
                self.hyperparams["time_now"] = float(timestamps.max())
            
            # Calculate time-decayed affinity
            T = self.hyperparams["time_decay_coefficient"]
            t0 = self.hyperparams["time_now"]
            
            # Calculate time decay factor
            time_diff = (t0 - timestamps) / (24 * 60 * 60)  # Convert to days
            decay_factor = np.power(0.5, time_diff / T)
            
            # Apply decay to ratings
            ratings = train_data[self.col_rating].values * decay_factor
        else:
            # Use raw ratings without time decay
            ratings = train_data[self.col_rating].values
        
        # Create user-item affinity matrix
        affinity = sparse.csr_matrix(
            (ratings, 
             (train_data[self.col_user].values, train_data[self.col_item].values))
        )
        
        return affinity
    
    def fit(self, train_data: pd.DataFrame, user_features: Optional[pd.DataFrame] = None, 
            item_features: Optional[pd.DataFrame] = None) -> None:
        """
        Train the SAR model on the provided data.
        
        Args:
            train_data: Training data with user_id, item_id, rating, and timestamp columns
            user_features: Optional DataFrame with user features (not used in SAR)
            item_features: Optional DataFrame with item features (not used in SAR)
        """
        start_time = time.time()
        logger.info(f"Fitting SAR model with hyperparameters: {self.hyperparams}")
        
        # Store training data for later use
        self.train_data = train_data
        
        # Create user and item indices
        self._create_indices(train_data)
        
        # Convert user_id and item_id to internal indices
        train_data = train_data.copy()
        train_data[self.col_user] = train_data[self.col_user].map(self.user_index)
        train_data[self.col_item] = train_data[self.col_item].map(self.item_index)
        
        # Calculate item similarity matrix
        logger.info("Calculating item similarity matrix")
        self.item_similarity = self._calculate_item_similarity(train_data)
        
        # Calculate user affinity matrix
        logger.info("Calculating user affinity matrix")
        self.user_affinity = self._calculate_user_affinity(train_data)
        
        # Calculate item popularity (for cold start and diversity metrics)
        item_counts = train_data.groupby(self.col_item).size()
        item_counts = item_counts.reindex(range(len(self.item_index)), fill_value=0)
        self.item_popularity = item_counts.values / np.sum(item_counts.values)
        
        # Update metadata
        self.metadata.update({
            "num_users": len(self.user_index),
            "num_items": len(self.item_index),
            "hyperparameters": self.hyperparams,
            "training_time": time.time() - start_time
        })
        
        self.fitted = True
        logger.info(f"SAR model fitting completed in {self.metadata['training_time']:.2f} seconds")
    
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
        
        # Calculate recommendation scores
        scores = self.user_affinity[user_idx].dot(self.item_similarity).toarray().squeeze()
        
        # Remove seen items if specified
        if self.hyperparams["remove_seen"]:
            seen_items = self.user_affinity[user_idx].nonzero()[1]
            scores[seen_items] = -np.inf
        
        if item_ids is not None:
            # Filter to only valid items that exist in the training data
            valid_items = [item_id for item_id in item_ids if item_id in self.item_index]
            if not valid_items:
                return []
            
            # Get internal item indices
            item_indices = [self.item_index[item_id] for item_id in valid_items]
            
            # Get scores for specific items
            item_scores = scores[item_indices]
            
            # Create (item_id, score) tuples and sort by score
            recommendations = [(valid_items[i], float(item_scores[i])) for i in range(len(valid_items))]
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            # Return top n
            return recommendations[:n]
        else:
            # Get top n items by score
            top_indices = np.argsort(scores)[-n:][::-1]
            top_scores = scores[top_indices]
            
            # Map internal indices back to original item IDs
            recommendations = [(self.reverse_item_index[idx], float(score)) 
                              for idx, score in zip(top_indices, top_scores)]
            
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
                      'coverage', 'novelty', 'diversity']
        
        results = {}
        
        # Get unique users from test data
        test_users = test_data[self.col_user].unique()
        test_users = [user for user in test_users if user in self.user_index]
        
        if not test_users:
            logger.warning("No valid test users found in the model")
            return {metric: 0.0 for metric in metrics}
        
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
        
        # Get similarity scores for the item
        similarity_scores = self.item_similarity[item_idx].toarray().squeeze()
        
        # Get top n similar items (excluding the item itself)
        top_indices = np.argsort(similarity_scores)[-n-1:-1][::-1]
        top_scores = similarity_scores[top_indices]
        
        # Map internal indices back to original item IDs
        similar_items = [(self.reverse_item_index[idx], float(score)) 
                        for idx, score in zip(top_indices, top_scores) if score > 0]
        
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
        
        # Prepare data to save - handle sparse matrices properly
        data = {
            "model": None,  # SAR doesn't have a complex model object
            "user_index": self.user_index,
            "item_index": self.item_index,
            "reverse_user_index": self.reverse_user_index,
            "reverse_item_index": self.reverse_item_index,
            "metadata": self.metadata,
            "fitted": self.fitted,
            "hyperparams": self.hyperparams,
            "item_similarity_data": self.item_similarity.data,
            "item_similarity_indices": self.item_similarity.indices,
            "item_similarity_indptr": self.item_similarity.indptr,
            "item_similarity_shape": self.item_similarity.shape,
            "user_affinity_data": self.user_affinity.data,
            "user_affinity_indices": self.user_affinity.indices,
            "user_affinity_indptr": self.user_affinity.indptr,
            "user_affinity_shape": self.user_affinity.shape,
            "item_popularity": self.item_popularity,
            "item_support": self.item_support
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
        self.item_popularity = data["item_popularity"]
        self.item_support = data["item_support"]
        
        # Rebuild sparse matrices
        self.item_similarity = sparse.csr_matrix(
            (data["item_similarity_data"], data["item_similarity_indices"], data["item_similarity_indptr"]),
            shape=data["item_similarity_shape"]
        )
        
        self.user_affinity = sparse.csr_matrix(
            (data["user_affinity_data"], data["user_affinity_indices"], data["user_affinity_indptr"]),
            shape=data["user_affinity_shape"]
        )
        
        logger.info(f"Model loaded from {path}")