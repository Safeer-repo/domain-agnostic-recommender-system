import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder

from src.models.model_registry import ModelRegistry
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

class ModelSelector:
    """
    Intelligent model selection component that analyzes dataset characteristics
    and selects the most appropriate recommendation algorithm.
    """
    
    def __init__(self, artifacts_dir: str = None, data_dir: str = None):
        """
        Initialize the model selector.
        
        Args:
            artifacts_dir: Directory where model artifacts are stored
            data_dir: Directory where data is stored
        """
        self.artifacts_dir = artifacts_dir
        self.data_dir = data_dir
        self.model_registry = ModelRegistry(artifacts_dir=artifacts_dir)
        self.preprocessing_pipeline = PreprocessingPipeline(data_dir=data_dir)
        
        # Model selection priorities based on dataset characteristics
        self.model_priorities = {
            "als": {
                "large_sparse": 10,
                "implicit_feedback": 8,
                "extreme_sparsity": 9,
                "balanced_distribution": 7
            },
            "svd": {
                "explicit_feedback": 10,
                "small_dataset": 9,
                "dense_interactions": 8,
                "balanced_distribution": 6
            },
            "lightfm": {
                "long_tail": 10,
                "hybrid_data": 9,
                "cold_start_potential": 8,
                "feature_rich": 7
            },
            "sar": {
                "time_series": 10,
                "session_based": 9,
                "high_temporal_dynamics": 8,
                "item_similarity_focused": 7
            },
            "baseline": {
                "cold_start": 10,
                "very_small_dataset": 9,
                "new_system": 8,
                "popularity_driven": 7
            }
        }
        
    def analyze_dataset(self, train_data: pd.DataFrame, test_data: pd.DataFrame = None,
                        user_features: pd.DataFrame = None, item_features: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze dataset characteristics to determine the most suitable model.
        
        Args:
            train_data: Training dataset
            test_data: Test dataset
            user_features: Optional user features
            item_features: Optional item features
            
        Returns:
            Dictionary of dataset characteristics
        """
        characteristics = {}
        
        # Basic dataset size characteristics
        characteristics["num_users"] = train_data["user_id"].nunique()
        characteristics["num_items"] = train_data["item_id"].nunique()
        characteristics["num_interactions"] = len(train_data)
        
        # Sparsity (percentage of empty user-item interactions)
        total_possible = characteristics["num_users"] * characteristics["num_items"]
        characteristics["sparsity"] = 1.0 - (characteristics["num_interactions"] / total_possible)
        
        # Rating distribution
        characteristics["rating_mean"] = train_data["rating"].mean()
        characteristics["rating_std"] = train_data["rating"].std()
        characteristics["rating_min"] = train_data["rating"].min()
        characteristics["rating_max"] = train_data["rating"].max()
        characteristics["rating_unique"] = train_data["rating"].nunique()
        
        # User activity distribution
        user_counts = train_data.groupby("user_id").size()
        characteristics["user_activity_mean"] = user_counts.mean()
        characteristics["user_activity_std"] = user_counts.std()
        characteristics["user_activity_min"] = user_counts.min()
        characteristics["user_activity_max"] = user_counts.max()
        characteristics["user_activity_entropy"] = entropy(user_counts.value_counts(normalize=True))
        
        # Item popularity distribution
        item_counts = train_data.groupby("item_id").size()
        characteristics["item_popularity_mean"] = item_counts.mean()
        characteristics["item_popularity_std"] = item_counts.std()
        characteristics["item_popularity_min"] = item_counts.min()
        characteristics["item_popularity_max"] = item_counts.max()
        characteristics["item_popularity_entropy"] = entropy(item_counts.value_counts(normalize=True))
        
        # Distribution metrics
        gini_coefficient = self._calculate_gini(item_counts.values)
        characteristics["item_popularity_gini"] = gini_coefficient
        characteristics["user_activity_gini"] = self._calculate_gini(user_counts.values)
        
 # Temporal characteristics
        if "timestamp" in train_data.columns:
            characteristics["has_temporal_data"] = True
            
            # Convert timestamp to numeric if it's not already
            if train_data["timestamp"].dtype == object or train_data["timestamp"].dtype == str:
                # Try to convert to integer (common for MovieLens timestamps)
                try:
                    timestamps = pd.to_numeric(train_data["timestamp"], errors='coerce')
                    # If that fails or has NaN values, try parsing as datetime
                    if timestamps.isna().any():
                        timestamps = pd.to_datetime(train_data["timestamp"]).astype(int) / 10**9
                except:
                    timestamps = pd.to_datetime(train_data["timestamp"]).astype(int) / 10**9
            else:
                timestamps = train_data["timestamp"].astype(float)
            
            characteristics["temporal_span_days"] = (timestamps.max() - timestamps.min()) / (24 * 60 * 60)
            characteristics["interactions_per_day"] = characteristics["num_interactions"] / characteristics["temporal_span_days"]
            
            # Time-based activity patterns
            temp_data = train_data.copy()
            temp_data["timestamp"] = timestamps
            time_diffs = temp_data.sort_values("timestamp").groupby("user_id")["timestamp"].diff().dropna()
            
            if len(time_diffs) > 0:
                characteristics["avg_time_between_interactions"] = time_diffs.mean()
                characteristics["temporal_interaction_std"] = time_diffs.std()
            else:
                characteristics["avg_time_between_interactions"] = 0
                characteristics["temporal_interaction_std"] = 0
        else:
            characteristics["has_temporal_data"] = False
        
        # Rating type characteristics
        unique_ratings = train_data["rating"].nunique()
        characteristics["binary_ratings"] = unique_ratings <= 2
        characteristics["explicit_ratings"] = unique_ratings > 5
        characteristics["implicit_feedback"] = characteristics["binary_ratings"] or all(train_data["rating"] >= 0)
        
        # Cold start analysis
        if test_data is not None:
            new_users = set(test_data["user_id"]) - set(train_data["user_id"])
            new_items = set(test_data["item_id"]) - set(train_data["item_id"])
            characteristics["cold_start_ratio_users"] = len(new_users) / test_data["user_id"].nunique()
            characteristics["cold_start_ratio_items"] = len(new_items) / test_data["item_id"].nunique()
        else:
            characteristics["cold_start_ratio_users"] = 0
            characteristics["cold_start_ratio_items"] = 0
        
        # Feature richness analysis
        if user_features is not None:
            characteristics["num_user_features"] = user_features.shape[1] - 1  # Exclude user_id
            characteristics["user_feature_sparsity"] = user_features.isna().sum().sum() / (user_features.shape[0] * (user_features.shape[1] - 1))
        else:
            characteristics["num_user_features"] = 0
            characteristics["user_feature_sparsity"] = 1.0
            
        if item_features is not None:
            characteristics["num_item_features"] = item_features.shape[1] - 1  # Exclude item_id
            characteristics["item_feature_sparsity"] = item_features.isna().sum().sum() / (item_features.shape[0] * (item_features.shape[1] - 1))
        else:
            characteristics["num_item_features"] = 0
            characteristics["item_feature_sparsity"] = 1.0
        
        # Dataset complexity metrics
        characteristics["avg_items_per_user"] = train_data.groupby("user_id")["item_id"].nunique().mean()
        characteristics["avg_users_per_item"] = train_data.groupby("item_id")["user_id"].nunique().mean()
        characteristics["recommendation_difficulty"] = (characteristics["sparsity"] * characteristics["item_popularity_gini"] * characteristics["user_activity_gini"]) ** 0.33
        
        logger.info(f"Dataset analysis complete")
        logger.debug(f"Dataset characteristics: {characteristics}")
        return characteristics
    
    def _calculate_gini(self, array: np.ndarray) -> float:
        """
        Calculate the Gini coefficient of a numpy array.
        The Gini coefficient measures inequality (0 = complete equality, 1 = complete inequality)
        
        Args:
            array: Input array
            
        Returns:
            Gini coefficient
        """
        # Convert to float to avoid type issues
        array = array.astype(float)
        
        if np.amin(array) < 0:
            array -= np.amin(array)
        
        # Add small constant to avoid division by zero
        array = array + 0.0000001
        
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))
    
    def calculate_model_scores(self, characteristics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate suitability scores for each model based on dataset characteristics.
        
        Args:
            characteristics: Dataset characteristics
            
        Returns:
            Dictionary mapping model names to suitability scores
        """
        scores = {}
        
        # Define characteristic categories
        is_large_dataset = characteristics["num_interactions"] > 100000
        is_very_sparse = characteristics["sparsity"] > 0.995
        is_long_tail = characteristics["item_popularity_gini"] > 0.8
        has_time_data = characteristics.get("has_temporal_data", False)
        has_user_features = characteristics["num_user_features"] > 0
        has_item_features = characteristics["num_item_features"] > 0
        has_cold_start = characteristics["cold_start_ratio_users"] > 0.1 or characteristics["cold_start_ratio_items"] > 0.1
        is_implicit = characteristics["implicit_feedback"]
        is_explicit = characteristics["explicit_ratings"]
        is_very_small = characteristics["num_interactions"] < 1000
        
        # Calculate scores for each model
        for model_name, priorities in self.model_priorities.items():
            score = 0
            
            if model_name == "als":
                if is_large_dataset and is_very_sparse:
                    score += priorities["large_sparse"]
                if is_implicit:
                    score += priorities["implicit_feedback"]
                if characteristics["sparsity"] > 0.999:
                    score += priorities["extreme_sparsity"]
                if abs(characteristics["item_popularity_gini"] - 0.5) < 0.3:
                    score += priorities["balanced_distribution"]
            
            elif model_name == "svd":
                if is_explicit:
                    score += priorities["explicit_feedback"]
                if characteristics["num_interactions"] < 50000:
                    score += priorities["small_dataset"]
                if characteristics["sparsity"] < 0.98:
                    score += priorities["dense_interactions"]
                if abs(characteristics["item_popularity_gini"] - 0.5) < 0.3:
                    score += priorities["balanced_distribution"]
            
            elif model_name == "lightfm":
                if is_long_tail:
                    score += priorities["long_tail"]
                if has_user_features or has_item_features:
                    score += priorities["hybrid_data"]
                if has_cold_start:
                    score += priorities["cold_start_potential"]
                if (has_user_features and has_item_features) or characteristics["num_item_features"] > 5:
                    score += priorities["feature_rich"]
            
            elif model_name == "sar":
                if has_time_data:
                    score += priorities["time_series"]
                if characteristics.get("avg_time_between_interactions", float('inf')) < 3600:  # Less than 1 hour
                    score += priorities["session_based"]
                if characteristics.get("temporal_interaction_std", 0) > 86400:  # High temporal variability
                    score += priorities["high_temporal_dynamics"]
                if characteristics["avg_items_per_user"] < 10:
                    score += priorities["item_similarity_focused"]
            
            elif model_name == "baseline":
                if has_cold_start:
                    score += priorities["cold_start"]
                if is_very_small:
                    score += priorities["very_small_dataset"]
                if characteristics["num_users"] < 100:
                    score += priorities["new_system"]
                if characteristics["item_popularity_gini"] > 0.9:
                    score += priorities["popularity_driven"]
            
            scores[model_name] = score
        
        return scores
    
    def select_best_model(self, domain: str, dataset: str, 
                          characteristics: Optional[Dict[str, Any]] = None,
                          metric: str = "map_at_k") -> Tuple[str, Dict[str, Any]]:
        """
        Select the best model for the given dataset based on characteristics.
        
        Args:
            domain: Domain name (entertainment, ecommerce, education)
            dataset: Dataset name
            characteristics: Optional pre-computed dataset characteristics
            metric: Performance metric to optimize (defaults to map_at_k as per the proposal)
            
        Returns:
            Tuple of (best_model_name, best_hyperparameters)
        """
        logger.info(f"Selecting best model for {domain}/{dataset} with metric {metric}")
        
        if characteristics is None:
            # Load the data and analyze if characteristics not provided
            train_data, test_data = self.preprocessing_pipeline.preprocess(domain, dataset)
            features = self.preprocessing_pipeline.create_features(domain, dataset)
            user_features = features.get("user_features")
            item_features = features.get("item_features")
            characteristics = self.analyze_dataset(train_data, test_data, user_features, item_features)
        
        # Check if we have tuning results available for all models
        models_to_check = ["als", "svd", "lightfm", "sar", "baseline"]
        tuning_results = {}
        
        for model_name in models_to_check:
            # Different path structure for different models
            if model_name == "als":
                model_tuning_path = os.path.join(
                    self.artifacts_dir, "tuning_results", domain, dataset, "best_params.json"
                )
            else:
                model_tuning_path = os.path.join(
                    self.artifacts_dir, "tuning_results", domain, dataset, model_name, "best_params.json"
                )
            
            if os.path.exists(model_tuning_path):
                try:
                    with open(model_tuning_path, 'r') as f:
                        tuning_data = json.load(f)
                        tuning_results[model_name] = tuning_data
                        logger.info(f"Loaded tuning results for {model_name}")
                except Exception as e:
                    logger.error(f"Error loading tuning results for {model_name}: {str(e)}")
        
        # If we have tuning results, use them to select the best model
        if tuning_results:
            # Priority list of metrics as per the proposal
            priority_metrics = [metric, "map_at_k", "coverage", "novelty", "diversity"]
            
            for current_metric in priority_metrics:
                best_model = None
                best_score = float('-inf') if current_metric not in ["rmse", "mae"] else float('inf')
                best_params = None
                
                for model_name, results in tuning_results.items():
                    # Check if this model has been evaluated with the current metric
                    if results.get("metric") == current_metric:
                        score = results.get("best_score", float('-inf'))
                        
                        # Compare scores (lower is better for RMSE and MAE, higher is better for others)
                        is_better = False
                        if current_metric in ["rmse", "mae"]:
                            if score < best_score:
                                is_better = True
                        else:
                            if score > best_score:
                                is_better = True
                        
                        if is_better:
                            best_score = score
                            best_model = model_name
                            best_params = results.get("best_params", {})
                
                # If we found a best model for this metric, return it
                if best_model:
                    logger.info(f"Selected {best_model} based on tuning results for {current_metric} (score: {best_score})")
                    return best_model, best_params
            
            # If we have tuning results but none match our metrics, use the model with the best MAP@K
            if "map_at_k" in priority_metrics:
                best_map_model = None
                best_map_score = float('-inf')
                best_map_params = None
                
                for model_name, results in tuning_results.items():
                    if results.get("metric") == "map_at_k":
                        score = results.get("best_score", float('-inf'))
                        if score > best_map_score:
                            best_map_score = score
                            best_map_model = model_name
                            best_map_params = results.get("best_params", {})
                
                if best_map_model:
                    logger.info(f"Selected {best_map_model} based on MAP@K (score: {best_map_score})")
                    return best_map_model, best_map_params
        
        # If no tuning results or no good matches, use heuristics
        model_scores = self.calculate_model_scores(characteristics)
        best_model = max(model_scores.items(), key=lambda x: x[1])[0]
        
        # Get default hyperparameters for the selected model
        if best_model == "als":
            default_hyperparams = {
                "factors": 100,
                "regularization": 0.01,
                "iterations": 15,
                "alpha": 40.0 if characteristics["implicit_feedback"] else 1.0
            }
        elif best_model == "svd":
            default_hyperparams = {
                "n_factors": 100,
                "n_epochs": 20,
                "lr_all": 0.005,
                "reg_all": 0.02
            }
        elif best_model == "lightfm":
            default_hyperparams = {
                "no_components": 64,
                "loss": "bpr" if characteristics["implicit_feedback"] else "warp",
                "learning_rate": 0.05,
                "item_alpha": 0.001,
                "user_alpha": 0.001,
                "epochs": 20
            }
        elif best_model == "sar":
            default_hyperparams = {
                "similarity_type": "jaccard",
                "time_decay_coefficient": 30,
                "use_timedecay": characteristics.get("has_temporal_data", False),
                "remove_seen": True
            }
        elif best_model == "baseline":
            default_hyperparams = {
                "recommendation_strategy": "popularity",
                "remove_seen": True
            }
        
        logger.info(f"Selected {best_model} based on dataset characteristics (heuristic score: {model_scores[best_model]})")
        return best_model, default_hyperparams
    
    def get_best_model(self, domain: str, dataset: str, metric: str = "map_at_k"):
        """
        Get the best model instance for the given domain and dataset.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            metric: Performance metric to optimize (default changed to map_at_k per proposal)
            
        Returns:
            Instantiated and configured model (not fitted)
        """
        # Load the data
        train_data, test_data = self.preprocessing_pipeline.preprocess(domain, dataset)
        features = self.preprocessing_pipeline.create_features(domain, dataset)
        user_features = features.get("user_features")
        item_features = features.get("item_features")
        
        # Analyze dataset characteristics
        characteristics = self.analyze_dataset(train_data, test_data, user_features, item_features)
        
        # Select the best model and hyperparameters
        model_name, hyperparams = self.select_best_model(domain, dataset, characteristics, metric)
        
        # Get model instance from registry
        model = self.model_registry.get_model(model_name)
        
        # Set hyperparameters
        model.set_hyperparameters(**hyperparams)
        
        logger.info(f"Configured {model_name} for {domain}/{dataset} optimized for {metric}")
        
        return model
    
    def fit_best_model(self, domain: str, dataset: str, metric: str = "map_at_k"):
        """
        Fit the best model for the given domain and dataset.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            metric: Performance metric to optimize (default changed to map_at_k per proposal)
            
        Returns:
            Fitted model
        """
        # Load the data
        train_data, _ = self.preprocessing_pipeline.preprocess(domain, dataset)
        features = self.preprocessing_pipeline.create_features(domain, dataset)
        
        # Get the best model (not fitted)
        model = self.get_best_model(domain, dataset, metric)
        
        # Fit the model
        logger.info(f"Training best selected model for {domain}/{dataset}")
        model.fit(train_data, features.get("user_features"), features.get("item_features"))
        
        return model
    
    def load_best_tuned_model(self, domain: str, dataset: str, metric: str = "map_at_k"):
        """
        Load the best tuned model for the given domain and dataset.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            metric: Performance metric to optimize (default changed to map_at_k per proposal)
            
        Returns:
            Loaded best tuned model
        """
        # Select the best model type based on tuning results
        model_name, _ = self.select_best_model(domain, dataset, None, metric)
        
        # Construct the path to the best model
        if model_name == "als":
            best_params_path = os.path.join(
                self.artifacts_dir, "tuning_results", domain, dataset, "best_params.json"
            )
        else:
            best_params_path = os.path.join(
                self.artifacts_dir, "tuning_results", domain, dataset, model_name, "best_params.json"
            )
        
        # Load the best model path
        try:
            with open(best_params_path, 'r') as f:
                best_params_data = json.load(f)
                
            model_path = best_params_data.get("model_path")
            
            if not model_path or not os.path.exists(model_path):
                logger.warning(f"Best tuned model path not found: {model_path}")
                return self.fit_best_model(domain, dataset, metric)
            
            # Load the model
            model = self.model_registry.get_model(model_name)
            model.load(model_path)
            
            logger.info(f"Loaded best tuned {model_name} model for {domain}/{dataset}")
            
            return model
        except Exception as e:
            logger.error(f"Error loading best tuned model: {str(e)}")
            logger.info("Falling back to training a new model")
            return self.fit_best_model(domain, dataset, metric)