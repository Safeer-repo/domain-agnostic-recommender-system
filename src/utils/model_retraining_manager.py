import os
import json
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import shutil
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

class ModelRetrainingManager:
    """
    Manages the retraining schedule for recommendation models,
    detecting data distribution shifts, tracking model versions,
    and facilitating A/B testing.
    """
    
    def __init__(self, data_dir: str, models_dir: Optional[str] = None):
        """
        Initialize the model retraining manager.
        
        Args:
            data_dir: Root directory for data storage
            models_dir: Optional directory for model storage (uses data_dir/models if None)
        """
        self.data_dir = data_dir
        self.models_dir = models_dir if models_dir else os.path.join(data_dir, "models")
        self.metrics_dir = os.path.join(data_dir, "model_metrics")
        self.stats_dir = os.path.join(data_dir, "dataset_stats")
        self.ab_tests_dir = os.path.join(data_dir, "ab_tests")
        self._ensure_directory_structure()
        
    def _ensure_directory_structure(self) -> None:
        """Create necessary directory structure if it doesn't exist."""
        for directory in [self.models_dir, self.metrics_dir, self.ab_tests_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory at {directory}")
    
    def _get_domain_dataset_model_path(self, domain: str, dataset: str, algorithm: str) -> str:
        """
        Get the path to store models for a specific domain/dataset/algorithm.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            algorithm: Model algorithm name
            
        Returns:
            Path to model directory
        """
        model_path = os.path.join(self.models_dir, domain, dataset, algorithm)
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        return model_path
    
    def _get_metrics_path(self, domain: str, dataset: str) -> str:
        """
        Get the path to metrics for a specific domain/dataset.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            
        Returns:
            Path to metrics directory
        """
        metrics_path = os.path.join(self.metrics_dir, domain, dataset)
        if not os.path.exists(metrics_path):
            os.makedirs(metrics_path, exist_ok=True)
        return metrics_path
    
    def _get_ab_test_path(self, domain: str, dataset: str) -> str:
        """
        Get the path for A/B test results for a specific domain/dataset.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            
        Returns:
            Path to A/B test directory
        """
        ab_test_path = os.path.join(self.ab_tests_dir, domain, dataset)
        if not os.path.exists(ab_test_path):
            os.makedirs(ab_test_path, exist_ok=True)
        return ab_test_path
    
    def get_latest_model_version(self, domain: str, dataset: str, algorithm: str) -> Optional[str]:
        """
        Get the latest model version for a domain/dataset/algorithm.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            algorithm: Model algorithm name
            
        Returns:
            Latest model version string or None if no models exist
        """
        model_path = self._get_domain_dataset_model_path(domain, dataset, algorithm)
        
        # List all model files
        model_files = [f for f in os.listdir(model_path) 
                      if f.startswith(f"{algorithm}_") and f.endswith(".model")]
        
        if not model_files:
            return None
            
        # Sort by version (timestamp)
        model_files.sort(reverse=True)
        
        # Extract version from filename
        latest_version = model_files[0].replace(f"{algorithm}_", "").replace(".model", "")
        
        return latest_version
    
    def get_model_file_path(self, domain: str, dataset: str, algorithm: str, 
                           version: Optional[str] = None) -> Optional[str]:
        """
        Get the file path for a specific model version.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            algorithm: Model algorithm name
            version: Model version (uses latest if None)
            
        Returns:
            Path to the model file or None if not found
        """
        model_path = self._get_domain_dataset_model_path(domain, dataset, algorithm)
        
        if version is None:
            version = self.get_latest_model_version(domain, dataset, algorithm)
            
        if version is None:
            return None
            
        model_file = os.path.join(model_path, f"{algorithm}_{version}.model")
        
        if os.path.exists(model_file):
            return model_file
        else:
            return None
    
    def save_model_metrics(self, domain: str, dataset: str, algorithm: str, 
                          version: str, metrics: Dict[str, Any]) -> bool:
        """
        Save metrics for a specific model version.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            algorithm: Model algorithm name
            version: Model version
            metrics: Dictionary of model metrics
            
        Returns:
            True if successful, False otherwise
        """
        metrics_path = self._get_metrics_path(domain, dataset)
        metrics_file = os.path.join(metrics_path, f"{algorithm}_{version}_metrics.json")
        
        try:
            # Add timestamp and version info
            metrics["timestamp"] = int(time.time())
            metrics["algorithm"] = algorithm
            metrics["version"] = version
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            logger.info(f"Saved metrics for {algorithm} model version {version} to {metrics_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model metrics: {str(e)}")
            return False
    
    def get_model_metrics(self, domain: str, dataset: str, algorithm: str, 
                        version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get metrics for a specific model version.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            algorithm: Model algorithm name
            version: Model version (uses latest if None)
            
        Returns:
            Dictionary of model metrics or None if not found
        """
        metrics_path = self._get_metrics_path(domain, dataset)
        
        if version is None:
            version = self.get_latest_model_version(domain, dataset, algorithm)
            
        if version is None:
            return None
            
        metrics_file = os.path.join(metrics_path, f"{algorithm}_{version}_metrics.json")
        
        if not os.path.exists(metrics_file):
            return None
            
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            return metrics
        except Exception as e:
            logger.error(f"Error loading model metrics: {str(e)}")
            return None
    
    def get_model_history(self, domain: str, dataset: str, algorithm: str, 
                         limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the history of model versions and their metrics.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            algorithm: Model algorithm name
            limit: Maximum number of versions to return
            
        Returns:
            List of dictionaries with model version information
        """
        metrics_path = self._get_metrics_path(domain, dataset)
        
        # List all metrics files for this algorithm
        metrics_files = [f for f in os.listdir(metrics_path) 
                       if f.startswith(f"{algorithm}_") and f.endswith("_metrics.json")]
        
        if not metrics_files:
            return []
            
        # Load metrics for each version
        history = []
        for metrics_file in metrics_files:
            try:
                with open(os.path.join(metrics_path, metrics_file), 'r') as f:
                    metrics = json.load(f)
                history.append(metrics)
            except Exception as e:
                logger.error(f"Error loading metrics from {metrics_file}: {str(e)}")
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        # Limit the number of results
        return history[:limit]
    
    def detect_distribution_shift(self, domain: str, dataset: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if there has been a significant distribution shift in the dataset.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            
        Returns:
            Tuple of (should_retrain, report)
        """
        # In detect_distribution_shift method
        stats_path = os.path.join(self.stats_dir, domain, dataset)

        if not os.path.exists(stats_path):
            os.makedirs(stats_path, exist_ok=True)
            default_stats = {
                "timestamp": int(time.time()),
                "min_rating": 1.0,
                "max_rating": 5.0,
                "avg_rating": 3.0,
                "rating_distribution": [0.2, 0.2, 0.2, 0.2, 0.2],  # default uniform
                "total_users": 0,
                "total_items": 0,
                "total_ratings": 0,
                "avg_ratings_per_user": 0,
                "avg_ratings_per_item": 0,
                "sparsity": 0.0
            }
            
            stats_file = os.path.join(stats_path, "dataset_stats.json")
            try:
                with open(stats_file, 'w') as f:
                    json.dump(default_stats, f, indent=2)
                logger.info(f"Created default stats file for {domain}/{dataset}")
            except Exception as e:
                logger.error(f"Error creating default stats file: {str(e)}")
                
            return True, {"status": "no_stats", "message": "Created default stats file. Retraining recommended."}
            
        # Load current stats
        try:
            with open(stats_file, 'r') as f:
                current_stats = json.load(f)
        except Exception as e:
            return False, {"status": "error", "message": f"Error loading stats: {str(e)}"}
            
        # Look for previous training stats
        training_stats_file = os.path.join(stats_path, "last_training_stats.json")
        if not os.path.exists(training_stats_file):
            # No previous training stats, just use current stats as baseline
            try:
                with open(training_stats_file, 'w') as f:
                    json.dump(current_stats, f, indent=2)
                return True, {"status": "no_baseline", "message": "No previous training stats found. Retraining recommended."}
            except Exception as e:
                return False, {"status": "error", "message": f"Error saving training stats: {str(e)}"}
        
        # Load previous training stats
        try:
            with open(training_stats_file, 'r') as f:
                training_stats = json.load(f)
        except Exception as e:
            return False, {"status": "error", "message": f"Error loading training stats: {str(e)}"}
        
        # Initialize report
        report = {
            "timestamp": int(time.time()),
            "domain": domain,
            "dataset": dataset,
            "shifts_detected": {},
            "should_retrain": False,
            "reasons": []
        }
        
        # Check for data volume change
        current_ratings = current_stats.get("total_ratings", 0)
        training_ratings = training_stats.get("total_ratings", 0)
        ratings_increase = current_ratings - training_ratings
        ratings_increase_pct = (ratings_increase / training_ratings * 100) if training_ratings > 0 else 100
        
        report["shifts_detected"]["data_volume"] = {
            "previous": training_ratings,
            "current": current_ratings,
            "absolute_change": ratings_increase,
            "percent_change": ratings_increase_pct,
            "significant": ratings_increase_pct > 10  # Consider significant if >10% more data
        }
        
        if ratings_increase_pct > 10:
            report["should_retrain"] = True
            report["reasons"].append(f"Data volume increased by {ratings_increase_pct:.1f}%")
        
        # Check for rating distribution shift
        current_dist = current_stats.get("rating_distribution", [])
        training_dist = training_stats.get("rating_distribution", [])
        
        if current_dist and training_dist and len(current_dist) == len(training_dist):
            dist_shift = np.mean(np.abs(np.array(current_dist) - np.array(training_dist)))
            
            report["shifts_detected"]["rating_distribution"] = {
                "previous": training_dist,
                "current": current_dist,
                "shift_magnitude": float(dist_shift),
                "significant": dist_shift > 0.05  # Consider significant if distribution changed by >5%
            }
            
            if dist_shift > 0.05:
                report["should_retrain"] = True
                report["reasons"].append(f"Rating distribution shifted by {dist_shift:.1%}")
        
        # Check for user base shift
        current_users = current_stats.get("total_users", 0)
        training_users = training_stats.get("total_users", 0)
        users_increase = current_users - training_users
        users_increase_pct = (users_increase / training_users * 100) if training_users > 0 else 100
        
        report["shifts_detected"]["user_base"] = {
            "previous": training_users,
            "current": current_users,
            "absolute_change": users_increase,
            "percent_change": users_increase_pct,
            "significant": users_increase_pct > 15  # Consider significant if >15% more users
        }
        
        if users_increase_pct > 15:
            report["should_retrain"] = True
            report["reasons"].append(f"User base increased by {users_increase_pct:.1f}%")
        
        # Check for item catalog shift
        current_items = current_stats.get("total_items", 0)
        training_items = training_stats.get("total_items", 0)
        items_increase = current_items - training_items
        items_increase_pct = (items_increase / training_items * 100) if training_items > 0 else 100
        
        report["shifts_detected"]["item_catalog"] = {
            "previous": training_items,
            "current": current_items,
            "absolute_change": items_increase,
            "percent_change": items_increase_pct,
            "significant": items_increase_pct > 15  # Consider significant if >15% more items
        }
        
        if items_increase_pct > 15:
            report["should_retrain"] = True
            report["reasons"].append(f"Item catalog increased by {items_increase_pct:.1f}%")
        
        # Check time since last training
        current_time = int(time.time())
        last_training_time = training_stats.get("timestamp", 0)
        time_diff_days = (current_time - last_training_time) / (60 * 60 * 24)  # Convert to days
        
        report["shifts_detected"]["time_elapsed"] = {
            "last_training": last_training_time,
            "current_time": current_time,
            "days_elapsed": time_diff_days,
            "significant": time_diff_days > 30  # Consider significant if >30 days
        }
        
        if time_diff_days > 30:
            report["should_retrain"] = True
            report["reasons"].append(f"Time since last training: {time_diff_days:.1f} days")
        
        return report["should_retrain"], report
    
    def setup_ab_test(self, domain: str, dataset: str, 
                     model_a: Dict[str, str], model_b: Dict[str, str], 
                     traffic_split: float = 0.5,
                     test_duration_days: int = 7) -> Dict[str, Any]:
        """
        Set up an A/B test between two model versions.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            model_a: Dictionary with 'algorithm' and 'version' for model A
            model_b: Dictionary with 'algorithm' and 'version' for model B
            traffic_split: Percentage of traffic to send to model B (0-1)
            test_duration_days: Duration of the test in days
            
        Returns:
            Dictionary with A/B test configuration
        """
        # Validate models exist
        model_a_path = self.get_model_file_path(
            domain, dataset, model_a['algorithm'], model_a.get('version'))
        model_b_path = self.get_model_file_path(
            domain, dataset, model_b['algorithm'], model_b.get('version'))
        
        if model_a_path is None:
            raise ValueError(f"Model A ({model_a['algorithm']} {model_a.get('version', 'latest')}) not found")
        
        if model_b_path is None:
            raise ValueError(f"Model B ({model_b['algorithm']} {model_b.get('version', 'latest')}) not found")
        
        # Get metrics for both models
        model_a_metrics = self.get_model_metrics(
            domain, dataset, model_a['algorithm'], model_a.get('version'))
        model_b_metrics = self.get_model_metrics(
            domain, dataset, model_b['algorithm'], model_b.get('version'))
        
        # Create AB test configuration
        test_id = f"abtest_{int(time.time())}"
        end_time = int(time.time() + (test_duration_days * 24 * 60 * 60))
        
        ab_test = {
            "test_id": test_id,
            "domain": domain,
            "dataset": dataset,
            "start_time": int(time.time()),
            "end_time": end_time,
            "status": "active",
            "traffic_split": traffic_split,
            "model_a": {
                "algorithm": model_a['algorithm'],
                "version": model_a.get('version', self.get_latest_model_version(domain, dataset, model_a['algorithm'])),
                "path": model_a_path,
                "baseline_metrics": model_a_metrics
            },
            "model_b": {
                "algorithm": model_b['algorithm'],
                "version": model_b.get('version', self.get_latest_model_version(domain, dataset, model_b['algorithm'])),
                "path": model_b_path,
                "baseline_metrics": model_b_metrics
            },
            "results": {
                "impressions_a": 0,
                "conversions_a": 0,
                "impressions_b": 0,
                "conversions_b": 0,
                "metrics_a": {},
                "metrics_b": {}
            }
        }
        
        # Save AB test configuration
        ab_test_path = self._get_ab_test_path(domain, dataset)
        ab_test_file = os.path.join(ab_test_path, f"{test_id}.json")
        
        try:
            with open(ab_test_file, 'w') as f:
                json.dump(ab_test, f, indent=2)
            logger.info(f"Created A/B test {test_id} between {model_a['algorithm']} and {model_b['algorithm']}")
            return ab_test
        except Exception as e:
            logger.error(f"Error creating A/B test: {str(e)}")
            raise
    
    def get_active_ab_tests(self, domain: str, dataset: str) -> List[Dict[str, Any]]:
        """
        Get all active A/B tests for a domain/dataset.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            
        Returns:
            List of active A/B test configurations
        """
        ab_test_path = self._get_ab_test_path(domain, dataset)
        active_tests = []
        
        for filename in os.listdir(ab_test_path):
            if filename.startswith("abtest_") and filename.endswith(".json"):
                try:
                    with open(os.path.join(ab_test_path, filename), 'r') as f:
                        test = json.load(f)
                    
                    # Check if test is still active
                    current_time = int(time.time())
                    if test.get("status") == "active" and test.get("end_time", 0) > current_time:
                        active_tests.append(test)
                    
                except Exception as e:
                    logger.error(f"Error loading A/B test {filename}: {str(e)}")
        
        return active_tests
    
    def update_ab_test_results(self, domain: str, dataset: str, test_id: str, 
                              group: str, impression: bool = False, 
                              conversion: bool = False, metrics: Optional[Dict[str, float]] = None) -> bool:
        """
        Update results for an A/B test.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            test_id: A/B test ID
            group: 'a' or 'b' indicating which model group
            impression: Whether to increment impression count
            conversion: Whether to increment conversion count
            metrics: Optional additional metrics to track
            
        Returns:
            True if successful, False otherwise
        """
        ab_test_path = self._get_ab_test_path(domain, dataset)
        ab_test_file = os.path.join(ab_test_path, f"{test_id}.json")
        
        if not os.path.exists(ab_test_file):
            logger.error(f"A/B test {test_id} not found")
            return False
        
        try:
            # Load test
            with open(ab_test_file, 'r') as f:
                test = json.load(f)
            
            # Check if test is still active
            current_time = int(time.time())
            if test.get("status") != "active" or test.get("end_time", 0) <= current_time:
                logger.warning(f"A/B test {test_id} is no longer active")
                return False
            
            # Update results based on group
            group_key = group.lower()
            if group_key not in ['a', 'b']:
                logger.error(f"Invalid group '{group}'. Must be 'a' or 'b'.")
                return False
            
            if impression:
                test["results"][f"impressions_{group_key}"] += 1
                
            if conversion:
                test["results"][f"conversions_{group_key}"] += 1
            
            # Update additional metrics
            if metrics and isinstance(metrics, dict):
                for key, value in metrics.items():
                    # Initialize metric if not exists
                    if key not in test["results"][f"metrics_{group_key}"]:
                        test["results"][f"metrics_{group_key}"][key] = []
                    
                    # Add new value
                    test["results"][f"metrics_{group_key}"][key].append(value)
            
            # Save updated test
            with open(ab_test_file, 'w') as f:
                json.dump(test, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating A/B test results: {str(e)}")
            return False
    
    def conclude_ab_test(self, domain: str, dataset: str, test_id: str) -> Dict[str, Any]:
        """
        Conclude an A/B test and determine the winning model.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            test_id: A/B test ID
            
        Returns:
            Dictionary with test results and winner information
        """
        ab_test_path = self._get_ab_test_path(domain, dataset)
        ab_test_file = os.path.join(ab_test_path, f"{test_id}.json")
        
        if not os.path.exists(ab_test_file):
            raise ValueError(f"A/B test {test_id} not found")
        
        try:
            # Load test
            with open(ab_test_file, 'r') as f:
                test = json.load(f)
            
            # Check if test is already concluded
            if test.get("status") == "concluded":
                return test.get("conclusion", {})
            
            # Calculate basic metrics
            results = test["results"]
            impressions_a = results.get("impressions_a", 0)
            conversions_a = results.get("conversions_a", 0)
            impressions_b = results.get("impressions_b", 0)
            conversions_b = results.get("conversions_b", 0)
            
            ctr_a = conversions_a / impressions_a if impressions_a > 0 else 0
            ctr_b = conversions_b / impressions_b if impressions_b > 0 else 0
            
            # Calculate improvement
            relative_improvement = (ctr_b - ctr_a) / ctr_a if ctr_a > 0 else float('inf')
            
            # Determine winner
            if ctr_b > ctr_a and relative_improvement > 0.05:  # B is at least 5% better
                winner = "b"
                reason = f"Model B had {relative_improvement:.1%} higher conversion rate"
            elif ctr_a > ctr_b and (ctr_a - ctr_b) / ctr_b > 0.05:  # A is at least 5% better
                winner = "a"
                reason = f"Model A had {((ctr_a - ctr_b) / ctr_b):.1%} higher conversion rate"
            else:
                winner = "tie"
                reason = "No significant difference between models"
            
            # Create conclusion
            conclusion = {
                "timestamp": int(time.time()),
                "metrics": {
                    "model_a": {
                        "impressions": impressions_a,
                        "conversions": conversions_a,
                        "conversion_rate": ctr_a
                    },
                    "model_b": {
                        "impressions": impressions_b,
                        "conversions": conversions_b,
                        "conversion_rate": ctr_b
                    }
                },
                "relative_improvement": relative_improvement,
                "winner": winner,
                "reason": reason,
                "winning_model": test[f"model_{winner}"] if winner != "tie" else None
            }
            
            # Update test with conclusion
            test["status"] = "concluded"
            test["conclusion"] = conclusion
            
            # Save updated test
            with open(ab_test_file, 'w') as f:
                json.dump(test, f, indent=2)
            
            logger.info(f"A/B test {test_id} concluded. Winner: {winner}")
            return conclusion
            
        except Exception as e:
            logger.error(f"Error concluding A/B test: {str(e)}")
            raise
    
    def get_recommended_model(self, domain: str, dataset: str) -> Dict[str, Any]:
        """
        Get the recommended model to use for a domain/dataset based on A/B test results.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            
        Returns:
            Dictionary with recommended model information
        """
        # Look for concluded A/B tests with clear winners
        ab_test_path = self._get_ab_test_path(domain, dataset)
        
        if not os.path.exists(ab_test_path):
            return self._get_latest_available_model(domain, dataset)
        
        winning_models = []
        
        for filename in os.listdir(ab_test_path):
            if filename.startswith("abtest_") and filename.endswith(".json"):
                try:
                    with open(os.path.join(ab_test_path, filename), 'r') as f:
                        test = json.load(f)
                    
                    # Check if test is concluded with a winner
                    if (test.get("status") == "concluded" and 
                        test.get("conclusion", {}).get("winner") in ["a", "b"]):
                        
                        winner = test["conclusion"]["winner"]
                        winning_model = test[f"model_{winner}"]
                        winning_model["test_id"] = test["test_id"]
                        winning_model["conclusion_time"] = test["conclusion"]["timestamp"]
                        winning_models.append(winning_model)
                    
                except Exception as e:
                    logger.error(f"Error loading A/B test {filename}: {str(e)}")
        
        if winning_models:
            # Sort by conclusion time (newest first)
            winning_models.sort(key=lambda x: x.get("conclusion_time", 0), reverse=True)
            
            # Return the most recent winner
            recommendation = {
                "domain": domain,
                "dataset": dataset,
                "algorithm": winning_models[0]["algorithm"],
                "version": winning_models[0]["version"],
                "path": winning_models[0]["path"],
                "source": "ab_test",
                "test_id": winning_models[0]["test_id"]
            }
            
            return recommendation
        
        # If no A/B test winners, return the latest available model
        return self._get_latest_available_model(domain, dataset)
    
    def _get_latest_available_model(self, domain: str, dataset: str) -> Dict[str, Any]:
        """
        Get the latest available model for a domain/dataset.
        """
        # Look in the models directory for the domain/dataset
        models_path = os.path.join(self.models_dir, domain, dataset)
        
        if not os.path.exists(models_path):
            return {
                "domain": domain,
                "dataset": dataset,
                "algorithm": None,
                "version": None,
                "path": None,
                "source": "none_available",
                "error": f"No models found for {domain}/{dataset}"
            }
        
        # Check for model files directly in the directory
        model_files = [f for f in os.listdir(models_path) 
                    if f.endswith('.pkl') and not f.endswith('_metadata.json')]
        
        if model_files:
            # Extract algorithm names from filenames
            algorithms = [os.path.splitext(f)[0] for f in model_files]
            logger.info(f"Found algorithms: {algorithms}")
            
            # Get timestamps from metadata files
            latest_models = []
            for algorithm in algorithms:
                metadata_file = os.path.join(models_path, f"{algorithm}_metadata.json")
                timestamp = int(time.time())  # Default timestamp
                
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            timestamp = metadata.get("timestamp", timestamp)
                    except Exception as e:
                        logger.error(f"Error reading metadata for {algorithm}: {str(e)}")
                
                path = os.path.join(models_path, f"{algorithm}.pkl")
                latest_models.append({
                    "algorithm": algorithm,
                    "version": "latest",
                    "path": path,
                    "metrics": {},
                    "timestamp": timestamp
                })
            
            # Find the most recent model
            if latest_models:
                latest_models.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
                latest_model = latest_models[0]
                
                recommendation = {
                    "domain": domain,
                    "dataset": dataset,
                    "algorithm": latest_model["algorithm"],
                    "version": latest_model["version"],
                    "path": latest_model["path"],
                    "source": "latest_available",
                    "timestamp": latest_model.get("timestamp", 0)
                }
                
                return recommendation
        
        # No valid models found
        return {
            "domain": domain,
            "dataset": dataset,
            "algorithm": None,
            "version": None,
            "path": None,
            "source": "none_available",
            "error": f"No valid models found for {domain}/{dataset}"
        }
    
    def update_training_stats(self, domain: str, dataset: str) -> bool:
        """
        Update the last training stats with the current dataset stats.
        This should be called after model training to mark current state as the baseline.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            
        Returns:
            True if successful, False otherwise
        """
        stats_path = os.path.join(self.stats_dir, domain, dataset)
        
        if not os.path.exists(stats_path):
            os.makedirs(stats_path, exist_ok=True)
            logger.info(f"Created stats directory for {domain}/{dataset}")
        
        # Get the current stats file
        stats_file = os.path.join(stats_path, "dataset_stats.json")
        if not os.path.exists(stats_file):
            # Create a default stats file
            default_stats = {
                "timestamp": int(time.time()),
                "min_rating": 1.0,
                "max_rating": 5.0,
                "avg_rating": 3.0,
                "rating_distribution": [0.2, 0.2, 0.2, 0.2, 0.2],  # default uniform
                "total_users": 0,
                "total_items": 0,
                "total_ratings": 0,
                "avg_ratings_per_user": 0,
                "avg_ratings_per_item": 0,
                "sparsity": 0.0
            }
            
            try:
                with open(stats_file, 'w') as f:
                    json.dump(default_stats, f, indent=2)
                logger.info(f"Created default stats file for {domain}/{dataset}")
                
                # Save as training stats and return
                training_stats_file = os.path.join(stats_path, "last_training_stats.json")
                with open(training_stats_file, 'w') as f:
                    json.dump(default_stats, f, indent=2)
                    
                logger.info(f"Created default training stats for {domain}/{dataset}")
                return True
            except Exception as e:
                logger.error(f"Error creating default stats file: {str(e)}")
                return False
    
    def should_retrain(self, domain: str, dataset: str) -> Dict[str, Any]:
        """
        Check if a model should be retrained based on various factors.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            
        Returns:
            Dictionary with retraining recommendation
        """
        # First check for distribution shifts
        should_retrain, shift_report = self.detect_distribution_shift(domain, dataset)
        
        # Create recommendation
        recommendation = {
            "timestamp": int(time.time()),
            "domain": domain,
            "dataset": dataset,
            "should_retrain": should_retrain,
            "reasons": shift_report.get("reasons", []),
            "distribution_shift_report": shift_report
        }
        
        # Add current model info
        current_model = self.get_recommended_model(domain, dataset)
        recommendation["current_model"] = current_model
        
        # Check if we have a recommended model
        if current_model.get("algorithm") is None:
            recommendation["should_retrain"] = True
            recommendation["reasons"].append("No available model found")
            
        # Check for active A/B tests
        active_tests = self.get_active_ab_tests(domain, dataset)
        recommendation["active_ab_tests"] = len(active_tests)
        
        # If there are active A/B tests, we might want to wait before retraining
        if active_tests and recommendation["should_retrain"]:
            recommendation["notes"] = ["There are active A/B tests. Consider waiting until they conclude."]
        
        return recommendation
    
    def register_trained_model(self, domain: str, dataset: str, algorithm: str, 
                              model_data: bytes, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a newly trained model.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            algorithm: Algorithm name
            model_data: Binary model data
            metrics: Dictionary of model metrics
            
        Returns:
            Dictionary with model registration information
        """
        # Generate version based on timestamp
        version = str(int(time.time()))
        
        # Ensure directories exist
        model_path = self._get_domain_dataset_model_path(domain, dataset, algorithm)
        
        # Create model file path
        model_file = os.path.join(model_path, f"{algorithm}_{version}.model")
        
        try:
            # Save model data
            with open(model_file, 'wb') as f:
                f.write(model_data)
                
            # Save metrics
            self.save_model_metrics(domain, dataset, algorithm, version, metrics)
            
            # Update training stats
            self.update_training_stats(domain, dataset)
            
            # Return registration info
            registration = {
                "domain": domain,
                "dataset": dataset,
                "algorithm": algorithm,
                "version": version,
                "path": model_file,
                "metrics": metrics,
                "timestamp": int(time.time())
            }
            
            logger.info(f"Registered new {algorithm} model version {version} for {domain}/{dataset}")
            return registration
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise
    
    def rollback_to_version(self, domain: str, dataset: str, algorithm: str, 
                          target_version: str) -> Dict[str, Any]:
        """
        Rollback to a previous model version by creating a new version
        that is a copy of the target version.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            algorithm: Algorithm name
            target_version: Target version to rollback to
            
        Returns:
            Dictionary with rollback information
        """
        # Validate target version exists
        target_path = self.get_model_file_path(domain, dataset, algorithm, target_version)
        if not target_path or not os.path.exists(target_path):
            raise ValueError(f"Target version {target_version} not found")
            
        # Get metrics for target version
        target_metrics = self.get_model_metrics(domain, dataset, algorithm, target_version)
        if not target_metrics:
            logger.warning(f"No metrics found for target version {target_version}")
            target_metrics = {}
            
        # Create a new version with current timestamp
        new_version = str(int(time.time()))
        model_path = self._get_domain_dataset_model_path(domain, dataset, algorithm)
        new_path = os.path.join(model_path, f"{algorithm}_{new_version}.model")
        
        try:
            # Copy the target model to the new version
            shutil.copy2(target_path, new_path)
            
            # Copy metrics and add rollback info
            rollback_metrics = target_metrics.copy()
            rollback_metrics["rollback_from"] = target_version
            rollback_metrics["rollback_timestamp"] = int(time.time())
            rollback_metrics["is_rollback"] = True
            
            # Save the metrics
            self.save_model_metrics(domain, dataset, algorithm, new_version, rollback_metrics)
            
            # Return rollback info
            rollback_info = {
                "domain": domain,
                "dataset": dataset,
                "algorithm": algorithm,
                "original_version": target_version,
                "new_version": new_version,
                "path": new_path,
                "timestamp": int(time.time())
            }
            
            logger.info(f"Rolled back to {algorithm} version {target_version} by creating new version {new_version}")
            return rollback_info
            
        except Exception as e:
            logger.error(f"Error rolling back to version {target_version}: {str(e)}")
            raise
