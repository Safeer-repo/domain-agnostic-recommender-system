import os
import pandas as pd
import numpy as np
import logging
import time
import json
from typing import Dict, Optional, List, Tuple, Any, Union
import shutil

logger = logging.getLogger(__name__)

class DatasetUpdater:
    """
    Handles the process of validating and incorporating new ratings
    into existing training datasets.
    """
    
    def __init__(self, data_dir: str, ratings_storage=None):
        """
        Initialize the dataset updater.
        
        Args:
            data_dir: Root directory for data storage
            ratings_storage: Optional RatingsStorage instance for accessing new ratings
        """
        self.data_dir = data_dir
        self.datasets_dir = os.path.join(data_dir, "datasets")
        self.staging_dir = os.path.join(data_dir, "staging")
        self.stats_dir = os.path.join(data_dir, "dataset_stats")
        self.ratings_storage = ratings_storage
        self._ensure_directory_structure()
        
    def _ensure_directory_structure(self) -> None:
        """Create necessary directory structure if it doesn't exist."""
        for directory in [self.datasets_dir, self.staging_dir, self.stats_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory at {directory}")
    
    def _get_dataset_path(self, domain: str, dataset: str) -> str:
        """
        Get the path to a domain/dataset directory.
        
        Args:
            domain: Domain name (e.g., entertainment, ecommerce)
            dataset: Dataset name (e.g., movielens, amazon_electronics)
            
        Returns:
            Path to the domain/dataset directory
        """
        return os.path.join(self.datasets_dir, domain, dataset)
    
    def _get_staging_path(self, domain: str, dataset: str) -> str:
        """
        Get the path to a domain/dataset staging directory.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            
        Returns:
            Path to the domain/dataset staging directory
        """
        return os.path.join(self.staging_dir, domain, dataset)
    
    def _get_stats_path(self, domain: str, dataset: str) -> str:
        """
        Get the path to a domain/dataset stats directory.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            
        Returns:
            Path to the domain/dataset stats directory
        """
        return os.path.join(self.stats_dir, domain, dataset)
    
    def _ensure_domain_dataset_dirs(self, domain: str, dataset: str) -> Tuple[str, str, str]:
        """
        Ensure the main, staging, and stats directories exist for a domain/dataset.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            
        Returns:
            Tuple of (dataset_path, staging_path, stats_path)
        """
        dataset_path = self._get_dataset_path(domain, dataset)
        staging_path = self._get_staging_path(domain, dataset)
        stats_path = self._get_stats_path(domain, dataset)
        
        for path in [dataset_path, staging_path, stats_path]:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                logger.debug(f"Created directory at {path}")
                
        return dataset_path, staging_path, stats_path
    
    def stage_new_ratings(self, domain: str, dataset: str, since_timestamp: Optional[int] = None) -> pd.DataFrame:
        """
        Pull new ratings from the RatingsStorage and stage them for validation.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            since_timestamp: Optional timestamp to filter ratings
            
        Returns:
            DataFrame containing the staged ratings
        """
        if self.ratings_storage is None:
            raise ValueError("RatingsStorage not provided. Cannot retrieve new ratings.")
            
        # Get new ratings from storage
        new_ratings = self.ratings_storage.get_new_ratings(domain, dataset, since_timestamp)
        
        if new_ratings.empty:
            logger.info(f"No new ratings found for {domain}/{dataset}")
            return pd.DataFrame()
            
        logger.info(f"Retrieved {len(new_ratings)} new ratings for {domain}/{dataset}")
        
        # Ensure directories exist
        _, staging_path, _ = self._ensure_domain_dataset_dirs(domain, dataset)
        
        # Create staging file with timestamp
        timestamp = int(time.time())
        staging_file = os.path.join(staging_path, f"staged_ratings_{timestamp}.csv")
        
        # Save to staging
        new_ratings.to_csv(staging_file, index=False)
        logger.info(f"Staged {len(new_ratings)} ratings to {staging_file}")
        
        return new_ratings
    
    def validate_staged_ratings(self, domain: str, dataset: str, 
                              staged_file: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate staged ratings by checking for anomalies.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            staged_file: Optional path to specific staged file (uses most recent if None)
            
        Returns:
            Tuple of (validated_ratings, validation_report)
        """
        # Ensure directories exist
        _, staging_path, stats_path = self._ensure_domain_dataset_dirs(domain, dataset)
        
        # Find the staged file to validate
        if staged_file is None:
            # Find most recent staging file
            staging_files = [f for f in os.listdir(staging_path) if f.startswith("staged_ratings_")]
            if not staging_files:
                logger.warning(f"No staged ratings found for {domain}/{dataset}")
                return pd.DataFrame(), {"status": "no_data"}
                
            # Sort by timestamp (newest first)
            staging_files.sort(reverse=True)
            staged_file = os.path.join(staging_path, staging_files[0])
        
        # Load staged ratings
        try:
            staged_ratings = pd.read_csv(staged_file)
            if staged_ratings.empty:
                logger.warning(f"Staged file {staged_file} is empty")
                return pd.DataFrame(), {"status": "empty_file"}
        except Exception as e:
            logger.error(f"Error loading staged file {staged_file}: {str(e)}")
            return pd.DataFrame(), {"status": "error", "message": str(e)}
        
        # Get or compute dataset statistics for comparison
        stats_file = os.path.join(stats_path, "dataset_stats.json")
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    dataset_stats = json.load(f)
            except Exception as e:
                logger.error(f"Error loading stats file {stats_file}: {str(e)}")
                dataset_stats = self._compute_dataset_stats(domain, dataset)
        else:
            dataset_stats = self._compute_dataset_stats(domain, dataset)
        
        # Perform validation
        validation_report = self._validate_ratings_quality(staged_ratings, dataset_stats)
        
        # Apply filtering based on validation
        validated_ratings = self._filter_invalid_ratings(staged_ratings, validation_report)
        
        # Save validation report
        report_file = os.path.join(stats_path, f"validation_report_{os.path.basename(staged_file).replace('staged_ratings_', '')}")
        try:
            with open(report_file, 'w') as f:
                json.dump(validation_report, f, indent=2)
            logger.info(f"Saved validation report to {report_file}")
        except Exception as e:
            logger.error(f"Error saving validation report: {str(e)}")
        
        # Count filtered ratings
        filtered_count = len(staged_ratings) - len(validated_ratings)
        if filtered_count > 0:
            logger.warning(f"Filtered out {filtered_count} invalid ratings ({filtered_count/len(staged_ratings):.1%})")
        
        return validated_ratings, validation_report
    
    def _validate_ratings_quality(self, ratings: pd.DataFrame, dataset_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate ratings quality by checking for outliers and anomalies.
        
        Args:
            ratings: DataFrame of ratings to validate
            dataset_stats: Statistics from the existing dataset
            
        Returns:
            Validation report dictionary
        """
        report = {
            "timestamp": int(time.time()),
            "total_ratings": len(ratings),
            "validation_checks": {},
            "status": "valid"  # default status, will be updated if issues found
        }
        
        # 1. Check rating values within allowed range
        min_rating = dataset_stats.get("min_rating", 1.0)
        max_rating = dataset_stats.get("max_rating", 5.0)
        
        invalid_ratings_mask = (ratings['rating'] < min_rating) | (ratings['rating'] > max_rating)
        report["validation_checks"]["invalid_rating_values"] = {
            "count": int(invalid_ratings_mask.sum()),
            "percent": float(invalid_ratings_mask.mean() * 100),
            "min_allowed": float(min_rating),
            "max_allowed": float(max_rating)
        }
        
        # 2. Check for rating distribution anomalies
        hist_current = np.histogram(ratings['rating'], bins=5, range=(min_rating, max_rating))[0]
        hist_current = hist_current / hist_current.sum() if hist_current.sum() > 0 else hist_current
        
        hist_expected = np.array(dataset_stats.get("rating_distribution", [0.2, 0.2, 0.2, 0.2, 0.2]))
        
        # Calculate divergence between distributions
        distribution_diff = np.abs(hist_current - hist_expected).mean()
        
        report["validation_checks"]["rating_distribution"] = {
            "current_distribution": [float(x) for x in hist_current],
            "expected_distribution": [float(x) for x in hist_expected],
            "distribution_difference": float(distribution_diff),
            "is_anomalous": bool(distribution_diff > 0.3)  # Convert numpy bool to Python bool
        }
        
        # 3. Check for user outliers (e.g., bot users submitting many ratings)
        user_counts = ratings['user_id'].value_counts()
        max_expected = dataset_stats.get("avg_ratings_per_user", 10) * 5  # allow 5x average
        
        outlier_users = user_counts[user_counts > max_expected].to_dict()
        # Convert keys to strings for JSON serialization
        outlier_users = {str(k): int(v) for k, v in outlier_users.items()}
        
        report["validation_checks"]["user_outliers"] = {
            "count": len(outlier_users),
            "outlier_users": outlier_users,
            "max_expected_per_user": float(max_expected)
        }
        
        # 4. Check for item outliers (unusual spikes in ratings for specific items)
        item_counts = ratings['item_id'].value_counts()
        max_expected_item = dataset_stats.get("avg_ratings_per_item", 10) * 5  # allow 5x average
        
        outlier_items = item_counts[item_counts > max_expected_item].to_dict()
        # Convert keys to strings for JSON serialization
        outlier_items = {str(k): int(v) for k, v in outlier_items.items()}
        
        report["validation_checks"]["item_outliers"] = {
            "count": len(outlier_items),
            "outlier_items": outlier_items,
            "max_expected_per_item": float(max_expected_item)
        }
        
        # Determine overall status
        if (report["validation_checks"]["invalid_rating_values"]["percent"] > 10 or
            report["validation_checks"]["rating_distribution"]["is_anomalous"] or
            report["validation_checks"]["user_outliers"]["count"] > len(ratings['user_id'].unique()) * 0.1):
            report["status"] = "warning"
            
        if report["validation_checks"]["invalid_rating_values"]["percent"] > 30:
            report["status"] = "critical"
            
        return report
    
    def _filter_invalid_ratings(self, ratings: pd.DataFrame, validation_report: Dict[str, Any]) -> pd.DataFrame:
        """
        Filter out invalid ratings based on validation report.
        
        Args:
            ratings: DataFrame of ratings to filter
            validation_report: Validation report from _validate_ratings_quality
            
        Returns:
            Filtered DataFrame with only valid ratings
        """
        # Create a copy to avoid modifying the original
        valid_ratings = ratings.copy()
        
        # 1. Filter out ratings outside the allowed range
        min_rating = validation_report["validation_checks"]["invalid_rating_values"]["min_allowed"]
        max_rating = validation_report["validation_checks"]["invalid_rating_values"]["max_allowed"]
        
        valid_ratings = valid_ratings[(valid_ratings['rating'] >= min_rating) & 
                                      (valid_ratings['rating'] <= max_rating)]
        
        # 2. Filter out ratings from outlier users if critical
        if validation_report["status"] == "critical":
            outlier_users = validation_report["validation_checks"]["user_outliers"]["outlier_users"]
            if outlier_users:
                valid_ratings = valid_ratings[~valid_ratings['user_id'].isin(outlier_users.keys())]
        
        return valid_ratings
    
    def _compute_dataset_stats(self, domain: str, dataset: str) -> Dict[str, Any]:
        """
        Compute statistics from the existing dataset for a domain/dataset.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            
        Returns:
            Dictionary of dataset statistics
        """
        dataset_path = self._get_dataset_path(domain, dataset)
        stats = {
            "timestamp": int(time.time()),
            "min_rating": 1.0,
            "max_rating": 5.0,
            "avg_rating": 0.0,
            "rating_distribution": [0.2, 0.2, 0.2, 0.2, 0.2],  # default uniform
            "total_users": 0,
            "total_items": 0,
            "total_ratings": 0,
            "avg_ratings_per_user": 0,
            "avg_ratings_per_item": 0,
            "sparsity": 0.0
        }
        
        # Look for main ratings file
        ratings_file = os.path.join(dataset_path, "ratings.csv")
        if not os.path.exists(ratings_file):
            logger.warning(f"No existing ratings file found for {domain}/{dataset}")
            return stats
        
        try:
            # Load existing ratings
            existing_ratings = pd.read_csv(ratings_file)
            
            # Compute statistics
            stats["total_ratings"] = len(existing_ratings)
            stats["total_users"] = existing_ratings['user_id'].nunique()
            stats["total_items"] = existing_ratings['item_id'].nunique()
            
            if stats["total_ratings"] > 0:
                stats["min_rating"] = float(existing_ratings['rating'].min())
                stats["max_rating"] = float(existing_ratings['rating'].max())
                stats["avg_rating"] = float(existing_ratings['rating'].mean())
                
                # Compute rating distribution
                hist = np.histogram(existing_ratings['rating'], 
                                     bins=5, 
                                     range=(stats["min_rating"], stats["max_rating"]))[0]
                stats["rating_distribution"] = [float(x) for x in (hist / hist.sum()).tolist()]
                
                # Compute average ratings per user and item
                if stats["total_users"] > 0:
                    stats["avg_ratings_per_user"] = float(stats["total_ratings"] / stats["total_users"])
                
                if stats["total_items"] > 0:
                    stats["avg_ratings_per_item"] = float(stats["total_ratings"] / stats["total_items"])
                
                # Compute sparsity
                potential_ratings = stats["total_users"] * stats["total_items"]
                if potential_ratings > 0:
                    stats["sparsity"] = float(1.0 - (stats["total_ratings"] / potential_ratings))
            
            # Save the stats to file
            stats_path = self._get_stats_path(domain, dataset)
            stats_file = os.path.join(stats_path, "dataset_stats.json")
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Computed and saved dataset stats for {domain}/{dataset}")
            
        except Exception as e:
            logger.error(f"Error computing dataset stats: {str(e)}")
        
        return stats
    
    def merge_validated_ratings(self, domain: str, dataset: str, 
                               validated_ratings: pd.DataFrame,
                               incremental: bool = True) -> bool:
        """
        Merge validated ratings into the main dataset.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            validated_ratings: DataFrame of validated ratings to merge
            incremental: Whether to use incremental feature engineering
            
        Returns:
            True if successful, False otherwise
        """
        if validated_ratings.empty:
            logger.warning(f"No validated ratings to merge for {domain}/{dataset}")
            return False
        
        # Ensure directories exist
        dataset_path, _, stats_path = self._ensure_domain_dataset_dirs(domain, dataset)
        
        # Determine the main ratings file
        ratings_file = os.path.join(dataset_path, "ratings.csv")
        
        try:
            # Check if main file exists
            if os.path.exists(ratings_file):
                # Backup existing file
                backup_file = os.path.join(dataset_path, f"ratings_backup_{int(time.time())}.csv")
                shutil.copy2(ratings_file, backup_file)
                logger.info(f"Created backup of ratings at {backup_file}")
                
                # Load existing ratings
                existing_ratings = pd.read_csv(ratings_file)
                
                # Merge with validated ratings
                if incremental:
                    # Incremental approach - just append new ratings
                    # First, deduplicate based on user_id, item_id (keep newest)
                    combined = pd.concat([existing_ratings, validated_ratings])
                    combined = combined.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
                else:
                    # Full reprocessing approach
                    # This could include more complex feature engineering steps
                    combined = pd.concat([existing_ratings, validated_ratings])
                    combined = combined.drop_duplicates(subset=['user_id', 'item_id'], keep='last')
                    # Additional feature engineering would go here if needed
                
                # Save the merged dataset
                combined.to_csv(ratings_file, index=False)
                
                # Log the merge results
                new_records = len(combined) - len(existing_ratings)
                logger.info(f"Merged {len(validated_ratings)} validated ratings into dataset (net new: {new_records})")
                
            else:
                # No existing file, just save the validated ratings
                validated_ratings.to_csv(ratings_file, index=False)
                logger.info(f"Created new ratings file with {len(validated_ratings)} ratings")
            
            # Update dataset statistics
            self._compute_dataset_stats(domain, dataset)
            
            return True
            
        except Exception as e:
            logger.error(f"Error merging validated ratings: {str(e)}")
            return False
    
    def process_new_ratings(self, domain: str, dataset: str, 
                           since_timestamp: Optional[int] = None,
                           incremental: bool = True) -> Dict[str, Any]:
        """
        Complete pipeline to process new ratings: stage, validate, and merge.
        
        Args:
            domain: Domain name
            dataset: Dataset name
            since_timestamp: Optional timestamp to filter ratings
            incremental: Whether to use incremental feature engineering
            
        Returns:
            Processing report dictionary
        """
        report = {
            "timestamp": int(time.time()),
            "domain": domain,
            "dataset": dataset,
            "stages": {}
        }
        
        # Stage new ratings
        try:
            new_ratings = self.stage_new_ratings(domain, dataset, since_timestamp)
            report["stages"]["staging"] = {
                "status": "success" if not new_ratings.empty else "no_data",
                "ratings_count": len(new_ratings)
            }
            
            if new_ratings.empty:
                report["overall_status"] = "no_data"
                return report
                
        except Exception as e:
            logger.error(f"Error staging new ratings: {str(e)}")
            report["stages"]["staging"] = {
                "status": "error",
                "message": str(e)
            }
            report["overall_status"] = "error"
            return report
        
        # Validate staged ratings
        try:
            validated_ratings, validation_report = self.validate_staged_ratings(domain, dataset)
            report["stages"]["validation"] = {
                "status": validation_report["status"],
                "validated_count": len(validated_ratings),
                "filtered_count": len(new_ratings) - len(validated_ratings),
                "validation_details": validation_report["validation_checks"]
            }
            
            if validated_ratings.empty:
                report["overall_status"] = "validation_failed"
                return report
                
        except Exception as e:
            logger.error(f"Error validating staged ratings: {str(e)}")
            report["stages"]["validation"] = {
                "status": "error",
                "message": str(e)
            }
            report["overall_status"] = "error"
            return report
        
        # Merge validated ratings
        try:
            merge_success = self.merge_validated_ratings(domain, dataset, validated_ratings, incremental)
            report["stages"]["merge"] = {
                "status": "success" if merge_success else "error",
                "incremental": incremental,
                "merged_count": len(validated_ratings) if merge_success else 0
            }
            
            if not merge_success:
                report["overall_status"] = "merge_failed"
                return report
                
        except Exception as e:
            logger.error(f"Error merging validated ratings: {str(e)}")
            report["stages"]["merge"] = {
                "status": "error",
                "message": str(e)
            }
            report["overall_status"] = "error"
            return report
        
        # Clear processed ratings from storage if everything succeeded
        if self.ratings_storage is not None:
            try:
                # Use the timestamp from start of processing
                cleared_count = self.ratings_storage.clear_processed_ratings(
                    domain, dataset, report["timestamp"])
                
                report["stages"]["clear_processed"] = {
                    "status": "success",
                    "cleared_count": cleared_count
                }
            except Exception as e:
                logger.error(f"Error clearing processed ratings: {str(e)}")
                report["stages"]["clear_processed"] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Set overall status
        report["overall_status"] = "success"
        
        return report