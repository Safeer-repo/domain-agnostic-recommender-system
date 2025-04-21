#!/usr/bin/env python3
"""
Script to test the ModelRetrainingManager functionality.
Usage: python scripts/test_retraining_manager.py
"""

import os
import sys
import logging
import time
import json
import shutil
import numpy as np
from pprint import pformat

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.model_retraining_manager import ModelRetrainingManager

def clean_test_data():
    """Clean up test data directories to start fresh."""
    for dir_name in ["models", "model_metrics", "dataset_stats", "ab_tests"]:
        dir_path = os.path.join(project_root, dir_name)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            logging.info(f"Removed test directory: {dir_path}")
    
    logging.info("Test environment cleaned")

def create_test_stats():
    """Create test dataset stats to simulate distribution shift detection."""
    # Create directory
    stats_dir = os.path.join(project_root, "dataset_stats", "entertainment", "movielens")
    os.makedirs(stats_dir, exist_ok=True)
    
    # Create current stats
    current_stats = {
        "timestamp": int(time.time()),
        "total_ratings": 1000,
        "total_users": 100,
        "total_items": 200,
        "rating_distribution": [0.1, 0.2, 0.3, 0.3, 0.1],
        "avg_rating": 3.5,
        "min_rating": 1.0,
        "max_rating": 5.0,
        "avg_ratings_per_user": 10.0,
        "avg_ratings_per_item": 5.0,
        "sparsity": 0.95
    }
    
    # Create previous training stats (with 20% less data)
    training_stats = {
        "timestamp": int(time.time()) - (60 * 60 * 24 * 15),  # 15 days ago
        "total_ratings": 800,
        "total_users": 80,
        "total_items": 160,
        "rating_distribution": [0.15, 0.25, 0.3, 0.2, 0.1],
        "avg_rating": 3.2,
        "min_rating": 1.0,
        "max_rating": 5.0,
        "avg_ratings_per_user": 10.0,
        "avg_ratings_per_item": 5.0,
        "sparsity": 0.94
    }
    
    # Save stats files
    with open(os.path.join(stats_dir, "dataset_stats.json"), 'w') as f:
        json.dump(current_stats, f, indent=2)
    
    with open(os.path.join(stats_dir, "last_training_stats.json"), 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    logging.info(f"Created test stats files in {stats_dir}")
    
    return current_stats, training_stats

def create_test_model():
    """Create a test model file and metrics."""
    domain = "entertainment"
    dataset = "movielens"
    algorithm = "als"
    version = str(int(time.time()) - 86400)  # yesterday
    
    # Create model directory
    model_dir = os.path.join(project_root, "models", domain, dataset, algorithm)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create metrics directory
    metrics_dir = os.path.join(project_root, "model_metrics", domain, dataset)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Create dummy model file
    model_file = os.path.join(model_dir, f"{algorithm}_{version}.model")
    with open(model_file, 'wb') as f:
        f.write(b"DUMMY MODEL DATA")
    
    # Create metrics file
    metrics = {
        "algorithm": algorithm,
        "version": version,
        "timestamp": int(version),
        "map_at_k": 0.32,
        "ndcg": 0.28,
        "precision": 0.22,
        "recall": 0.35,
        "training_time": 120,
        "hyperparameters": {
            "factors": 100,
            "regularization": 0.1,
            "iterations": 20
        }
    }
    
    metrics_file = os.path.join(metrics_dir, f"{algorithm}_{version}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logging.info(f"Created test model {algorithm}_{version} for {domain}/{dataset}")
    
    return {
        "domain": domain,
        "dataset": dataset,
        "algorithm": algorithm,
        "version": version,
        "model_file": model_file,
        "metrics_file": metrics_file,
        "metrics": metrics
    }

def test_detect_distribution_shift():
    """Test detecting distribution shifts."""
    # Create test stats
    create_test_stats()
    
    # Create manager
    manager = ModelRetrainingManager(data_dir=project_root)
    
    # Detect distribution shift
    should_retrain, report = manager.detect_distribution_shift(
        domain="entertainment",
        dataset="movielens"
    )
    
    logging.info(f"Should retrain: {should_retrain}")
    logging.info(f"Distribution shift report:")
    logging.info(pformat(report))
    
    # Check results
    if should_retrain:
        logging.info(f"Reasons for retraining:")
        for reason in report.get("reasons", []):
            logging.info(f"- {reason}")
    
    return should_retrain, report

def test_model_management():
    """Test model version management functionality."""
    # Create test model
    model_info = create_test_model()
    
    # Create manager
    manager = ModelRetrainingManager(data_dir=project_root)
    
    # Get latest model version
    latest_version = manager.get_latest_model_version(
        domain=model_info["domain"],
        dataset=model_info["dataset"],
        algorithm=model_info["algorithm"]
    )
    
    logging.info(f"Latest model version: {latest_version}")
    
    # Get model file path
    model_path = manager.get_model_file_path(
        domain=model_info["domain"],
        dataset=model_info["dataset"],
        algorithm=model_info["algorithm"],
        version=latest_version
    )
    
    logging.info(f"Model file path: {model_path}")
    
    # Get model metrics
    metrics = manager.get_model_metrics(
        domain=model_info["domain"],
        dataset=model_info["dataset"],
        algorithm=model_info["algorithm"],
        version=latest_version
    )
    
    logging.info(f"Model metrics:")
    logging.info(pformat(metrics))
    
    # Register a new model
    new_model_data = b"NEW MODEL DATA VERSION 2"
    new_metrics = model_info["metrics"].copy()
    new_metrics["map_at_k"] = 0.35  # Improved performance
    
    registration = manager.register_trained_model(
        domain=model_info["domain"],
        dataset=model_info["dataset"],
        algorithm=model_info["algorithm"],
        model_data=new_model_data,
        metrics=new_metrics
    )
    
    logging.info(f"Registered new model:")
    logging.info(pformat(registration))
    
    # Get model history
    history = manager.get_model_history(
        domain=model_info["domain"],
        dataset=model_info["dataset"],
        algorithm=model_info["algorithm"]
    )
    
    logging.info(f"Model history ({len(history)} entries):")
    for entry in history:
        logging.info(f"- Version: {entry.get('version')}, Performance: {entry.get('map_at_k')}")
    
    # Get recommended model
    recommendation = manager.get_recommended_model(
        domain=model_info["domain"],
        dataset=model_info["dataset"]
    )
    
    logging.info(f"Recommended model:")
    logging.info(pformat(recommendation))
    
    return history, recommendation

def test_ab_testing():
    """Test A/B testing functionality."""
    # Create test models (we need two different algorithms to test)
    model_als = create_test_model()
    
    # Create a second model with different algorithm
    model_ncf = {
        "domain": model_als["domain"],
        "dataset": model_als["dataset"],
        "algorithm": "ncf",
        "version": str(int(time.time()) - 43200)  # 12 hours ago
    }
    
    # Create model directory
    model_dir = os.path.join(project_root, "models", model_ncf["domain"], 
                            model_ncf["dataset"], model_ncf["algorithm"])
    os.makedirs(model_dir, exist_ok=True)
    
    # Create dummy model file
    model_file = os.path.join(model_dir, f"{model_ncf['algorithm']}_{model_ncf['version']}.model")
    with open(model_file, 'wb') as f:
        f.write(b"DUMMY NCF MODEL DATA")
    
    # Create metrics file
    metrics = {
        "algorithm": model_ncf["algorithm"],
        "version": model_ncf["version"],
        "timestamp": int(model_ncf["version"]),
        "map_at_k": 0.33,  # slightly better than ALS
        "ndcg": 0.29,
        "precision": 0.23,
        "recall": 0.36,
        "training_time": 180,
        "hyperparameters": {
            "layers": [64, 32, 16],
            "learning_rate": 0.001,
            "epochs": 20
        }
    }
    
    metrics_dir = os.path.join(project_root, "model_metrics", model_ncf["domain"], model_ncf["dataset"])
    metrics_file = os.path.join(metrics_dir, f"{model_ncf['algorithm']}_{model_ncf['version']}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create manager
    manager = ModelRetrainingManager(data_dir=project_root)
    
    # Set up A/B test
    ab_test = manager.setup_ab_test(
        domain=model_als["domain"],
        dataset=model_als["dataset"],
        model_a={"algorithm": model_als["algorithm"]},
        model_b={"algorithm": model_ncf["algorithm"]},
        traffic_split=0.5,
        test_duration_days=7
    )
    
    logging.info(f"Created A/B test:")
    logging.info(pformat(ab_test))
    
    # Simulate some test results
    # Model B (ncf) will perform better
    for _ in range(100):
        manager.update_ab_test_results(
            domain=model_als["domain"],
            dataset=model_als["dataset"],
            test_id=ab_test["test_id"],
            group="a",
            impression=True,
            conversion=np.random.random() < 0.2  # 20% conversion
        )
        
        manager.update_ab_test_results(
            domain=model_als["domain"],
            dataset=model_als["dataset"],
            test_id=ab_test["test_id"],
            group="b",
            impression=True,
            conversion=np.random.random() < 0.3  # 30% conversion (better)
        )
    
    # Get active tests
    active_tests = manager.get_active_ab_tests(
        domain=model_als["domain"],
        dataset=model_als["dataset"]
    )
    
    logging.info(f"Active A/B tests: {len(active_tests)}")
    
    # Conclude the test
    conclusion = manager.conclude_ab_test(
        domain=model_als["domain"],
        dataset=model_als["dataset"],
        test_id=ab_test["test_id"]
    )
    
    logging.info(f"A/B test conclusion:")
    logging.info(pformat(conclusion))
    
    # Get recommended model (should be model B now)
    recommendation = manager.get_recommended_model(
        domain=model_als["domain"],
        dataset=model_als["dataset"]
    )
    
    logging.info(f"Recommended model after A/B test:")
    logging.info(pformat(recommendation))
    
    return ab_test, conclusion, recommendation

def test_retraining_decision():
    """Test the retraining decision logic."""
    # Create manager
    manager = ModelRetrainingManager(data_dir=project_root)
    
    # Get retraining recommendation
    recommendation = manager.should_retrain(
        domain="entertainment",
        dataset="movielens"
    )
    
    logging.info(f"Retraining recommendation:")
    logging.info(pformat(recommendation))
    
    if recommendation["should_retrain"]:
        logging.info(f"Reasons for retraining:")
        for reason in recommendation["reasons"]:
            logging.info(f"- {reason}")
    
    return recommendation

def test_rollback():
    """Test model rollback functionality."""
    # Get model info from previous tests
    model_info = create_test_model()
    
    # Create manager
    manager = ModelRetrainingManager(data_dir=project_root)
    
    # Register a "bad" model
    bad_model_data = b"BAD MODEL DATA"
    bad_metrics = model_info["metrics"].copy()
    bad_metrics["map_at_k"] = 0.15  # Much worse performance
    
    registration = manager.register_trained_model(
        domain=model_info["domain"],
        dataset=model_info["dataset"],
        algorithm=model_info["algorithm"],
        model_data=bad_model_data,
        metrics=bad_metrics
    )
    
    logging.info(f"Registered bad model:")
    logging.info(pformat(registration))
    
    # Get model history
    history = manager.get_model_history(
        domain=model_info["domain"],
        dataset=model_info["dataset"],
        algorithm=model_info["algorithm"]
    )
    
    logging.info(f"Model history before rollback ({len(history)} entries):")
    for entry in history:
        logging.info(f"- Version: {entry.get('version')}, Performance: {entry.get('map_at_k')}")
    
    # Rollback to previous version
    rollback = manager.rollback_to_version(
        domain=model_info["domain"],
        dataset=model_info["dataset"],
        algorithm=model_info["algorithm"],
        target_version=model_info["version"]
    )
    
    logging.info(f"Rollback result:")
    logging.info(pformat(rollback))
    
    # Get updated history
    updated_history = manager.get_model_history(
        domain=model_info["domain"],
        dataset=model_info["dataset"],
        algorithm=model_info["algorithm"]
    )
    
    logging.info(f"Model history after rollback ({len(updated_history)} entries):")
    for entry in updated_history:
        logging.info(f"- Version: {entry.get('version')}, Performance: {entry.get('map_at_k')}, " +
                    f"Rollback: {entry.get('is_rollback', False)}")
    
    # Get recommended model (should be the rollback version)
    recommendation = manager.get_recommended_model(
        domain=model_info["domain"],
        dataset=model_info["dataset"]
    )
    
    logging.info(f"Recommended model after rollback:")
    logging.info(pformat(recommendation))
    
    return rollback, updated_history, recommendation

def main():
    """Main function to test model retraining manager functionality."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    logging.info("Testing ModelRetrainingManager functionality")
    
    try:
        # Clean up any existing test data
        clean_test_data()
        
        # Test detecting distribution shifts
        logging.info("\n=== Testing distribution shift detection ===")
        test_detect_distribution_shift()
        
        # Test model management
        logging.info("\n=== Testing model management ===")
        test_model_management()
        
        # Test A/B testing
        logging.info("\n=== Testing A/B testing ===")
        test_ab_testing()
        
        # Test retraining decision
        logging.info("\n=== Testing retraining decision ===")
        test_retraining_decision()
        
        # Test rollback
        logging.info("\n=== Testing model rollback ===")
        test_rollback()
        
        logging.info("\nAll tests completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()