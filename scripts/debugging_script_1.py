#!/usr/bin/env python3
"""
Script to debug model selection and similar items issues
"""
import os
import sys
import logging

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.model_retraining_manager import ModelRetrainingManager
from src.models.model_registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def debug_model_selection():
    """Debug why SAR is being selected as the best model"""
    logger.info("=== Debugging Model Selection ===")
    
    # Initialize managers
    model_manager = ModelRetrainingManager(data_dir="./data", models_dir="./artifacts/models")
    
    # Test for both domains
    domains = [
        ("entertainment", "movielens"),
        ("ecommerce", "amazon")
    ]
    
    for domain, dataset in domains:
        logger.info(f"\nChecking {domain}/{dataset}:")
        
        # Get recommended model
        model_info = model_manager.get_recommended_model(domain, dataset)
        logger.info(f"Selected model: {model_info}")
        
        # Get model history for all algorithms
        model_dir = os.path.join("./data/models", domain, dataset)
        if os.path.exists(model_dir):
            algorithms = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
            logger.info(f"Available algorithms: {algorithms}")
            
            # Get metrics for each algorithm
            for algorithm in algorithms:
                try:
                    history = model_manager.get_model_history(domain, dataset, algorithm, limit=1)
                    if history:
                        latest = history[0]
                        logger.info(f"\n{algorithm} latest metrics:")
                        logger.info(f"  Version: {latest.get('version')}")
                        logger.info(f"  Timestamp: {latest.get('timestamp')}")
                        logger.info(f"  MAP@K: {latest.get('map_at_k', 'N/A')}")
                        logger.info(f"  NDCG: {latest.get('ndcg', 'N/A')}")
                        logger.info(f"  Precision: {latest.get('precision', 'N/A')}")
                        logger.info(f"  Recall: {latest.get('recall', 'N/A')}")
                except Exception as e:
                    logger.error(f"Error getting metrics for {algorithm}: {e}")

def debug_similar_items():
    """Debug why similar items returns empty array"""
    logger.info("\n=== Debugging Similar Items ===")
    
    # Initialize registry
    model_registry = ModelRegistry(artifacts_dir="./artifacts")
    
    # Test for both domains
    test_cases = [
        ("entertainment", "movielens", 1),  # MovieLens item ID
        ("ecommerce", "amazon", "B00005N5PF")  # Amazon item ID
    ]
    
    for domain, dataset, item_id in test_cases:
        logger.info(f"\nTesting {domain}/{dataset} with item_id: {item_id}")
        
        try:
            # Load SAR model
            model = model_registry.load_model("sar", domain, dataset)
            logger.info(f"Model loaded successfully")
            
            # Check if the model has necessary components
            if hasattr(model, 'item_similarity'):
                logger.info(f"Item similarity matrix shape: {model.item_similarity.shape}")
            else:
                logger.warning("Model doesn't have item_similarity matrix")
            
            if hasattr(model, 'item_index'):
                logger.info(f"Number of items in index: {len(model.item_index)}")
                # Check if our test item exists
                if item_id in model.item_index:
                    logger.info(f"Item {item_id} found in index")
                else:
                    logger.warning(f"Item {item_id} NOT found in index")
                    # Show some sample item IDs
                    sample_items = list(model.item_index.keys())[:5]
                    logger.info(f"Sample item IDs: {sample_items}")
            
            # Try to get similar items
            try:
                similar_items = model.get_similar_items(item_id, n=5)
                logger.info(f"Similar items result: {similar_items}")
            except Exception as e:
                logger.error(f"Error getting similar items: {e}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")

def main():
    """Run all debug functions"""
    debug_model_selection()
    debug_similar_items()

if __name__ == "__main__":
    main()