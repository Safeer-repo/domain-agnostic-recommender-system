#!/usr/bin/env python3
"""
Script to test loading the Amazon dataset from Downloads folder.
"""
import os
import sys
import logging

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessing.data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to test loading the Amazon dataset."""
    # Use the project_root to get the data directory
    data_dir = os.path.join(project_root, "data")
    
    # Create data loader
    data_loader = DataLoader(data_dir)
    
    # Path to the Amazon dataset in Downloads folder - make sure the extension is included
    amazon_file_path = "/Users/safeermoosvi/Downloads/ratings_Electronics (1).csv"
    
    # Load Amazon dataset
    try:
        # Load from downloads
        ratings = data_loader.load_amazon_from_downloads(amazon_file_path)
        
        logger.info("\n" + "="*50)
        logger.info("Amazon dataset loaded from Downloads")
        logger.info(f"Number of ratings: {len(ratings):,}")
        logger.info(f"Number of users: {ratings['user_id'].nunique():,}")
        logger.info(f"Number of products: {ratings['item_id'].nunique():,}")
        logger.info(f"Rating range: {ratings['rating'].min()} - {ratings['rating'].max()}")
        logger.info("\nFirst 5 rows:")
        logger.info(f"\n{ratings.head()}")
        logger.info("="*50)
        
        # Test loading from project structure
        logger.info("\nTesting loading from project structure...")
        ratings_from_structure = data_loader.load_dataset("ecommerce", "amazon")
        
        logger.info("\n" + "="*50)
        logger.info("Amazon dataset loaded from project structure")
        logger.info(f"Number of ratings: {len(ratings_from_structure):,}")
        logger.info(f"Number of users: {ratings_from_structure['user_id'].nunique():,}")
        logger.info(f"Number of products: {ratings_from_structure['item_id'].nunique():,}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")

if __name__ == "__main__":
    main()