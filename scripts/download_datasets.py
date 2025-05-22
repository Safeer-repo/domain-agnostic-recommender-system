#!/usr/bin/env python3
"""
Script to download datasets for the recommender system.
Usage: python scripts/download_datasets.py [--domain DOMAIN]
"""

import os
import argparse
import logging
import requests
import zipfile
import io
import shutil
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_file(url, destination):
    """Download a file from a URL to a destination path."""
    logger.info(f"Downloading from {url} to {destination}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise exception for HTTP errors
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Download complete: {destination}")

def download_and_extract(url, extract_dir):
    """Download a zip file and extract its contents."""
    logger.info(f"Downloading and extracting from {url} to {extract_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    
    response = requests.get(url)
    response.raise_for_status()  # Raise exception for HTTP errors
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_dir)
    
    logger.info(f"Extraction complete: {extract_dir}")

def download_movielens(data_dir):
    """Download MovieLens dataset."""
    domain_dir = os.path.join(data_dir, "raw", "entertainment", "movielens")
    
    # MovieLens 100K
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    
    download_and_extract(url, domain_dir)
    
    # Move files from subdirectory to the main domain directory
    extracted_dir = os.path.join(domain_dir, "ml-100k")
    if os.path.exists(extracted_dir):
        for item in os.listdir(extracted_dir):
            src = os.path.join(extracted_dir, item)
            dst = os.path.join(domain_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        
        # Remove the now-empty directory
        if os.path.exists(extracted_dir) and os.path.isdir(extracted_dir):
            try:
                os.rmdir(extracted_dir)
            except:
                logger.warning(f"Could not remove directory {extracted_dir} - it may not be empty")
    
    logger.info("MovieLens dataset download complete")

def download_amazon(data_dir):
    """Download Amazon Electronics dataset from Kaggle."""
    domain_dir = os.path.join(data_dir, "raw", "ecommerce", "amazon")
    os.makedirs(domain_dir, exist_ok=True)
    
    logger.info("Attempting to download Amazon Product Reviews dataset")
    
    try:
        # Check if kaggle is installed and configured
        import importlib.util
        kaggle_spec = importlib.util.find_spec('kaggle')
        
        if kaggle_spec is not None:
            logger.info("Using Kaggle API to download dataset")
            import kaggle
            # Download the dataset
            kaggle.api.dataset_download_files(
                'saurav9786/amazon-product-reviews',
                path=domain_dir,
                unzip=True
            )
            logger.info(f"Dataset successfully downloaded to {domain_dir}")
        else:
            logger.warning("Kaggle API not found. Please follow manual download instructions.")
            _show_manual_amazon_download_instructions(domain_dir)
    
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        _show_manual_amazon_download_instructions(domain_dir)

def _show_manual_amazon_download_instructions(domain_dir):
    """Display instructions for manually downloading the Amazon dataset."""
    logger.info("\n" + "="*80)
    logger.info("MANUAL DOWNLOAD INSTRUCTIONS FOR AMAZON PRODUCT REVIEWS DATASET")
    logger.info("="*80)
    logger.info("1. Visit: https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews/data")
    logger.info("2. Click 'Download' (you may need to create a Kaggle account if you don't have one)")
    logger.info("3. Once downloaded, extract the ZIP file")
    logger.info(f"4. Place the extracted CSV files in this directory: {domain_dir}")
    logger.info("5. Ensure the main review file is named 'amazon_product_reviews.csv'")
    logger.info("="*80 + "\n")

def download_open_university(data_dir):
    """Download Open University Learning Analytics dataset."""
    # This is a placeholder. The actual download may vary.
    logger.info("Open University dataset download would be implemented here")
    logger.info("Please manually download from https://analyse.kmi.open.ac.uk/open_dataset")

def main():
    """Main function to handle dataset downloads."""
    parser = argparse.ArgumentParser(description="Download datasets for the recommender system")
    parser.add_argument("--domain", choices=["all", "entertainment", "ecommerce", "education"],
                        default="all", help="Domain to download datasets for")
    parser.add_argument("--data-dir", default="./data", 
                        help="Path to the data directory (default: ./data)")
    args = parser.parse_args()
    
    data_dir = os.path.abspath(args.data_dir)
    logger.info(f"Using data directory: {data_dir}")
    
    if args.domain in ["all", "entertainment"]:
        download_movielens(data_dir)
    
    if args.domain in ["all", "ecommerce"]:
        download_amazon(data_dir)
    
    if args.domain in ["all", "education"]:
        download_open_university(data_dir)
    
    logger.info("Dataset downloads complete")

if __name__ == "__main__":
    main()
