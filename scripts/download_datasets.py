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

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import load_config

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

def download_movielens(config):
    """Download MovieLens dataset."""
    domain_dir = os.path.join(config.data_dir, "raw", "entertainment", "movielens")
    
    # MovieLens 100K
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    
    download_and_extract(url, domain_dir)
    
    # Move files from subdirectory to the main domain directory
    extracted_dir = os.path.join(domain_dir, "ml-100k")
    if os.path.exists(extracted_dir):
        for item in os.listdir(extracted_dir):
            src = os.path.join(extracted_dir, item)
            dst = os.path.join(domain_dir, item)
            shutil.move(src, dst)
        
        # Remove the now-empty directory
        os.rmdir(extracted_dir)
    
    logger.info("MovieLens dataset download complete")

def download_amazon(config):
    """Download Amazon Electronics dataset."""
    # Note: This is a placeholder. The actual Amazon Electronics dataset
    # may be too large or require different download methods.
    logger.info("Amazon Electronics dataset download would be implemented here")
    logger.info("Please manually download from https://jmcauley.ucsd.edu/data/amazon/")

def download_open_university(config):
    """Download Open University Learning Analytics dataset."""
    # This is a placeholder. The actual download may vary.
    logger.info("Open University dataset download would be implemented here")
    logger.info("Please manually download from https://analyse.kmi.open.ac.uk/open_dataset")

def main():
    """Main function to handle dataset downloads."""
    parser = argparse.ArgumentParser(description="Download datasets for the recommender system")
    parser.add_argument("--domain", choices=["all", "entertainment", "ecommerce", "education"],
                        default="all", help="Domain to download datasets for")
    args = parser.parse_args()
    
    config = load_config()
    
    if args.domain in ["all", "entertainment"]:
        download_movielens(config)
    
    if args.domain in ["all", "ecommerce"]:
        download_amazon(config)
    
    if args.domain in ["all", "education"]:
        download_open_university(config)
    
    logger.info("Dataset downloads complete")

if __name__ == "__main__":
    main()
