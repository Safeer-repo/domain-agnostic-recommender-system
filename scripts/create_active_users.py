#!/usr/bin/env python3
"""
Script to identify active users from datasets and create accounts for them.
"""

import os
import sys
import pandas as pd
import logging
import random
import string

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.user_manager import UserManager
from passlib.context import CryptContext

# Setup logging and password handling
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    """Generate password hash"""
    return pwd_context.hash(password)

def find_active_users(domain, dataset, num_users=5):
    """
    Find the most active users in a dataset.
    
    Args:
        domain: Domain name (entertainment, ecommerce)
        dataset: Dataset name (movielens, amazon)
        num_users: Number of users to find
        
    Returns:
        List of user IDs sorted by activity
    """
    data_dir = os.path.join(project_root, "data")
    processed_dir = os.path.join(data_dir, "processed", domain, dataset)
    train_path = os.path.join(processed_dir, "train.csv")
    
    if not os.path.exists(train_path):
        logger.error(f"Training data not found: {train_path}")
        return []
    
    # Load the training data
    df = pd.read_csv(train_path)
    
    # Count ratings per user
    user_counts = df['user_id'].value_counts()
    
    # Get top users
    active_users = user_counts.head(num_users).index.tolist()
    
    logger.info(f"Found {len(active_users)} active users in {domain}/{dataset}")
    for i, user_id in enumerate(active_users):
        num_ratings = user_counts[user_id]
        logger.info(f"  User {i+1}: ID {user_id}, {num_ratings} ratings")
    
    return active_users

def create_user_accounts(domain, dataset, user_ids, user_manager):
    """
    Create user accounts for dataset users.
    
    Args:
        domain: Domain name
        dataset: Dataset name
        user_ids: List of dataset user IDs
        user_manager: UserManager instance
        
    Returns:
        List of created user info dictionaries
    """
    created_users = []
    
    for i, original_id in enumerate(user_ids):
        # Create username and password
        username = f"{domain}_{dataset}_user{i+1}"
        password = "password123"  # Simple password for testing
        email = f"{username}@example.com"
        
        # Set domain preferences
        domain_preferences = {
            domain: [dataset]
        }
        
        # Check if user already exists
        if user_manager.user_exists(username) or user_manager.user_exists(email):
            logger.info(f"User {username} already exists, skipping creation")
            existing_user_id = user_manager.get_user_id(username)
            if existing_user_id:
                user = user_manager.get_user(existing_user_id)
                created_users.append(user)
            continue
        
        # Create the user
        user_id = user_manager.create_user(
            username=username,
            email=email,
            domain_preferences=domain_preferences
        )
        
        if not user_id:
            logger.error(f"Failed to create user {username}")
            continue
        
        # Add password and original dataset ID to metadata
        user = user_manager.get_user(user_id)
        metadata = user.get("metadata", {})
        metadata["password"] = get_password_hash(password)
        metadata["original_dataset_id"] = original_id
        user_manager.update_user(user_id, {"metadata": metadata})
        
        # Get updated user info
        user = user_manager.get_user(user_id)
        
        # Display user info (without password)
        user_info = user.copy()
        if "metadata" in user_info and "password" in user_info["metadata"]:
            del user_info["metadata"]["password"]
        
        logger.info(f"Created user: {username}")
        logger.info(f"  User ID: {user_id}")
        logger.info(f"  Original Dataset ID: {original_id}")
        
        created_users.append(user_info)
    
    return created_users

def main():
    """Main function to find active users and create accounts."""
    data_dir = os.path.join(project_root, "data")
    user_manager = UserManager(data_dir=data_dir)
    
    # Process entertainment/movielens
    logger.info("Finding active users in entertainment/movielens...")
    ent_users = find_active_users("entertainment", "movielens", 5)
    
    if ent_users:
        logger.info("Creating user accounts for entertainment/movielens users...")
        ent_created = create_user_accounts("entertainment", "movielens", ent_users, user_manager)
        logger.info(f"Created {len(ent_created)} entertainment user accounts")
    
    # Process ecommerce/amazon
    logger.info("\nFinding active users in ecommerce/amazon...")
    eco_users = find_active_users("ecommerce", "amazon", 5)
    
    if eco_users:
        logger.info("Creating user accounts for ecommerce/amazon users...")
        eco_created = create_user_accounts("ecommerce", "amazon", eco_users, user_manager)
        logger.info(f"Created {len(eco_created)} ecommerce user accounts")
    
    # Print summary
    logger.info("\n===== SUMMARY =====")
    logger.info("Entertainment Users:")
    for i, user in enumerate(ent_created):
        logger.info(f"  {i+1}. Username: {user['username']}")
        logger.info(f"     Original ID: {user['metadata']['original_dataset_id']}")
    
    logger.info("\nEcommerce Users:")
    for i, user in enumerate(eco_created):
        logger.info(f"  {i+1}. Username: {user['username']}")
        logger.info(f"     Original ID: {user['metadata']['original_dataset_id']}")
    
    logger.info("\nAll users created with password: password123")

if __name__ == "__main__":
    main()