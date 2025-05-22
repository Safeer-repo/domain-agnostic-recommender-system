#!/usr/bin/env python3
"""
Script to add original dataset user IDs to our test users.
"""

import os
import sys
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.user_manager import UserManager

def add_dataset_user_id():
    """Add dataset user IDs to test users"""
    data_dir = os.path.join(project_root, "data")
    user_manager = UserManager(data_dir=data_dir)
    
    # Get all users
    all_users = user_manager.get_all_users()
    
    # Find our test users
    entertainment_user = None
    ecommerce_user = None
    
    for user in all_users:
        username = user.get("username", "")
        if username == "entertainment_user":
            entertainment_user = user
        elif username == "ecommerce_user":
            ecommerce_user = user
    
    if entertainment_user:
        # Add original_dataset_id for entertainment user (use a valid MovieLens user ID)
        user_id = entertainment_user.get("user_id")
        metadata = entertainment_user.get("metadata", {})
        metadata["original_dataset_id"] = 1  # Use a common MovieLens user ID (assuming 1 exists)
        user_manager.update_user(user_id, {"metadata": metadata})
        print(f"Added original_dataset_id=1 to entertainment user")
    
    if ecommerce_user:
        # Add original_dataset_id for ecommerce user (use a valid Amazon user ID)
        user_id = ecommerce_user.get("user_id")
        metadata = ecommerce_user.get("metadata", {})
        metadata["original_dataset_id"] = 5001  # Use a common Amazon user ID
        user_manager.update_user(user_id, {"metadata": metadata})
        print(f"Added original_dataset_id=5001 to ecommerce user")

if __name__ == "__main__":
    add_dataset_user_id()