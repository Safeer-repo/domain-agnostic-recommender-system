#!/usr/bin/env python3
"""
Create demo users for Amazon dataset
"""

import os
import sys
from passlib.context import CryptContext

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.user_manager import UserManager

# Define password hashing function
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

# Initialize the user manager
data_dir = os.path.abspath("./data")
user_manager = UserManager(data_dir=data_dir)

# Define demo users using the actual Amazon user IDs
demo_users = [
    {"original_id": "AKZTRHX9HPT4J", "username": "user_AKZTRHX9HPT4J", "password": "password123"},
    {"original_id": "AD1MRUOBFE8FG", "username": "user_AD1MRUOBFE8FG", "password": "password123"},
    {"original_id": "A12L7POEN3MQEM", "username": "user_A12L7POEN3MQEM", "password": "password123"},
    {"original_id": "A1JNFS3IZ1XIXN", "username": "user_A1JNFS3IZ1XIXN", "password": "password123"},
    {"original_id": "A1NJY361HDQG5B", "username": "user_A1NJY361HDQG5B", "password": "password123"},
]

# OR if you prefer simpler usernames:
demo_users = [
    {"original_id": "AKZTRHX9HPT4J", "username": "amazon_user_1", "password": "password123"},
    {"original_id": "AD1MRUOBFE8FG", "username": "amazon_user_2", "password": "password123"},
    {"original_id": "A12L7POEN3MQEM", "username": "amazon_user_3", "password": "password123"},
    {"original_id": "A1JNFS3IZ1XIXN", "username": "amazon_user_4", "password": "password123"},
    {"original_id": "A1NJY361HDQG5B", "username": "amazon_user_5", "password": "password123"},
]

def create_demo_users():
    for user in demo_users:
        # Check if user already exists
        if user_manager.user_exists(user["username"]):
            print(f"User {user['username']} already exists. Skipping.")
            continue
        
        # Create the user
        user_id = user_manager.create_user(
            username=user["username"],
            email=f"{user['username']}@example.com",
            domain_preferences={"ecommerce": ["amazon"]}
        )
        
        if user_id:
            # Add password and original ID to metadata
            user_data = user_manager.get_user(user_id)
            metadata = user_data.get("metadata", {})
            metadata["password"] = get_password_hash(user["password"])
            metadata["original_dataset_id"] = user["original_id"]
            user_manager.update_user(user_id, {"metadata": metadata})
            
            print(f"Created demo user: {user['username']} with ID {user_id}")
        else:
            print(f"Failed to create user: {user['username']}")

if __name__ == "__main__":
    create_demo_users()
    print("Demo user creation complete.")