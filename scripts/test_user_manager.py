#!/usr/bin/env python3
"""
Script to test the UserManager functionality.
Usage: python scripts/test_user_manager.py
"""

import os
import sys
import logging
import time
import shutil
from pprint import pformat

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.user_manager import UserManager

def clean_test_data():
    """Clean up test data to start fresh."""
    users_dir = os.path.join(project_root, "users")
    if os.path.exists(users_dir):
        shutil.rmtree(users_dir)
        logging.info(f"Removed test directory: {users_dir}")
    logging.info("Test environment cleaned")

def test_create_users():
    """Test creating users with unique information."""
    manager = UserManager(data_dir=project_root)
    
    # Create first user
    user1_id = manager.create_user(
        username="test_user1",
        email="user1@example.com",
        domain_preferences={
            "entertainment": ["movielens", "spotify"],
            "ecommerce": ["amazon_electronics"]
        }
    )
    
    # Create second user
    user2_id = manager.create_user(
        username="test_user2",
        email="user2@example.com"
    )
    
    # Try to create user with existing username (should fail)
    duplicate_id = manager.create_user(
        username="test_user1",
        email="another_email@example.com"
    )
    
    logging.info(f"User 1 created with ID: {user1_id}")
    logging.info(f"User 2 created with ID: {user2_id}")
    logging.info(f"Duplicate user creation attempt result: {duplicate_id}")
    
    return user1_id, user2_id

def test_user_retrieval(user_id):
    """Test retrieving user information."""
    manager = UserManager(data_dir=project_root)
    
    # Get user by ID
    user_data = manager.get_user(user_id)
    logging.info(f"Retrieved user data:")
    logging.info(pformat(user_data))
    
    # Check if user exists by various identifiers
    exists_by_id = manager.user_exists(user_id)
    exists_by_username = manager.user_exists(user_data["username"])
    exists_by_email = manager.user_exists(user_data["email"])
    
    logging.info(f"User exists by ID: {exists_by_id}")
    logging.info(f"User exists by username: {exists_by_username}")
    logging.info(f"User exists by email: {exists_by_email}")
    
    # Get user ID from username and email
    id_from_username = manager.get_user_id(user_data["username"])
    id_from_email = manager.get_user_id(user_data["email"])
    
    logging.info(f"User ID from username: {id_from_username}")
    logging.info(f"User ID from email: {id_from_email}")
    
    return user_data

def test_user_update(user_id):
    """Test updating user information."""
    manager = UserManager(data_dir=project_root)
    
    # Get original data
    original_data = manager.get_user(user_id)
    logging.info(f"Original user data:")
    logging.info(pformat(original_data))
    
    # Update user data
    updates = {
        "email": f"updated_{user_id}@example.com",
        "metadata": {
            "age_group": "25-34",
            "location": "New York"
        }
    }
    
    success = manager.update_user(user_id, updates)
    logging.info(f"User update success: {success}")
    
    # Get updated data
    updated_data = manager.get_user(user_id)
    logging.info(f"Updated user data:")
    logging.info(pformat(updated_data))
    
    # Check that email was updated in the index
    id_from_new_email = manager.get_user_id(updates["email"])
    logging.info(f"User ID from new email: {id_from_new_email}")
    
    return original_data, updated_data

def test_domain_preferences(user_id):
    """Test adding domain preferences."""
    manager = UserManager(data_dir=project_root)
    
    # Add new domain preference
    success = manager.add_domain_preference(
        user_id=user_id,
        domain="education",
        dataset="open_university"
    )
    
    logging.info(f"Adding domain preference success: {success}")
    
    # Get updated user data with new preference
    user_data = manager.get_user(user_id)
    logging.info(f"User preferences after update:")
    logging.info(pformat(user_data.get("domain_preferences", {})))
    
    return user_data

def test_user_filtering():
    """Test filtering users by domain interests."""
    manager = UserManager(data_dir=project_root)
    
    # Get all users
    all_users = manager.get_all_users()
    logging.info(f"Total users: {len(all_users)}")
    
    # Get users interested in entertainment
    entertainment_users = manager.get_users_by_domain_interest("entertainment")
    logging.info(f"Users interested in entertainment: {len(entertainment_users)}")
    
    # Get users interested in movielens
    movielens_users = manager.get_users_by_domain_interest("entertainment", "movielens")
    logging.info(f"Users interested in movielens: {len(movielens_users)}")
    
    # Get users interested in education
    education_users = manager.get_users_by_domain_interest("education")
    logging.info(f"Users interested in education: {len(education_users)}")
    
    return all_users, entertainment_users, education_users

def main():
    """Main function to test user manager functionality."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    logging.info("Testing UserManager functionality")
    
    try:
        # Clean up any existing test data
        clean_test_data()
        
        # Test creating users
        logging.info("\n=== Testing user creation ===")
        user1_id, user2_id = test_create_users()
        
        # Test retrieving user information
        logging.info("\n=== Testing user retrieval ===")
        test_user_retrieval(user1_id)
        
        # Test updating user information
        logging.info("\n=== Testing user update ===")
        test_user_update(user2_id)
        
        # Test domain preferences
        logging.info("\n=== Testing domain preferences ===")
        test_domain_preferences(user1_id)
        
        # Test user filtering
        logging.info("\n=== Testing user filtering ===")
        test_user_filtering()
        
        logging.info("\nAll tests completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()