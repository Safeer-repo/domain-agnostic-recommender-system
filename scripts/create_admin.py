#!/usr/bin/env python3
"""
Script to create a dedicated admin user with proper privileges.
"""

import os
import sys
import requests
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.user_manager import UserManager

def create_admin_via_api():
    """Create admin user through the API endpoint"""
    base_url = "http://localhost:8000"
    
    # First try to create the admin user via API
    create_data = {
        "username": "admin",
        "email": "admin@example.com",
        "password": "adminpass123",
        "domain_preferences": {
            "entertainment": ["movielens"],
            "ecommerce": ["amazon"]
        }
    }
    
    response = requests.post(f"{base_url}/user/create", json=create_data)
    print(f"Admin user creation response: {response.status_code}")
    
    if response.status_code == 200:
        print("Admin user created successfully via API")
        return response.json().get("user_id")
    else:
        print(f"Error creating admin: {response.text}")
        return None

def set_admin_privileges_directly():
    """Set admin privileges directly using the UserManager"""
    data_dir = os.path.join(project_root, "data")
    user_manager = UserManager(data_dir=data_dir)
    
    # Try to get user ID by username
    admin_id = user_manager.get_user_id("admin")
    
    if admin_id:
        # Get the user data
        admin_user = user_manager.get_user(admin_id)
        
        if admin_user:
            # Update metadata to include admin privileges
            metadata = admin_user.get("metadata", {})
            metadata["is_admin"] = True
            
            # Update the user
            user_manager.update_user(admin_id, {"metadata": metadata})
            print(f"Admin privileges set for user ID: {admin_id}")
            
            # Verify
            updated_user = user_manager.get_user(admin_id)
            if updated_user.get("metadata", {}).get("is_admin"):
                print("Verified admin privileges are set")
            else:
                print("Failed to set admin privileges")
        else:
            print(f"Admin user not found with ID: {admin_id}")
    else:
        print("Admin user not found in user_manager")

def main():
    # First try to create the admin user via API
    admin_id = create_admin_via_api()
    
    # Set admin privileges directly
    set_admin_privileges_directly()
    
    print("\nAdmin user setup complete!")

if __name__ == "__main__":
    main()