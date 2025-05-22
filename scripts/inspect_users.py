#!/usr/bin/env python3
"""
Script to inspect users in the recommender system.
This verifies that users are properly stored with their domain preferences.
"""

import os
import sys
import json
from pprint import pprint

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.user_manager import UserManager

def inspect_users():
    """Inspect users and their domain preferences"""
    # Initialize user manager
    data_dir = os.path.join(project_root, "data")
    user_manager = UserManager(data_dir=data_dir)
    
    # Get all users
    all_users = user_manager.get_all_users()
    print(f"Total users in system: {len(all_users)}")
    
    # Load users index
    users_index_path = os.path.join(data_dir, "users", "users_index.json")
    if os.path.exists(users_index_path):
        with open(users_index_path, 'r') as f:
            users_index = json.load(f)
        print(f"Users in index: {len(users_index)}")
    else:
        users_index = {}
        print("Users index file not found!")
    
    # Find our test users
    entertainment_user = None
    ecommerce_user = None
    
    for user in all_users:
        username = user.get("username", "")
        if username == "entertainment_user":
            entertainment_user = user
        elif username == "ecommerce_user":
            ecommerce_user = user
    
    # Display entertainment user details
    print("\n" + "="*60)
    print("ENTERTAINMENT USER DETAILS")
    print("="*60)
    if entertainment_user:
        user_id = entertainment_user.get("user_id")
        print(f"Username: {entertainment_user.get('username')}")
        print(f"User ID: {user_id}")
        print(f"Email: {entertainment_user.get('email')}")
        print("Domain Preferences:")
        pprint(entertainment_user.get('domain_preferences'))
        
        # Verify user file exists
        user_file = os.path.join(data_dir, "users", f"{user_id}.json")
        if os.path.exists(user_file):
            print(f"User file exists at: {user_file}")
        else:
            print(f"Warning: User file not found at: {user_file}")
            
        # Verify index contains this user
        if users_index.get("entertainment_user") == user_id:
            print("User correctly indexed in users_index.json")
        else:
            print("Warning: User not correctly indexed in users_index.json")
    else:
        print("Entertainment user not found!")
    
    # Display ecommerce user details
    print("\n" + "="*60)
    print("ECOMMERCE USER DETAILS")
    print("="*60)
    if ecommerce_user:
        user_id = ecommerce_user.get("user_id")
        print(f"Username: {ecommerce_user.get('username')}")
        print(f"User ID: {user_id}")
        print(f"Email: {ecommerce_user.get('email')}")
        print("Domain Preferences:")
        pprint(ecommerce_user.get('domain_preferences'))
        
        # Verify user file exists
        user_file = os.path.join(data_dir, "users", f"{user_id}.json")
        if os.path.exists(user_file):
            print(f"User file exists at: {user_file}")
        else:
            print(f"Warning: User file not found at: {user_file}")
            
        # Verify index contains this user
        if users_index.get("ecommerce_user") == user_id:
            print("User correctly indexed in users_index.json")
        else:
            print("Warning: User not correctly indexed in users_index.json")
    else:
        print("Ecommerce user not found!")
        
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if entertainment_user and ecommerce_user:
        print("✅ Both users created successfully with correct domain preferences")
        print("\nEntertainment user has preferences for:")
        for domain, datasets in entertainment_user.get('domain_preferences', {}).items():
            print(f"  - Domain: {domain}")
            print(f"    Datasets: {', '.join(datasets)}")
            
        print("\nEcommerce user has preferences for:")
        for domain, datasets in ecommerce_user.get('domain_preferences', {}).items():
            print(f"  - Domain: {domain}")
            print(f"    Datasets: {', '.join(datasets)}")
    else:
        if not entertainment_user:
            print("❌ Entertainment user not created correctly")
        if not ecommerce_user:
            print("❌ Ecommerce user not created correctly")
        
if __name__ == "__main__":
    inspect_users()