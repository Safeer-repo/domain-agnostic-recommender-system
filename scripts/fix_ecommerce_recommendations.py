#!/usr/bin/env python3
"""
Comprehensive script to diagnose and fix ecommerce recommendations issues.
"""

import os
import sys
import requests
import json
import time
import pandas as pd
import random

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.user_manager import UserManager

# API base URL
BASE_URL = "http://localhost:8000"

def login_user(username, password):
    """Login and get access token"""
    url = f"{BASE_URL}/user/login"
    data = {
        "username": username,
        "password": password
    }
    response = requests.post(url, json=data)
    print(f"Login response ({username}): {response.status_code}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.text}")
        return None

def login_admin():
    """Login as admin user"""
    return login_user("admin", "adminpass123")

def submit_rating(user_id, item_id, rating, domain, dataset, token):
    """Submit a rating"""
    url = f"{BASE_URL}/user/rate"
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "user_id": user_id,
        "item_id": item_id,
        "rating": rating,
        "domain": domain,
        "dataset": dataset,
        "source": "explicit"
    }
    response = requests.post(url, json=data, headers=headers)
    return response.status_code == 200

def get_user_recommendations(user_id, domain, dataset, token):
    """Get recommendations for a user"""
    url = f"{BASE_URL}/recommendations/user"
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "user_id": user_id,
        "domain": domain,
        "dataset": dataset,
        "count": 10
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error getting recommendations: {response.text}")
        return None

def trigger_retraining(token, domain, dataset):
    """Trigger model retraining"""
    url = f"{BASE_URL}/admin/retrain"
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "domain": domain,
        "dataset": dataset,
        "force": True
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error triggering retraining: {response.text}")
        return None

def find_valid_item_ids(domain, dataset):
    """Try to find valid item IDs from the dataset files"""
    # Check processed data first
    processed_path = os.path.join(project_root, "data", "processed", domain, dataset, "train.csv")
    
    if os.path.exists(processed_path):
        try:
            df = pd.read_csv(processed_path)
            if 'item_id' in df.columns:
                # Get the most common item IDs (most likely to have ratings)
                item_counts = df['item_id'].value_counts()
                popular_items = item_counts.head(20).index.tolist()
                print(f"Found {len(popular_items)} popular item IDs in {domain}/{dataset}")
                return popular_items
        except Exception as e:
            print(f"Error reading processed data: {str(e)}")
    
    # If we couldn't find any, try some random IDs
    if domain == "entertainment":
        return list(range(1, 21))  # MovieLens items typically start from 1
    else:
        return list(range(100000, 100020))  # Amazon items might have large IDs

def find_valid_user_ids(domain, dataset):
    """Try to find valid user IDs from the dataset files"""
    # Check processed data first
    processed_path = os.path.join(project_root, "data", "processed", domain, dataset, "train.csv")
    
    if os.path.exists(processed_path):
        try:
            df = pd.read_csv(processed_path)
            if 'user_id' in df.columns:
                # Get the most active users (most likely to get recommendations)
                user_counts = df['user_id'].value_counts()
                active_users = user_counts.head(10).index.tolist()
                print(f"Found {len(active_users)} active user IDs in {domain}/{dataset}")
                return active_users
        except Exception as e:
            print(f"Error reading processed data: {str(e)}")
    
    # If we couldn't find any, return some common IDs
    return [1, 2, 3, 4, 5, 10, 100, 1000]

def main():
    # First, let's find some valid item IDs
    print("\n=== FINDING VALID ITEM IDs ===")
    ecommerce_items = find_valid_item_ids("ecommerce", "amazon")
    print(f"Ecommerce items to try: {ecommerce_items}")
    
    # Find some valid user IDs
    print("\n=== FINDING VALID USER IDs ===")
    ecommerce_users = find_valid_user_ids("ecommerce", "amazon")
    print(f"Ecommerce users to try: {ecommerce_users}")
    
    # Login as ecommerce user
    ecommerce_login = login_user("ecommerce_user", "password123")
    if not ecommerce_login:
        print("Failed to login as ecommerce user")
        return
    
    token = ecommerce_login.get("access_token")
    user_id = ecommerce_login.get("user_id")
    
    # Login as admin for retraining
    admin_login = login_admin()
    if not admin_login:
        print("Failed to login as admin")
        return
    
    admin_token = admin_login.get("access_token")
    
    # Submit ratings for valid item IDs
    print("\n=== SUBMITTING RATINGS FOR VALID ITEMS ===")
    success_count = 0
    
    for item_id in ecommerce_items:
        # Generate a rating between 3 and 5 (generally positive)
        rating = round(random.uniform(3.0, 5.0), 1)
        print(f"Submitting rating: item_id={item_id}, rating={rating}")
        
        if submit_rating(user_id, item_id, rating, "ecommerce", "amazon", token):
            success_count += 1
    
    print(f"Successfully submitted {success_count} ratings for valid items")
    
    # Force retrain the ecommerce model
    print("\n=== RETRAINING ECOMMERCE MODEL ===")
    retrain_result = trigger_retraining(admin_token, "ecommerce", "amazon")
    
    if retrain_result:
        print("Retraining successfully triggered")
    
    # Wait for retraining
    print("Waiting for retraining to complete...")
    time.sleep(10)
    
    # Try each of the valid user IDs as original_dataset_id
    print("\n=== TRYING DIFFERENT ORIGINAL_DATASET_IDs ===")
    
    user_manager = UserManager(data_dir=os.path.join(project_root, "data"))
    
    for original_id in ecommerce_users:
        print(f"\nTrying original_dataset_id={original_id}...")
        
        # Update the user's original_dataset_id
        metadata = user_manager.get_user(user_id).get("metadata", {})
        metadata["original_dataset_id"] = original_id
        user_manager.update_user(user_id, {"metadata": metadata})
        
        # Try getting recommendations
        eco_recs = get_user_recommendations(
            user_id=user_id,
            domain="ecommerce",
            dataset="amazon",
            token=token
        )
        
        if eco_recs and eco_recs.get("recommendations") and not eco_recs.get("status") == "cold_start":
            print(f"Success with original_dataset_id={original_id}!")
            print(f"Got {len(eco_recs['recommendations'])} ecommerce recommendations")
            print(json.dumps(eco_recs, indent=2))
            return  # Exit if we found a working ID
    
    # If we get here, try a different approach - use the user's own ID directly
    print("\n=== TRYING DIRECT RECOMMENDATION APPROACH ===")
    
    # Update to use UUID directly
    metadata = user_manager.get_user(user_id).get("metadata", {})
    metadata["original_dataset_id"] = None  # Remove mapping to use direct UUID
    user_manager.update_user(user_id, {"metadata": metadata})
    
    # Try one more time
    eco_recs = get_user_recommendations(
        user_id=user_id,
        domain="ecommerce",
        dataset="amazon",
        token=token
    )
    
    if eco_recs:
        recommendations = eco_recs.get("recommendations", [])
        status = eco_recs.get("status")
        
        if status == "cold_start" or not recommendations:
            print("\n=== DIAGNOSTICS ===")
            print("Still in cold start mode. Let's check model info:")
            
            # Check model info
            model_url = f"{BASE_URL}/models/ecommerce/amazon"
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(model_url, headers=headers)
            
            if response.status_code == 200:
                model_info = response.json()
                print("Model info:")
                print(json.dumps(model_info, indent=2))
            else:
                print(f"Error getting model info: {response.text}")
            
            print("\nPossible issues:")
            print("1. The model might not support cold start recommendations")
            print("2. The system might require more ratings or different item IDs")
            print("3. The Amazon dataset might have special constraints")
            print("\nRecommendation: Contact the system developer for specific Amazon dataset details")
        else:
            print(f"Got {len(recommendations)} ecommerce recommendations")
            print(json.dumps(eco_recs, indent=2))

if __name__ == "__main__":
    main()