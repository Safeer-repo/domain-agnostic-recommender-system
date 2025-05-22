#!/usr/bin/env python3
"""
Script to test domain-specific rating submission.
Submits ratings from entertainment and ecommerce users to their respective domains.
"""

import os
import sys
import time
import requests
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

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
    print(f"Rating submission: {response.status_code}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.text}")
        return None

def submit_batch_ratings(user_id, domain, dataset, token):
    """Submit multiple ratings at once"""
    url = f"{BASE_URL}/user/rate/batch"
    headers = {"Authorization": f"Bearer {token}"}
    
    # Generate some test item IDs based on domain
    start_id = 1000 if domain == "entertainment" else 5000
    
    # Create 5 ratings
    ratings = []
    for i in range(5):
        item_id = start_id + i
        # Random-ish rating between 1 and 5
        rating = ((item_id % 5) + 1) * 1.0
        ratings.append({
            "user_id": user_id,
            "item_id": item_id,
            "rating": rating
        })
    
    data = {
        "domain": domain,
        "dataset": dataset,
        "ratings": ratings
    }
    
    response = requests.post(url, json=data, headers=headers)
    print(f"Batch rating submission: {response.status_code}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.text}")
        return None

def get_user_recommendations(user_id, domain, dataset, token):
    """Get recommendations for a user"""
    url = f"{BASE_URL}/recommendations/user"
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "user_id": user_id,
        "domain": domain,
        "dataset": dataset,
        "count": 5
    }
    response = requests.post(url, json=data, headers=headers)
    print(f"Recommendations response: {response.status_code}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.text}")
        return None

def main():
    # Login entertainment user
    print("\n=== ENTERTAINMENT USER ===")
    entertainment_login = login_user("entertainment_user", "password123")
    if not entertainment_login:
        print("Failed to login entertainment user")
        return
    
    entertainment_token = entertainment_login.get("access_token")
    entertainment_user_id = entertainment_login.get("user_id")
    
    # Submit ratings for entertainment user
    print("\nSubmitting ratings for entertainment domain...")
    single_rating = submit_rating(
        user_id=entertainment_user_id,
        item_id=1,  # A MovieLens movie ID
        rating=4.5,
        domain="entertainment",
        dataset="movielens",
        token=entertainment_token
    )
    
    if single_rating:
        print(f"Single rating submitted: {json.dumps(single_rating, indent=2)}")
    
    batch_ratings = submit_batch_ratings(
        user_id=entertainment_user_id,
        domain="entertainment",
        dataset="movielens",
        token=entertainment_token
    )
    
    if batch_ratings:
        print(f"Batch ratings submitted: {json.dumps(batch_ratings, indent=2)}")
    
    # Login ecommerce user
    print("\n=== ECOMMERCE USER ===")
    ecommerce_login = login_user("ecommerce_user", "password123")
    if not ecommerce_login:
        print("Failed to login ecommerce user")
        return
    
    ecommerce_token = ecommerce_login.get("access_token")
    ecommerce_user_id = ecommerce_login.get("user_id")
    
    # Submit ratings for ecommerce user
    print("\nSubmitting ratings for ecommerce domain...")
    single_rating = submit_rating(
        user_id=ecommerce_user_id,
        item_id=5001,  # An Amazon product ID
        rating=5.0,
        domain="ecommerce",
        dataset="amazon",
        token=ecommerce_token
    )
    
    if single_rating:
        print(f"Single rating submitted: {json.dumps(single_rating, indent=2)}")
    
    batch_ratings = submit_batch_ratings(
        user_id=ecommerce_user_id,
        domain="ecommerce",
        dataset="amazon",
        token=ecommerce_token
    )
    
    if batch_ratings:
        print(f"Batch ratings submitted: {json.dumps(batch_ratings, indent=2)}")
    
    # Wait a moment for processing
    print("\nWaiting for rating processing...")
    time.sleep(2)
    
    # Get recommendations for both users
    print("\n=== GETTING RECOMMENDATIONS ===")
    print("\nEntertainment recommendations:")
    ent_recs = get_user_recommendations(
        user_id=entertainment_user_id,
        domain="entertainment",
        dataset="movielens",
        token=entertainment_token
    )
    
    if ent_recs:
        print(f"Entertainment recommendations: {json.dumps(ent_recs, indent=2)}")
    
    print("\nEcommerce recommendations:")
    eco_recs = get_user_recommendations(
        user_id=ecommerce_user_id,
        domain="ecommerce",
        dataset="amazon",
        token=ecommerce_token
    )
    
    if eco_recs:
        print(f"Ecommerce recommendations: {json.dumps(eco_recs, indent=2)}")
    
    print("\nDomain isolation test complete!")

if __name__ == "__main__":
    main()