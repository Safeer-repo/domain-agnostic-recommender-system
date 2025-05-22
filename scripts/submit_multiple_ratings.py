#!/usr/bin/env python3
"""
Script to submit enough ratings to generate recommendations.
"""

import requests
import json
import time
import random

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
    if response.status_code == 200:
        return True
    else:
        print(f"Error submitting rating for item {item_id}: {response.text}")
        return False

def submit_multiple_ratings(user_id, domain, dataset, token, count=10):
    """Submit multiple individual ratings"""
    print(f"Submitting {count} ratings for {domain}/{dataset}...")
    
    # Base item ID based on domain
    base_id = 1 if domain == "entertainment" else 5001
    
    success_count = 0
    for i in range(count):
        item_id = base_id + i
        # Random rating between 3.0 and 5.0 (generally positive)
        rating = round(random.uniform(3.0, 5.0), 1)
        
        print(f"Submitting rating: item_id={item_id}, rating={rating}")
        if submit_rating(user_id, item_id, rating, domain, dataset, token):
            success_count += 1
            # Small delay to avoid overwhelming the API
            time.sleep(0.2)
    
    print(f"Successfully submitted {success_count} out of {count} ratings")
    return success_count

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
    print(f"Recommendations response: {response.status_code}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.text}")
        return None

def main():
    # Entertainment domain
    print("\n=== ENTERTAINMENT USER ===")
    entertainment_login = login_user("entertainment_user", "password123")
    if entertainment_login:
        ent_token = entertainment_login.get("access_token")
        ent_user_id = entertainment_login.get("user_id")
        
        # Submit enough ratings to overcome cold start
        submit_multiple_ratings(
            user_id=ent_user_id,
            domain="entertainment",
            dataset="movielens",
            token=ent_token,
            count=5  # Submit 15 ratings
        )
        
        # Get recommendations
        print("\nGetting entertainment recommendations...")
        ent_recs = get_user_recommendations(
            user_id=ent_user_id,
            domain="entertainment",
            dataset="movielens",
            token=ent_token
        )
        
        if ent_recs and ent_recs.get("recommendations"):
            print(f"Got {len(ent_recs['recommendations'])} recommendations")
            print(json.dumps(ent_recs, indent=2))
        else:
            print("No recommendations or still in cold start")
    
    # Ecommerce domain
    print("\n=== ECOMMERCE USER ===")
    ecommerce_login = login_user("ecommerce_user", "password123")
    if ecommerce_login:
        eco_token = ecommerce_login.get("access_token")
        eco_user_id = ecommerce_login.get("user_id")
        
        # Submit enough ratings to overcome cold start
        submit_multiple_ratings(
            user_id=eco_user_id,
            domain="ecommerce",
            dataset="amazon",
            token=eco_token,
            count=15  # Submit 15 ratings
        )
        
        # Get recommendations
        print("\nGetting ecommerce recommendations...")
        eco_recs = get_user_recommendations(
            user_id=eco_user_id,
            domain="ecommerce",
            dataset="amazon",
            token=eco_token
        )
        
        if eco_recs and eco_recs.get("recommendations"):
            print(f"Got {len(eco_recs['recommendations'])} recommendations")
            print(json.dumps(eco_recs, indent=2))
        else:
            print("No recommendations or still in cold start")

if __name__ == "__main__":
    main()