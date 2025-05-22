#!/usr/bin/env python3
"""
Script to debug and fix batch rating submission.
"""

import requests
import json

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

def submit_batch_ratings(user_id, domain, dataset, token):
    """Submit multiple ratings at once with simpler format"""
    url = f"{BASE_URL}/user/rate/batch"
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create just 2 ratings to keep it simple
    ratings = [
        {
            "user_id": user_id,
            "item_id": 1001,
            "rating": 4.0,
            "source": "explicit"
        },
        {
            "user_id": user_id,
            "item_id": 1002,
            "rating": 5.0,
            "source": "explicit"
        }
    ]
    
    data = {
        "domain": domain,
        "dataset": dataset,
        "ratings": ratings
    }
    
    print(f"Sending batch request with data: {json.dumps(data, indent=2)}")
    response = requests.post(url, json=data, headers=headers)
    print(f"Batch rating submission: {response.status_code}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.text}")
        return None

def main():
    # Test only with entertainment user for simplicity
    entertainment_login = login_user("entertainment_user", "password123")
    if not entertainment_login:
        print("Failed to login")
        return
    
    token = entertainment_login.get("access_token")
    user_id = entertainment_login.get("user_id")
    
    # Try batch submission with entertainment domain
    result = submit_batch_ratings(
        user_id=user_id,
        domain="entertainment",
        dataset="movielens",
        token=token
    )
    
    if result:
        print(f"Success! {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    main()