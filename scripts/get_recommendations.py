#!/usr/bin/env python3
"""
Script to get recommendations after user setup and model retraining.
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
    # Try entertainment user
    print("\n=== ENTERTAINMENT USER RECOMMENDATIONS ===")
    entertainment_login = login_user("entertainment_user", "password123")
    if entertainment_login:
        token = entertainment_login.get("access_token")
        user_id = entertainment_login.get("user_id")
        
        ent_recs = get_user_recommendations(
            user_id=user_id,
            domain="entertainment",
            dataset="movielens",
            token=token
        )
        
        if ent_recs:
            recommendations = ent_recs.get("recommendations", [])
            status = ent_recs.get("status")
            
            if status == "cold_start" or not recommendations:
                print("No recommendations (cold start)")
            else:
                print(f"Got {len(recommendations)} entertainment recommendations")
                print(json.dumps(ent_recs, indent=2))
    
    # Try ecommerce user
    print("\n=== ECOMMERCE USER RECOMMENDATIONS ===")
    ecommerce_login = login_user("ecommerce_user", "password123")
    if ecommerce_login:
        token = ecommerce_login.get("access_token")
        user_id = ecommerce_login.get("user_id")
        
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
                print("No recommendations (cold start)")
            else:
                print(f"Got {len(recommendations)} ecommerce recommendations")
                print(json.dumps(eco_recs, indent=2))

if __name__ == "__main__":
    main()