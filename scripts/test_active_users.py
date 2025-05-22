#!/usr/bin/env python3
"""
Script to test recommendations for active users.
"""

import os
import sys
import requests
import json
import logging
from prettytable import PrettyTable

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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
    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Login failed for {username}: {response.text}")
        return None

def get_recommendations(user_id, domain, dataset, token, count=5):
    """Get recommendations for a user"""
    url = f"{BASE_URL}/recommendations/user"
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "user_id": user_id,
        "domain": domain,
        "dataset": dataset,
        "count": count
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Failed to get recommendations: {response.text}")
        return None

def test_users(domain, dataset, num_users=5):
    """Test recommendations for users in a domain/dataset"""
    table = PrettyTable()
    table.field_names = ["Username", "User ID", "Original ID", "Rec. Status", "# of Recs"]
    
    for i in range(1, num_users + 1):
        username = f"{domain}_{dataset}_user{i}"
        
        # Login
        login_result = login_user(username, "password123")
        if not login_result:
            table.add_row([username, "N/A", "N/A", "Login Failed", 0])
            continue
        
        user_id = login_result["user_id"]
        token = login_result["access_token"]
        
        # Get recommendations
        recs = get_recommendations(user_id, domain, dataset, token)
        if not recs:
            table.add_row([username, user_id, "Unknown", "Rec. Failed", 0])
            continue
        
        status = recs.get("status", "Success")
        rec_count = len(recs.get("recommendations", []))
        original_id = "Unknown"  # We'd need to fetch the user profile to get this
        
        table.add_row([username, user_id[:8] + "...", original_id, status, rec_count])
        
        # Print first recommendation if available
        if rec_count > 0:
            logger.info(f"First recommendation for {username}:")
            logger.info(f"  Item ID: {recs['recommendations'][0]['item_id']}")
            logger.info(f"  Score: {recs['recommendations'][0]['score']}")
            logger.info(f"  Reason: {recs['recommendations'][0]['reason']}")
    
    return table

def main():
    """Main function to test recommendations for active users."""
    logger.info("Testing recommendations for entertainment users...")
    ent_table = test_users("entertainment", "movielens")
    
    logger.info("\nTesting recommendations for ecommerce users...")
    eco_table = test_users("ecommerce", "amazon")
    
    # Print summary tables
    logger.info("\n===== ENTERTAINMENT USERS =====")
    print(ent_table)
    
    logger.info("\n===== ECOMMERCE USERS =====")
    print(eco_table)

if __name__ == "__main__":
    main()