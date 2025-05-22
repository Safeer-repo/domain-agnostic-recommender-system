#!/usr/bin/env python3
"""
Script to login as admin and trigger model retraining.
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def login_as_admin():
    """Login as admin user"""
    url = f"{BASE_URL}/user/login"
    data = {
        "username": "admin",
        "password": "adminpass123"
    }
    response = requests.post(url, json=data)
    print(f"Admin login response: {response.status_code}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.text}")
        return None

def trigger_dataset_update(token, domain, dataset):
    """Trigger dataset update to incorporate new ratings"""
    url = f"{BASE_URL}/admin/update-dataset"
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "domain": domain,
        "dataset": dataset,
        "incremental": True
    }
    response = requests.post(url, json=data, headers=headers)
    print(f"Dataset update response: {response.status_code}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.text}")
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
    print(f"Retraining response: {response.status_code}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.text}")
        return None

def main():
    # Login as admin
    admin_login = login_as_admin()
    if not admin_login:
        print("Failed to login as admin")
        return
    
    admin_token = admin_login.get("access_token")
    
    # Trigger updates for entertainment domain
    print("\nUpdating entertainment/movielens dataset...")
    entertainment_update = trigger_dataset_update(admin_token, "entertainment", "movielens")
    
    # Wait a moment for processing
    print("Waiting for processing...")
    time.sleep(5)
    
    # Trigger retraining for entertainment domain
    print("\nRetraining entertainment/movielens model...")
    entertainment_retrain = trigger_retraining(admin_token, "entertainment", "movielens")
    
    # Trigger updates for ecommerce domain
    print("\nUpdating ecommerce/amazon dataset...")
    ecommerce_update = trigger_dataset_update(admin_token, "ecommerce", "amazon")
    
    # Wait a moment for processing
    print("Waiting for processing...")
    time.sleep(5)
    
    # Trigger retraining for ecommerce domain
    print("\nRetraining ecommerce/amazon model...")
    ecommerce_retrain = trigger_retraining(admin_token, "ecommerce", "amazon")
    
    print("\nAll updates and retraining complete!")

if __name__ == "__main__":
    main()