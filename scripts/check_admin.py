#!/usr/bin/env python3
"""
Script to check admin status.
"""

import requests
import json

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

def check_admin_stats(token):
    """Check if we can access admin stats"""
    url = f"{BASE_URL}/admin/stats"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    print(f"Admin stats response: {response.status_code}")
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
    
    # Check if we can access admin stats
    stats = check_admin_stats(admin_token)
    if stats:
        print("Admin privileges confirmed!")
        print("\nSystem stats:")
        print(json.dumps(stats, indent=2))
    else:
        print("Failed to confirm admin privileges")

if __name__ == "__main__":
    main()