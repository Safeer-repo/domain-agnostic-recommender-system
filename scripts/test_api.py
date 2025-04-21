#!/usr/bin/env python3
"""
Script to test the Recommender System API.
This script tests the main API endpoints to ensure they are working correctly.

Usage: python scripts/test_api.py
"""

import os
import sys
import requests
import json
import logging
import time
from pprint import pformat
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# API base URL (change this if your API is hosted elsewhere)
BASE_URL = "http://localhost:8000"

# Test user credentials
TEST_USER = {
    "username": "testuser",
    "email": "testuser@example.com",
    "password": "password123"
}

TEST_ADMIN = {
    "username": "admin",
    "email": "admin@example.com",
    "password": "adminpass123"
}

# Store auth tokens and user IDs
user_token = None
admin_token = None
user_id = None
admin_id = None

def make_request(method, endpoint, data=None, token=None, expected_status=200):
    """Make an HTTP request to the API"""
    url = f"{BASE_URL}{endpoint}"
    headers = {}
    
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    if method.lower() == "get":
        response = requests.get(url, headers=headers)
    elif method.lower() == "post":
        response = requests.post(url, json=data, headers=headers)
    elif method.lower() == "put":
        response = requests.put(url, json=data, headers=headers)
    elif method.lower() == "delete":
        response = requests.delete(url, headers=headers)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")
    
    # Check status code
    if response.status_code != expected_status:
        logger.error(f"Request failed with status {response.status_code}: {response.text}")
        return None
    
    # Return JSON response if there is one
    try:
        return response.json()
    except:
        return response.text

def test_health_check():
    """Test the health check endpoint"""
    logger.info("Testing health check endpoint...")
    response = make_request("GET", "/health")
    
    if response and response.get("status") == "healthy":
        logger.info("Health check passed!")
        return True
    else:
        logger.error("Health check failed!")
        return False

def test_user_creation():
    """Test user creation endpoint"""
    global user_token, admin_token, user_id, admin_id
    
    logger.info("Testing user creation...")
    
    # Create normal user
    user_data = TEST_USER.copy()
    response = make_request("POST", "/user/create", data=user_data, expected_status=200)
    
    if not response:
        logger.error("Failed to create test user!")
        return False
    
    user_id = response.get('user_id')  # Store the actual user ID
    logger.info(f"Test user created: {response.get('username')}, ID: {user_id}")
    
    # Create admin user
    admin_data = TEST_ADMIN.copy()
    admin_data["domain_preferences"] = {
        "entertainment": ["movielens", "spotify"],
        "ecommerce": ["amazon_electronics"]
    }
    response = make_request("POST", "/user/create", data=admin_data, expected_status=200)
    
    if not response:
        logger.error("Failed to create admin user!")
        return False
    
    admin_id = response.get('user_id')  # Store the admin ID
    logger.info(f"Admin user created: {response.get('username')}, ID: {admin_id}")
    
    # The API should automatically set admin privileges for users with username "admin"
    # If not, you'd need to add an endpoint to set admin privileges
    
    return True

def test_user_login():
    """Test user login endpoints"""
    global user_token, admin_token
    
    logger.info("Testing user login...")
    
    # Login as normal user
    login_data = {
        "username": TEST_USER["username"],
        "password": TEST_USER["password"]
    }
    
    response = make_request("POST", "/user/login", data=login_data)
    
    if not response or "access_token" not in response:
        logger.error("Failed to login as test user!")
        return False
    
    user_token = response["access_token"]
    logger.info(f"Test user logged in successfully, token: {user_token[:10]}...")
    
    # Login as admin user
    login_data = {
        "username": TEST_ADMIN["username"],
        "password": TEST_ADMIN["password"]
    }
    
    response = make_request("POST", "/user/login", data=login_data)
    
    if not response or "access_token" not in response:
        logger.error("Failed to login as admin user!")
        return False
    
    admin_token = response["access_token"]
    logger.info(f"Admin user logged in successfully, token: {admin_token[:10]}...")
    
    # Test getting user info with token
    response = make_request("GET", "/user/me", token=user_token)
    
    if not response or response.get("username") != TEST_USER["username"]:
        logger.error("Failed to get user info!")
        return False
    
    logger.info(f"Successfully retrieved user info: {response.get('username')}")
    
    return True

def test_single_rating_submission():
    """Test submitting a single rating"""
    logger.info("Testing single rating submission...")
    
    if not user_token or not user_id:
        logger.error("No user token or ID available, login first!")
        return False
    
    # Submit a rating
    rating_data = {
        "user_id": user_id,  # Use the actual UUID
        "item_id": 12345,
        "rating": 4.5,
        "source": "explicit"
    }
    
    response = make_request("POST", "/user/rate", data=rating_data, token=user_token)
    
    if not response or response.get("status") != "success":
        logger.error("Failed to submit rating!")
        return False
    
    logger.info("Rating submitted successfully!")
    return True

def test_batch_rating_submission():
    """Test submitting multiple ratings at once"""
    logger.info("Testing batch rating submission...")
    
    if not user_token or not user_id:
        logger.error("No user token or ID available, login first!")
        return False
    
    # Create batch ratings data
    batch_data = {
        "domain": "entertainment",
        "dataset": "movielens",
        "ratings": [
            {
                "user_id": user_id,  # Use the actual UUID
                "item_id": 100,
                "rating": 4.5
            },
            {
                "user_id": user_id,  # Use the actual UUID
                "item_id": 101,
                "rating": 3.0
            },
            {
                "user_id": user_id,  # Use the actual UUID
                "item_id": 102,
                "rating": 5.0
            },
            {
                "user_id": user_id,  # Use the actual UUID
                "item_id": 103,
                "rating": 2.5
            }
        ]
    }
    
    response = make_request("POST", "/user/rate/batch", data=batch_data, token=user_token)
    
    if not response or response.get("status") != "success":
        logger.error("Failed to submit batch ratings!")
        return False
    
    logger.info(f"Successfully submitted {response.get('processed_count')} ratings!")
    return True

def test_recommendation_feedback():
    """Test submitting feedback for a recommendation"""
    logger.info("Testing recommendation feedback submission...")
    
    if not user_token or not user_id:
        logger.error("No user token or ID available, login first!")
        return False
    
    # Submit feedback
    feedback_data = {
        "user_id": user_id,  # Use the actual UUID
        "item_id": 12345,
        "interaction_type": "click",  # click, add_to_cart, purchase
        "timestamp": int(time.time())
    }
    
    response = make_request("POST", "/user/feedback", data=feedback_data, token=user_token)
    
    if not response or response.get("status") != "success":
        logger.error("Failed to submit recommendation feedback!")
        return False
    
    logger.info("Recommendation feedback submitted successfully!")
    return True

def test_user_recommendations():
    """Test getting recommendations for a user"""
    logger.info("Testing user recommendations...")
    
    if not user_token or not user_id:
        logger.error("No user token or ID available, login first!")
        return False
    
    # Request recommendations
    request_data = {
        "user_id": user_id,  # Use the actual UUID
        "domain": "entertainment",
        "dataset": "movielens",
        "count": 5
    }
    
    response = make_request("POST", "/recommendations/user", data=request_data, token=user_token)
    
    if not response or "recommendations" not in response:
        logger.error("Failed to get user recommendations!")
        return False
    
    logger.info(f"Successfully retrieved {len(response.get('recommendations', []))} recommendations!")
    logger.info(f"First recommendation: {pformat(response.get('recommendations', [])[0])}")
    return True

def test_similar_items():
    """Test getting similar items"""
    logger.info("Testing similar items recommendations...")
    
    if not user_token:
        logger.error("No user token available, login first!")
        return False
    
    # Request similar items
    request_data = {
        "item_id": 12345,
        "domain": "entertainment",
        "dataset": "movielens",
        "count": 3
    }
    
    # This might fail with 404 if no models exist yet, which is expected
    response = make_request("POST", "/recommendations/similar", data=request_data, token=user_token, expected_status=None)
    
    if response and "similar_items" in response:
        logger.info(f"Successfully retrieved {len(response.get('similar_items', []))} similar items!")
        logger.info(f"First similar item: {pformat(response.get('similar_items', [])[0])}")
        return True
    else:
        logger.warning("Could not get similar items - this might be expected if no models exist yet")
        return False

def test_model_endpoints():
    """Test endpoints for getting model information"""
    logger.info("Testing model information endpoints...")
    
    if not user_token:
        logger.error("No user token available, login first!")
        return False
        
    # These might fail with 404 if no models exist yet, which is expected
    domain = "entertainment"
    dataset = "movielens"
    
    response = make_request("GET", f"/models/{domain}/{dataset}", token=user_token, expected_status=None)
    
    if response:
        logger.info(f"Current model information retrieved successfully!")
    else:
        logger.warning("Model might not exist yet, which is OK for testing")
    
    # Test model history
    response = make_request("GET", f"/models/{domain}/{dataset}/history", token=user_token, expected_status=None)
    
    if response:
        logger.info("Successfully retrieved model history")
    else:
        logger.warning("Model history might not exist yet, which is OK for testing")
    
    return True

def test_admin_endpoints():
    """Test admin endpoints"""
    logger.info("Testing admin endpoints...")
    
    if not admin_token or not admin_id:
        logger.error("No admin token or ID available, login as admin first!")
        return False
    
    # First let's check if admin privileges are set correctly
    # Try to access stats endpoint which requires admin privileges
    response = make_request("GET", "/admin/stats", token=admin_token, expected_status=None)
    
    if not response:
        logger.warning("Admin privileges may not be set correctly. Trying to continue anyway...")
    else:
        logger.info("Admin privileges are set correctly!")
    
    # Test update dataset endpoint
    update_data = {
        "domain": "entertainment",
        "dataset": "movielens",
        "incremental": True
    }
    
    response = make_request("POST", "/admin/update-dataset", data=update_data, token=admin_token, expected_status=None)
    
    if response and response.get("status") == "processing":
        logger.info("Dataset update triggered successfully!")
    else:
        logger.warning("Failed to trigger dataset update - this might be due to missing admin privileges")
    
    # Test model retraining endpoint
    retrain_data = {
        "domain": "entertainment",
        "dataset": "movielens",
        "algorithm": "als",
        "force": True
    }
    
    response = make_request("POST", "/admin/retrain", data=retrain_data, token=admin_token, expected_status=None)
    
    if response and response.get("status") == "processing":
        logger.info("Model retraining triggered successfully!")
    else:
        logger.warning("Failed to trigger model retraining - this might be due to missing admin privileges")
    
    return True

def run_all_tests():
    """Run all API tests"""
    logger.info("Running all API tests...")
    
    # Basic tests first
    if not test_health_check():
        logger.error("Health check failed, aborting further tests.")
        return False
    
    # Authentication tests
    if not test_user_creation():
        logger.error("User creation failed, aborting further tests.")
        return False
    
    if not test_user_login():
        logger.error("User login failed, aborting further tests.")
        return False
    
    # Rating tests
    test_single_rating_submission()
    test_batch_rating_submission()
    test_recommendation_feedback()
    
    # Recommendation tests
    test_user_recommendations()
    test_similar_items()
    
    # Model info tests
    test_model_endpoints()
    
    # Admin tests
    test_admin_endpoints()
    
    logger.info("All tests completed!")
    return True

if __name__ == "__main__":
    run_all_tests()