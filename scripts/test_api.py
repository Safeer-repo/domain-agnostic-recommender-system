#!/usr/bin/env python3
"""
Script to test the API server.
Usage: python scripts/test_api.py
"""

import requests
import json
import argparse
import time

def test_health(base_url):
    """Test the health check endpoint."""
    response = requests.get(f"{base_url}/health")
    print(f"Health check: {response.status_code}")
    print(response.json())
    print()

def test_get_recommendations(base_url, user_id, domain, dataset):
    """Test getting recommendations for a user."""
    url = f"{base_url}/recommendations/user"
    
    payload = {
        "user_id": user_id,
        "n": 5,
        "domain": domain,
        "dataset": dataset
    }
    
    print(f"Getting recommendations for user {user_id}...")
    response = requests.post(url, json=payload)
    
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Recommendations for user {data['user_id']} using {data['model_id']}:")
        for i, rec in enumerate(data["recommendations"], 1):
            print(f"  {i}. Item {rec['item_id']}: {rec['score']:.4f}")
    else:
        print(f"Error: {response.text}")
    print()

def test_get_similar_items(base_url, item_id, domain, dataset):
    """Test getting similar items."""
    url = f"{base_url}/recommendations/similar"
    
    payload = {
        "item_id": item_id,
        "n": 5,
        "domain": domain,
        "dataset": dataset
    }
    
    print(f"Getting items similar to item {item_id}...")
    response = requests.post(url, json=payload)
    
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Similar items to {data['item_id']} using {data['model_id']}:")
        for i, item in enumerate(data["similar_items"], 1):
            print(f"  {i}. Item {item['item_id']}: {item['score']:.4f}")
    else:
        print(f"Error: {response.text}")
    print()

def test_list_models(base_url, domain=None, dataset=None):
    """Test listing available models."""
    url = f"{base_url}/models/"
    
    params = {}
    if domain:
        params["domain"] = domain
    if dataset:
        params["dataset"] = dataset
    
    print(f"Listing models...")
    response = requests.get(url, params=params)
    
    print(f"Status code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Available models:")
        for i, model in enumerate(data["models"], 1):
            print(f"  {i}. {model['model_id']} - {model['domain']}/{model['dataset']}")
            if model.get("metrics"):
                print(f"     Metrics: {json.dumps(model['metrics'])}")
    else:
        print(f"Error: {response.text}")
    print()

def main():
    """Main function to test the API."""
    parser = argparse.ArgumentParser(description="Test the API")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--user-id", type=int, default=123, help="User ID to get recommendations for")
    parser.add_argument("--item-id", type=int, default=123, help="Item ID to get similar items for")
    parser.add_argument("--domain", default="entertainment", help="Domain to use")
    parser.add_argument("--dataset", default="movielens", help="Dataset to use")
    args = parser.parse_args()
    
    # Run tests
    test_health(args.base_url)
    time.sleep(1)
    
    test_list_models(args.base_url, args.domain, args.dataset)
    time.sleep(1)
    
    test_get_recommendations(args.base_url, args.user_id, args.domain, args.dataset)
    time.sleep(1)
    
    test_get_similar_items(args.base_url, args.item_id, args.domain, args.dataset)

if __name__ == "__main__":
    main()
