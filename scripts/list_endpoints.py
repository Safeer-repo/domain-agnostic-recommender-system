import os
import sys
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the app directly from api.py
from src.api import app

# Get all endpoints
endpoints = []
for route in app.routes:
    endpoint = {
        "path": route.path,
        "name": route.name,
        "methods": list(route.methods) if hasattr(route, "methods") else None,
    }
    endpoints.append(endpoint)

# Print the endpoints
print(json.dumps(endpoints, indent=2))