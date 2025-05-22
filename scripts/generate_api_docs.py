#!/usr/bin/env python3
"""
Script to generate API documentation for the Domain-Agnostic Recommender System.
Outputs API.md in the project root directory.
"""

import os
import sys
import json
import inspect
import re
from datetime import datetime
from pydantic import Field

# Add the project root to the Python path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the FastAPI app and Pydantic models

from src.api.app import app, Rating, BatchRatings, RecommendationFeedback, RecommendationRequest, SimilarItemsRequest

def extract_docstring(func):
    """Extract the docstring from a function."""
    docstring = inspect.getdoc(func)
    if docstring:
        return docstring
    return "No description available."

def extract_path_params(path):
    """Extract path parameters from the path string."""
    params = []
    pattern = r"{([^}]+)}"
    matches = re.findall(pattern, path)
    for match in matches:
        params.append(match)
    return params

def generate_request_example(model_class):
    """Generate a request example based on the model class."""
    request_example = {}

    for field_name, field in model_class.model_fields.items():
        # Use field.annotation to get the type in Pydantic V2
        field_type = field.annotation
        if field_type == str:
            request_example[field_name] = "sample string"
        elif field_type == int:
            request_example[field_name] = 123
        elif field_type == float:
            request_example[field_name] = 123.45
        elif field_type == bool:
            request_example[field_name] = True
        elif field_type == list:
            request_example[field_name] = ["item1", "item2"]
        elif field_type == dict:
            request_example[field_name] = {"key": "value"}
        else:
            request_example[field_name] = None  # Default for unknown types

    return request_example

def generate_response_example(route_path, method):
    """Generate a response example based on the route and method."""

    # Debugging: Print the types and values of route_path and method
    print(f"DEBUG: route_path type={type(route_path)}, value={route_path}")
    print(f"DEBUG: method type={type(method)}, value={method}")

    # Ensure route_path is a string
    if not isinstance(route_path, str):
        raise ValueError("route_path must be a string")

    # If method is a set, loop through each method
    if isinstance(method, set):
        response_examples = {}
        for m in method:
            if isinstance(m, str):  # Ensure each method is a string
                response_example = generate_response_example(route_path, m)
                response_examples[m] = response_example
        return response_examples

    # Ensure method is a string
    if not isinstance(method, str):
        raise ValueError("method must be a string")

    # Predefined examples for specific endpoints
    examples = {
        "POST/token": {
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "token_type": "bearer",
            "user_id": "c210ca32-ad7c-41a5-a806-71c95dfce5ec",
            "username": "user123"
        },
        "POST/user/create": {
            "user_id": "c210ca32-ad7c-41a5-a806-71c95dfce5ec",
            "username": "user123",
            "email": "user@example.com",
            "created_at": 1620000000,
            "last_updated": 1620000000,
            "domain_preferences": {
                "entertainment": ["movielens"],
                "ecommerce": ["amazon"]
            },
            "metadata": {}
        },"POST/user/login": {
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "token_type": "bearer",
            "user_id": "c210ca32-ad7c-41a5-a806-71c95dfce5ec",
            "username": "user123"
        },
        "GET/user/me": {
            "user_id": "c210ca32-ad7c-41a5-a806-71c95dfce5ec",
            "username": "user123",
            "email": "user@example.com",
            "created_at": 1620000000,
            "last_updated": 1620000000,
            "domain_preferences": {
                "entertainment": ["movielens"]
            },
            "metadata": {
                "original_dataset_id": "123"
            }
        },
        "POST/user/rate": {
            "status": "success",
            "message": "Rating submitted successfully",
            "rating": {
                "user_id": "c210ca32-ad7c-41a5-a806-71c95dfce5ec",
                "item_id": 12345,
                "rating": 4.5,
                "source": "explicit",
                "domain": "entertainment",
                "dataset": "movielens"
            },
            "domain": "entertainment",
            "dataset": "movielens"
        },
        "POST/user/rate/batch": {
            "status": "success",
            "message": "Successfully stored 2 ratings",
            "processed_count": 2
        },
        "POST/user/feedback": {
            "status": "success",
            "message": "Feedback recorded successfully",
            "feedback": {
                "user_id": "c210ca32-ad7c-41a5-a806-71c95dfce5ec",
                "item_id": 12345,
                "interaction_type": "click",
                "timestamp": 1620000000,
                "domain": "entertainment",
                "dataset": "movielens"
            }
        },
        "POST/recommendations/user": {
            "user_id": "c210ca32-ad7c-41a5-a806-71c95dfce5ec",
            "domain": "entertainment",
            "dataset": "movielens",
            "model": {
                "algorithm": "sar",
                "version": "latest"
            },
            "recommendations": [
                {
                    "item_id": 123,
                    "score": 0.95,
                    "reason": "Based on your interaction history"
                },
                {
                    "item_id": 456,
                    "score": 0.85,
                    "reason": "Based on your interaction history"
                }
            ]
        },
        "POST/recommendations/similar": {
            "item_id": 12345,
            "domain": "entertainment",
            "dataset": "movielens",
            "model": {
                "algorithm": "sar",
                "version": "latest"
            },
            "similar_items": [
                {
                    "item_id": 678,
                    "similarity": 0.95,
                    "reason": "Based on item features and user interactions"
                },
                {
                    "item_id": 789,
                    "similarity": 0.85,
                    "reason": "Based on item features and user interactions"
                }
            ]
        },
        "GET/models/{domain}/{dataset}": {
            "domain": "entertainment",
            "dataset": "movielens",
            "algorithm": "sar",
            "models": [
                {
                    "model_id": "sar_v1",
                    "created_at": 1620000000,
                    "updated_at": 1620000000
                }
            ]
        }
    }

    return examples.get(route_path, {}).get(method, {})

def generate_error_codes_info():
    """Generate error codes information for the API documentation."""
    error_codes = {
        "400": {
            "title": "Bad Request",
            "description": "The server could not understand the request due to invalid syntax."
        },
        "401": {
            "title": "Unauthorized",
            "description": "The client must authenticate itself to get the requested response."
        },
        "404": {
            "title": "Not Found",
            "description": "The server could not find the requested resource."
        },
        "500": {
            "title": "Internal Server Error",
            "description": "The server encountered an error and could not complete the request."
        }
    }

    return error_codes

def generate_markdown_for_routes():
    """Generate the markdown for API routes."""
    markdown = "# API Documentation\n\n"

    # Include error codes section
    markdown += "## Error Codes\n\n"
    error_codes = generate_error_codes_info()
    for code, details in error_codes.items():
        markdown += f"### {code} - {details['title']}\n"
        markdown += f"{details['description']}\n\n"

    for route in app.routes:
        path = route.path
        method = route.methods
        endpoint = route.endpoint
        docstring = extract_docstring(endpoint)

        # Add basic information about the route
        markdown += f"## {method} {path}\n"
        markdown += f"**Description**: {docstring}\n\n"

        # Extract parameters for the route
        params = extract_path_params(path)
        if params:
            markdown += "**Path Parameters**:\n"
            for param in params:
                markdown += f"- `{param}`: Description of the parameter\n"
            markdown += "\n"

        # Add request example
        model_class = None
        if hasattr(endpoint, "__annotations__"):
            for arg, arg_type in endpoint.__annotations__.items():
                if arg_type in [Rating, BatchRatings, RecommendationFeedback, RecommendationRequest, SimilarItemsRequest]:
                    model_class = arg_type
                    break
        request_example = generate_request_example(model_class)
        if request_example:
            markdown += "**Request Example**:\n"
            markdown += f"```json\n{request_example}\n```\n\n"

        # Add response example
        response_example = generate_response_example(path, method)
        if response_example:
            markdown += "**Response Example**:\n"
            markdown += f"```json\n{response_example}\n```\n\n"

    return markdown


def write_api_docs_to_file():
    """Write the generated API documentation to a markdown file."""
    markdown_content = generate_markdown_for_routes()

    # Write to a file named 'API.md' in the project root directory
    api_docs_path = os.path.join(project_root, "API.md")
    with open(api_docs_path, "w") as f:
        f.write(markdown_content)
    print(f"API documentation written to {api_docs_path}")


if __name__ == "__main__":
    write_api_docs_to_file()
