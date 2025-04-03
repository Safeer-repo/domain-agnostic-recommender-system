#!/usr/bin/env python3
"""
Script to run the API server.
Usage: python scripts/run_api.py [--port PORT]
"""

import os
import sys
import argparse
import uvicorn

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    """Main function to run the API server."""
    parser = argparse.ArgumentParser(description="Run the API server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()
    
    print(f"Starting API server on {args.host}:{args.port}")
    print(f"Documentation will be available at http://localhost:{args.port}/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
