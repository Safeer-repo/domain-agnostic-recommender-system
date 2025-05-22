#!/usr/bin/env python3
"""
Script to debug model file discovery and counting.
"""

import os
import sys
import json

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def inspect_models_directory():
    """Check the models directory structure and files."""
    print("===== DEBUG: MODELS DIRECTORY STRUCTURE =====")
    
    # Define paths
    data_dir = os.path.join(project_root, "data")
    models_dir = os.path.join(project_root, "artifacts/models")  # Note: Using artifacts/models
    
    print(f"Models directory: {models_dir}")
    
    if not os.path.exists(models_dir):
        print(f"ERROR: Models directory does not exist: {models_dir}")
        return
    
    domains = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    print(f"Found domains: {domains}")
    
    total_models = 0
    
    for domain in domains:
        domain_dir = os.path.join(models_dir, domain)
        datasets = [d for d in os.listdir(domain_dir) if os.path.isdir(os.path.join(domain_dir, d))]
        
        print(f"\nDomain: {domain}")
        print(f"  Datasets: {datasets}")
        
        for dataset in datasets:
            dataset_dir = os.path.join(domain_dir, dataset)
            
            # Try to count files directly in the dataset directory
            direct_files = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]
            pkl_files = [f for f in direct_files if f.endswith('.pkl')]
            
            print(f"  Dataset: {dataset}")
            print(f"    Direct files: {len(direct_files)}")
            print(f"    PKL files: {len(pkl_files)}")
            print(f"    PKL file names: {pkl_files}")
            
            total_models += len(pkl_files)
            
            # Check for subdirectories (algorithms)
            algorithms = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
            
            if algorithms:
                print(f"    Algorithms: {algorithms}")
                
                for algorithm in algorithms:
                    algorithm_dir = os.path.join(dataset_dir, algorithm)
                    model_files = []
                    
                    # Check for multiple extensions
                    for ext in [".model", ".pkl", ".bin", ".h5", ".joblib"]:
                        model_files.extend([f for f in os.listdir(algorithm_dir) if f.endswith(ext)])
                    
                    print(f"    Algorithm: {algorithm}")
                    print(f"      Model files: {len(model_files)}")
                    print(f"      Model file names: {model_files}")
                    
                    total_models += len(model_files)
    
    print(f"\nTotal models found: {total_models}")

def debug_get_system_stats():
    """Debug the get_system_stats function from app.py."""
    print("\n===== DEBUG: GET_SYSTEM_STATS LOGIC =====")
    
    # Define paths as they are in app.py
    data_dir = os.path.join(project_root, "data")
    models_dir = os.path.join(data_dir, "models")  # Note: Using data/models
    
    print(f"Stats function models directory: {models_dir}")
    print(f"Does this directory exist? {os.path.exists(models_dir)}")
    
    # The rest of the logic from get_system_stats
    if not os.path.exists(models_dir):
        print("No models directory found")
        return
    
    domains = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    print(f"Found domains: {domains}")
    
    # Rest of the function would go here...

if __name__ == "__main__":
    inspect_models_directory()
    debug_get_system_stats()