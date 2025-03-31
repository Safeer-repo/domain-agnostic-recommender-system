#!/usr/bin/env python3
"""
Script to verify that all required packages are installed and working.
"""

import sys
import importlib
import pkg_resources

def check_package(package_name):
    try:
        importlib.import_module(package_name)
        version = pkg_resources.get_distribution(package_name).version
        return True, version
    except (ImportError, pkg_resources.DistributionNotFound):
        return False, None

# List of essential packages for the project
essential_packages = [
    'pandas',
    'numpy',
    'scipy',
    'scikit-learn',
    'implicit',
    'fastapi',
    'uvicorn',
    'pydantic',
    'tqdm',
    'matplotlib',
    'seaborn',
    'yaml'  # PyYAML module is imported as 'yaml'
]

# Optional packages
optional_packages = [
    'torch',
    'lightfm',  # We skipped this but let's check anyway
    'pyarrow',
    'jupyter'
]

if __name__ == "__main__":
    print("Verifying Python environment for Domain-Agnostic Recommender System\n")
    print(f"Python version: {sys.version}\n")
    
    # Check essential packages
    print("Essential packages:")
    all_essential_installed = True
    for package in essential_packages:
        installed, version = check_package(package)
        status = f"✅ {version}" if installed else "❌ Not installed"
        print(f"  {package:<15} {status}")
        if not installed:
            all_essential_installed = False
    
    # Check optional packages
    print("\nOptional packages:")
    for package in optional_packages:
        installed, version = check_package(package)
        status = f"✅ {version}" if installed else "❌ Not installed"
        print(f"  {package:<15} {status}")
    
    # Overall status
    print("\nEnvironment status:")
    if all_essential_installed:
        print("✅ All essential packages are installed and ready to use!")
        print("👍 You can proceed with data ingestion and preprocessing.")
    else:
        print("❌ Some essential packages are missing. Please install them before proceeding.")
