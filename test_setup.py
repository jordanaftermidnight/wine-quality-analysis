#!/usr/bin/env python3
"""
Test script to validate the wine quality analysis setup.
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        'numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn',
        'plotly', 'joblib', 'imblearn', 'tqdm', 'yaml'
    ]
    
    print("Testing package imports...")
    failed = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            failed.append(package)
    
    if failed:
        print(f"\nMissing packages: {', '.join(failed)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("\nAll packages installed successfully!")
    return True

def test_file_structure():
    """Test if required files and directories exist."""
    import os
    
    print("\nTesting file structure...")
    required_items = [
        'data/winequality-red.csv',
        'utils/__init__.py',
        'utils/visualization.py',
        'wine_analysis_cli.py',
        'wine_quality_analysis.ipynb',
        'config.yaml'
    ]
    
    missing = []
    for item in required_items:
        if os.path.exists(item):
            print(f"✓ {item}")
        else:
            print(f"✗ {item} - NOT FOUND")
            missing.append(item)
    
    if missing:
        print(f"\nMissing files: {', '.join(missing)}")
        return False
    
    print("\nFile structure is complete!")
    return True

def test_cli():
    """Test if CLI script can be imported and help works."""
    print("\nTesting CLI script...")
    
    try:
        import wine_analysis_cli
        print("✓ CLI script imports successfully")
        
        # Test help command
        import subprocess
        result = subprocess.run([sys.executable, 'wine_analysis_cli.py', '--help'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ CLI help command works")
            return True
        else:
            print("✗ CLI help command failed")
            return False
            
    except Exception as e:
        print(f"✗ Error testing CLI: {e}")
        return False

def main():
    """Run all tests."""
    print("="*50)
    print("WINE QUALITY ANALYSIS - SETUP TEST")
    print("="*50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test file structure
    if not test_file_structure():
        all_passed = False
    
    # Test CLI
    if not test_cli():
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✓ ALL TESTS PASSED! Setup is complete.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("="*50)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())