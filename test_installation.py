#!/usr/bin/env python
"""
Test script to verify CausalShapGNN installation
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all imports work correctly"""
    print("Testing imports...")
    
    try:
        from data import DataDownloader, DataPreprocessor, GraphData
        print("  ✓ data module")
    except ImportError as e:
        print(f"  ✗ data module: {e}")
        return False
    
    try:
        from data import RecommendationDataset, BipartiteGraphProcessor, collate_fn
        print("  ✓ dataset classes")
    except ImportError as e:
        print(f"  ✗ dataset classes: {e}")
        return False
    
    try:
        import torch
        print(f"  ✓ torch (version {torch.__version__})")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✓ numpy (version {np.__version__})")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        import scipy
        print(f"  ✓ scipy (version {scipy.__version__})")
    except ImportError as e:
        print(f"  ✗ scipy: {e}")
        return False
    
    return True


def test_downloader():
    """Test data downloader"""
    print("\nTesting DataDownloader...")
    
    from data import DataDownloader
    
    downloader = DataDownloader('./data')
    downloader.list_datasets()
    
    return True


def test_model_creation():
    """Test model can be created"""
    print("\nTesting model creation...")
    
    try:
        import torch
        from models import CausalShapGNN
        
        config = {
            'n_users': 100,
            'n_items': 200,
            'embed_dim': 64,
            'n_factors': 4,
            'n_layers': 2,
            'temperature': 0.2,
            'alpha': 0.1,
            'beta': 0.1,
            'gamma': 0.1,
            'delta': 0.1,
            'reg_weight': 1e-5
        }
        
        device = torch.device('cpu')
        model = CausalShapGNN(config, device)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model created with {n_params:,} parameters")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False


def main():
    print("=" * 60)
    print("CausalShapGNN Installation Test")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_downloader()
    all_passed &= test_model_creation()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed. ✗")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())