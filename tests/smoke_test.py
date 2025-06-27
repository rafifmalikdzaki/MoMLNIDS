#!/usr/bin/env python3
"""
Smoke test to verify that all imports work and basic functionality is available.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all major modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        # Core torch imports
        import torch
        import torch.nn as nn
        print("✓ PyTorch imports successful")
        
        # Model imports
        from skripsi_code.model.MoMLNIDS import MoMLDNIDS
        from skripsi_code.model.FeatureExtractor import DGFeatExt
        from skripsi_code.model.Discriminator import DomainDiscriminator
        from skripsi_code.model.Classifier import ClassifierANN
        print("✓ Model imports successful")
        
        # Utils imports
        from skripsi_code.utils.dataloader import random_split_dataloader
        from skripsi_code.utils.domain_dataset import MultiChunkParquet, MultiChunkDataset, Whole_Dataset
        from skripsi_code.utils.loss import EntropyLoss, MaximumSquareLoss
        from skripsi_code.utils.utils import get_optimizer, get_learning_rate_scheduler
        print("✓ Utils imports successful")
        
        # Clustering imports  
        from skripsi_code.clustering.cluster_utils import pseudolabeling
        from skripsi_code.clustering.cluster_methods import MiniK, Kmeans, GMM, Spectral, Agglomerative
        print("✓ Clustering imports successful")
        
        # Training imports
        from skripsi_code.TrainEval.TrainEval import train, eval
        print("✓ TrainEval imports successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_model():
    """Test that the basic model can be instantiated and run a forward pass."""
    print("\nTesting basic model instantiation...")
    
    try:
        import torch
        from skripsi_code.model.MoMLNIDS import MoMLDNIDS
        
        # Create a small test tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create test data
        x = torch.randn(10, 43).to(device)  # Small batch for testing
        
        # Create model
        model = MoMLDNIDS(
            input_nodes=x.size(dim=1),
            hidden_nodes=[64, 32, 16, 10],
            classifier_nodes=[64, 32, 16],
            num_domains=3,
            num_class=2,
        ).to(device)
        
        print("✓ Model instantiated successfully")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            class_output, domain_output = model(x)
            
        print(f"✓ Forward pass successful")
        print(f"  - Class output shape: {class_output.shape}")
        print(f"  - Domain output shape: {domain_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """Test basic data loading functionality."""
    print("\nTesting data loading components...")
    
    try:
        # Test if we can import and create basic data structures
        from skripsi_code.utils.domain_dataset import MultiChunkParquet, MultiChunkDataset, Whole_Dataset
        from skripsi_code.utils.dataloader import random_split_dataloader
        
        print("✓ Data loading components imported successfully")
        
        # Note: We don't test actual data loading since we don't have data files
        # but we can verify the classes are properly defined
        
        assert True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        traceback.print_exc()
        assert False

def main():
    """Run all smoke tests."""
    print("🔥 Running smoke tests for skripsi_code project...")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Run tests
    if test_imports():
        success_count += 1
    
    if test_basic_model():
        success_count += 1
        
    if test_data_loading():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Smoke test results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All smoke tests passed! The environment is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
