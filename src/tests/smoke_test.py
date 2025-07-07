#!/usr/bin/env python3
"""
Smoke test to verify that all imports work and basic functionality is available.
"""

import sys
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test that all major modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        # Core torch imports
        import torch
        import torch.nn as nn
        print("✓ PyTorch imports successful")
        
        # Model imports
        from skripsi_code.model.MoMLNIDS import momlnids
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
        from skripsi_code.model.MoMLNIDS import momlnids
        
        # Create a small test tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create test data
        x = torch.randn(10, 43).to(device)  # Small batch for testing
        
        # Create model
        model = momlnids(
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
        from skripsi_code.utils.dataloader import random_split_dataloader
        from skripsi_code.utils.utils import split_domain
        
        print("✓ Data loading components imported successfully")
        
        # Define parameters similar to main.py
        DOMAIN_LIST = [
            "NF-UNSW-NB15-v2",
            "NF-CSE-CIC-IDS2018-v2",
            "NF-ToN-IoT-v2",
            "NF-BoT-IoT-v2",
        ]
        TARGET_INDEX = 0  # Use the first domain as target for smoke test
        BATCH_SIZE = 1
        USE_CLUSTER = True
        USE_DOMAIN = not USE_CLUSTER
        DATA_PATH = str(project_root / "data" / "parquet")

        print(f"Attempting to load data from: {DATA_PATH}")
        
        source_domain, target_domain = split_domain(DOMAIN_LIST, TARGET_INDEX)

        source_train, source_val, target_test = random_split_dataloader(
            dir_path=DATA_PATH,
            source_dir=source_domain,
            target_dir=target_domain,
            source_domain=source_domain,
            target_domain=target_domain,
            batch_size=BATCH_SIZE,
            get_cluster=USE_CLUSTER,
            get_domain=USE_DOMAIN,
            chunk=True,
            n_workers=0,
        )
        
        print("✓ Dataloaders initialized successfully")

        # Verify dataloaders are not empty
        if len(source_train.dataset) > 0 and len(source_val.dataset) > 0 and len(target_test.dataset) > 0:
            print(f"✓ Data loaded successfully. Source train samples: {len(source_train.dataset)}, Source val samples: {len(source_val.dataset)}, Target test samples: {len(target_test.dataset)}")
        else:
            print("✗ One or more datasets are empty.")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        traceback.print_exc()
        return False

def test_clustering_utils():
    """Test basic clustering utilities functionality."""
    print("\nTesting clustering utilities...")

    try:
        import torch
        import numpy as np
        from torch.utils.data import Dataset, DataLoader
        from skripsi_code.model.MoMLNIDS import momlnids
        from skripsi_code.clustering.cluster_utils import pseudolabeling

        # Dummy Dataset for testing
        class DummyDataset(Dataset):
            def __init__(self, num_samples=100, num_features=39, num_domains=3, num_classes=2):
                self.features = torch.randn(num_samples, num_features).double()
                self.domain_label = np.random.randint(0, num_domains, num_samples)
                self.cluster_label = np.random.randint(0, num_classes, num_samples)
                self.length = num_samples

            def __len__(self):
                return self.length

            def __getitem__(self, idx):
                return self.features[idx], self.domain_label[idx], self.cluster_label[idx]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Create dummy model
        model = momlnids(
            input_nodes=39,
            hidden_nodes=[64, 32, 16, 10],
            classifier_nodes=[64, 32, 16],
            num_domains=3,
            num_class=2,
        ).double().to(device)
        model.eval()

        # Create dummy dataset
        dummy_dataset = DummyDataset()

        # Create a dummy log file path
        log_file_path = Path("temp_clustering_log.txt")
        if log_file_path.exists():
            log_file_path.unlink()

        # Call pseudolabeling
        print("Calling pseudolabeling...")
        new_cluster_labels = pseudolabeling(
            dataset=dummy_dataset,
            model=model,
            device=device,
            previous_cluster=dummy_dataset.cluster_label,
            log_file=str(log_file_path),
            epoch=1,
            n_clusters=2,
            method="Kmeans",
            batch_size=16,
            num_workers=0,
        )

        # Assertions
        assert isinstance(new_cluster_labels, np.ndarray)
        assert len(new_cluster_labels) == len(dummy_dataset)
        print("✓ Pseudolabeling executed successfully and returned expected type.")

        return True

    except Exception as e:
        print(f"✗ Clustering utilities test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    print("🔥 Running smoke tests for skripsi_code project...")
    print("=" * 50)
    
    success_count = 0
    total_tests = 4 # Updated total tests
    
    # Run tests
    if test_imports():
        success_count += 1
    
    if test_basic_model():
        success_count += 1
        
    if test_data_loading():
        success_count += 1

    if test_clustering_utils(): # New test
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
