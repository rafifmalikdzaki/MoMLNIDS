#!/usr/bin/env python3
"""
Smoke test to verify that all imports work and basic functionality is available.
"""

import sys
import traceback
from pathlib import Path
from collections import Counter # Corrected typo: Collections -> collections

from rich.console import Console
from rich.theme import Theme

# Define a custom theme for better output
custom_theme = Theme({
    "success": "green",
    "failure": "red",
    "info": "blue",
    "warning": "yellow",
    "header": "bold magenta",
    "test_name": "bold cyan",
})
console = Console(theme=custom_theme)

project_root = Path(__file__).resolve().parents[2]
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test that all major modules can be imported successfully."""
    console.print("[header]Testing imports...[/header]")
    
    try:
        # Core torch imports
        import torch
        import torch.nn as nn
        console.print("[success]✓ PyTorch imports successful[/success]")
        
        # Model imports
        from skripsi_code.model.MoMLNIDS import momlnids
        from skripsi_code.model.FeatureExtractor import DGFeatExt
        from skripsi_code.model.Discriminator import DomainDiscriminator
        from skripsi_code.model.Classifier import ClassifierANN
        console.print("[success]✓ Model imports successful[/success]")
        
        # Utils imports
        from skripsi_code.utils.dataloader import random_split_dataloader
        from skripsi_code.utils.domain_dataset import MultiChunkParquet, MultiChunkDataset, Whole_Dataset
        from skripsi_code.utils.loss import EntropyLoss, MaximumSquareLoss
        from skripsi_code.utils.utils import get_optimizer, get_learning_rate_scheduler
        console.print("[success]✓ Utils imports successful[/success]")
        
        # Clustering imports  
        from skripsi_code.clustering.cluster_utils import pseudolabeling
        from skripsi_code.clustering.cluster_methods import MiniK, Kmeans, GMM, Spectral, Agglomerative
        console.print("[success]✓ Clustering imports successful[/success]")
        
        # Training imports
        from skripsi_code.TrainEval.TrainEval import train, eval
        console.print("[success]✓ TrainEval imports successful[/success]")
        
        return True
        
    except Exception as e:
        console.print(f"[failure]✗ Import failed: {e}[/failure]")
        traceback.print_exc()
        return False

def test_basic_model():
    """Test that the basic model can be instantiated and run a forward pass."""
    console.print("\n[header]Testing basic model instantiation...[/header]")
    
    try:
        import torch
        from skripsi_code.model.MoMLNIDS import momlnids
        
        # Create a small test tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"[info]Using device: {device}[/info]")
        
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
        
        console.print("[success]✓ Model instantiated successfully[/success]")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            class_output, domain_output = model(x)
            
        console.print(f"[success]✓ Forward pass successful[/success]")
        console.print(f"  [info]- Class output shape: {class_output.shape}[/info]")
        console.print(f"  [info]- Domain output shape: {domain_output.shape}[/info]")
        
        return True
        
    except Exception as e:
        console.print(f"[failure]✗ Model test failed: {e}[/failure]")
        traceback.print_exc()
        return False

def test_data_loading():
    """Test basic data loading functionality."""
    console.print("\n[header]Testing data loading components...[/header]")
    
    try:
        # Test if we can import and create basic data structures
        from skripsi_code.utils.dataloader import random_split_dataloader
        from skripsi_code.utils.utils import split_domain
        
        console.print("[success]✓ Data loading components imported successfully[/success]")
        
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

        console.print(f"[info]Attempting to load data from: {DATA_PATH}[/info]")
        
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
        
        console.print("[success]✓ Dataloaders initialized successfully[/success]")

        # Verify dataloaders are not empty
        if len(source_train.dataset) > 0 and len(source_val.dataset) > 0 and len(target_test.dataset) > 0:
            console.print(f"[success]✓ Data loaded successfully. Source train samples: {len(source_train.dataset)}, Source val samples: {len(source_val.dataset)}, Target test samples: {len(target_test.dataset)}[/success]")
        else:
            console.print("[failure]✗ One or more datasets are empty.[/failure]")
            return False
            
        return True
        
    except Exception as e:
        console.print(f"[failure]✗ Data loading test failed: {e}[/failure]")
        traceback.print_exc()
        return False

def test_clustering_utils():
    """Test basic clustering utilities functionality."""
    console.print("\n[header]Testing clustering utilities...[/header]")

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
        console.print(f"[info]Using device: {device}[/info]")

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
        console.print("[info]Calling pseudolabeling...[/info]")
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
        console.print("[success]✓ Pseudolabeling executed successfully and returned expected type.[/success]")

        return True

    except Exception as e:
        console.print(f"[failure]✗ Clustering utilities test failed: {e}[/failure]")
        traceback.print_exc()
        return False



def main():
    """Run all smoke tests."""
    console.print("[header]🔥 Running smoke tests for skripsi_code project...[/header]")
    console.print("[header]=" * 50 + "[/header]")
    
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
    
    console.print("\n" + "[header]=" * 50 + "[/header]")
    console.print(f"[header]Smoke test results: {success_count}/{total_tests} tests passed[/header]")
    
    if success_count == total_tests:
        console.print("[success]🎉 All smoke tests passed! The environment is ready.[/success]")
        return 0
    else:
        console.print("[failure]❌ Some tests failed. Check the output above for details.[/failure]")
        return 1

if __name__ == "__main__":
    sys.exit(main())
