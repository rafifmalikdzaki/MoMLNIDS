
import pytest
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

# Assuming skripsi_code is in PYTHONPATH
from skripsi_code.utils.dataloader import random_split_dataloader
from skripsi_code.utils.domain_dataset import MultiChunkParquet

# Define paths relative to the project root for testing purposes
# These paths should ideally point to small, dummy data for quick tests
TEST_DATA_PATH = Path(__file__).parent.parent / "data" / "parquet"

# Create dummy parquet files for testing
# This is a simplified version. In a real scenario, you'd generate actual small parquet files.
# For now, we'll assume these directories exist and contain some dummy data.
# If they don't exist, the test will fail, indicating a need for dummy data generation.

# Dummy data setup (replace with actual small parquet files if needed)
# For a robust test, you'd create these files programmatically or have them in your test fixtures.
# For this exercise, we'll rely on the existing directory structure and assume some files are there.
# If the actual data is large, this test might be slow or fail due to memory.
# The user mentioned data is in @data/**, so we'll use that.

# Let's assume a minimal set of domains for testing
# These should correspond to actual subdirectories in data/parquet/
# Based on the file structure, these are good candidates:
# NF-BoT-IoT-v2, NF-CSE-CIC-IDS2018-v2, NF-ToN-IoT-v2, NF-UNSW-NB15-v2
TEST_SOURCE_DOMAINS = ["NF-BoT-IoT-v2", "NF-CSE-CIC-IDS2018-v2"]
TEST_TARGET_DOMAIN = "NF-ToN-IoT-v2"

@pytest.fixture(scope="module")
def dummy_parquet_data():
    """
    Fixture to ensure dummy parquet data exists for testing.
    In a real project, this would generate small, valid parquet files.
    For this exercise, we'll just check for the existence of the directories.
    """
    for domain in TEST_SOURCE_DOMAINS + [TEST_TARGET_DOMAIN]:
        domain_path = TEST_DATA_PATH / domain
        if not domain_path.exists():
            pytest.skip(f"Test data directory not found: {domain_path}. Please ensure dummy parquet data is available.")
        # Further check: ensure there's at least one .parquet file in each directory
        if not any(domain_path.glob("*.parquet")):
            pytest.skip(f"No parquet files found in {domain_path}. Please ensure dummy parquet data is available.")
    return True

def test_multichunkparquet_initialization(dummy_parquet_data):
    """Test MultiChunkParquet dataset initialization."""
    if not dummy_parquet_data:
        pytest.skip("Dummy parquet data not available.")

    dataset = MultiChunkParquet(
        dir_path=str(TEST_DATA_PATH),
        directories=TEST_SOURCE_DOMAINS,
        domain="test_domain", # This can be a placeholder
        get_domain=True,
        get_cluster=True,
        chunk_mode=True,
        buffer_size=2
    )
    assert isinstance(dataset, MultiChunkParquet)
    assert dataset.chunk_count > 0
    assert dataset.length > 0
    assert dataset.domain_label is not None
    assert dataset.cluster_label is not None

def test_multichunkparquet_getitem(dummy_parquet_data):
    """Test __getitem__ method of MultiChunkParquet dataset."""
    if not dummy_parquet_data:
        pytest.skip("Dummy parquet data not available.")

    dataset = MultiChunkParquet(
        dir_path=str(TEST_DATA_PATH),
        directories=TEST_SOURCE_DOMAINS,
        domain="test_domain",
        get_domain=True,
        get_cluster=True,
        chunk_mode=True,
        buffer_size=2
    )
    
    # Test fetching a single item (chunk)
    features, labels, domain_labels, cluster_labels = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(domain_labels, np.ndarray)
    assert isinstance(cluster_labels, np.ndarray)
    assert features.shape[0] > 0 # Ensure chunk is not empty
    assert labels.shape[0] > 0
    assert domain_labels.shape[0] > 0
    assert cluster_labels.shape[0] > 0
    assert features.dtype == torch.float32
    assert labels.dtype == torch.int64

def test_random_split_dataloader(dummy_parquet_data):
    """Test random_split_dataloader function."""
    if not dummy_parquet_data:
        pytest.skip("Dummy parquet data not available.")

    source_train_loader, source_val_loader, target_test_loader = random_split_dataloader(
        dir_path=str(TEST_DATA_PATH),
        source_dir=TEST_SOURCE_DOMAINS,
        target_dir=TEST_TARGET_DOMAIN,
        source_domain=TEST_SOURCE_DOMAINS,
        target_domain=[TEST_TARGET_DOMAIN],
        get_domain=True,
        get_cluster=True,
        batch_size=16,
        n_workers=0, # Use 0 workers for easier debugging in tests
        chunk=True
    )

    assert isinstance(source_train_loader, DataLoader)
    assert isinstance(source_val_loader, DataLoader)
    assert isinstance(target_test_loader, DataLoader)

    # Test if loaders are not empty (by trying to get one batch)
    try:
        train_features, train_labels, train_domain, train_cluster = next(iter(source_train_loader))
        assert train_features.shape[0] > 0
        assert train_labels.shape[0] > 0
        assert train_domain.shape[0] > 0
        assert train_cluster.shape[0] > 0
    except StopIteration:
        pytest.fail("Source train dataloader is empty.")

    try:
        val_features, val_labels, val_domain, val_cluster = next(iter(source_val_loader))
        assert val_features.shape[0] > 0
        assert val_labels.shape[0] > 0
        assert val_domain.shape[0] > 0
        assert val_cluster.shape[0] > 0
    except StopIteration:
        pytest.fail("Source validation dataloader is empty.")

    try:
        test_features, test_labels, test_domain, test_cluster = next(iter(target_test_loader))
        assert test_features.shape[0] > 0
        assert test_labels.shape[0] > 0
        # For target, domain and cluster labels might not be explicitly returned by default
        # depending on the implementation of MultiChunkParquet for target_data.
        # If they are expected, add assertions here.
    except StopIteration:
        pytest.fail("Target test dataloader is empty.")

    # Test data types and shapes for a batch
    assert train_features.dtype == torch.float32
    assert train_labels.dtype == torch.int64
    assert train_features.ndim == 3 # (batch_size, chunk_size, num_features)
    assert train_labels.ndim == 2 # (batch_size,)
    assert train_domain.ndim == 2 # (batch_size,)
    assert train_cluster.ndim == 2 # (batch_size,)
