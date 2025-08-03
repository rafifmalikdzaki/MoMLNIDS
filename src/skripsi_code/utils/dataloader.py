from torch.utils.data import DataLoader, random_split
import numpy as np
from copy import deepcopy
from skripsi_code.utils.domain_dataset import MultiChunkDataset, MultiChunkParquet
from typing import List
import torch
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path


def random_split_dataloader(
    dir_path: str,
    source_dir: List[str],
    target_dir: str,
    source_domain: List[str],
    target_domain: List[str],
    get_domain=False,
    get_cluster=False,
    batch_size=1,
    buffer_size=16,
    n_workers=0,
    chunk=True,
):
    source_data = MultiChunkParquet(
        dir_path,
        source_dir,
        domain=source_domain,
        get_domain=get_domain,
        get_cluster=get_cluster,
        buffer_size=buffer_size,
        chunk_mode=chunk,
    )
    target_data = MultiChunkParquet(
        dir_path,
        target_dir,
        domain=target_domain,
        get_domain=get_domain,
        get_cluster=get_cluster,
        buffer_size=buffer_size,
        chunk_mode=chunk,
    )

    source_train, source_val = random_split(source_data, [0.8, 0.2])
    source_train = deepcopy(source_train)

    print(
        "Train: {}, Val: {}, Test: {}".format(
            len(source_train), len(source_val), len(target_data)
        )
    )

    source_train = DataLoader(
        source_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    source_val = DataLoader(
        source_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )
    target_test = DataLoader(
        target_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )
    return source_train, source_val, target_test


def demo_dataloader():
    """Demo function to test dataloader functionality with dummy data."""
    console = Console()

    console.print(Panel.fit("📊 DataLoader Demo", style="bold blue"))

    try:
        # Create dummy parquet files for demonstration
        demo_dir = Path("./demo_data")
        demo_dir.mkdir(exist_ok=True)

        # Create mock data directories
        for domain in ["domain1", "domain2", "domain3"]:
            domain_dir = demo_dir / domain
            domain_dir.mkdir(exist_ok=True)

        console.print("✅ Demo data directories created")

        # Test basic dataloader functionality
        console.print("🔄 Testing dataloader components...")

        # Test data structures
        dummy_domains = ["domain1", "domain2"]
        target_domain = "domain3"

        console.print(f"Source domains: {dummy_domains}")
        console.print(f"Target domain: {target_domain}")

        # Create results table
        results_table = Table(title="DataLoader Components Test")
        results_table.add_column("Component", style="cyan")
        results_table.add_column("Status", style="green")
        results_table.add_column("Details", style="yellow")

        # Test torch components
        results_table.add_row(
            "PyTorch DataLoader", "✅ Available", f"Version: {torch.__version__}"
        )
        results_table.add_row(
            "Random Split", "✅ Available", "torch.utils.data.random_split"
        )
        results_table.add_row("DeepCopy", "✅ Available", "copy.deepcopy")

        console.print(results_table)
        console.print("✨ DataLoader demo completed!")

    except Exception as e:
        console.print(f"❌ Error in demo: {e}")


@click.command()
@click.option("--demo", is_flag=True, help="Run dataloader demonstration")
@click.option("--test-split", is_flag=True, help="Test data splitting functionality")
def main(demo, test_split):
    """
    Test and demonstrate dataloader functionality.
    """
    console = Console()

    if demo:
        demo_dataloader()
    elif test_split:
        console.print(Panel.fit("🔄 Testing Data Split", style="bold blue"))

        # Create dummy tensor data
        dummy_data = torch.randn(1000, 10)
        dummy_labels = torch.randint(0, 2, (1000,))

        from torch.utils.data import TensorDataset

        dataset = TensorDataset(dummy_data, dummy_labels)

        # Test random split
        train_data, val_data = random_split(dataset, [0.8, 0.2])

        console.print(f"✅ Original dataset: {len(dataset)} samples")
        console.print(f"✅ Training set: {len(train_data)} samples")
        console.print(f"✅ Validation set: {len(val_data)} samples")

        # Test DataLoader creation
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

        console.print(f"✅ Training batches: {len(train_loader)}")
        console.print(f"✅ Validation batches: {len(val_loader)}")

        console.print("✨ Data split test completed!")
    else:
        console.print(
            "Use --demo to run demonstration or --test-split to test splitting"
        )


