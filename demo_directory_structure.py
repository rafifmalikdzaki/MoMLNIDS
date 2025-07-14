#!/usr/bin/env python3
"""
Demo: Directory Structure for MoMLNIDS Training

This script demonstrates the new structured directory creation system
for organizing training results, logs, and model checkpoints.
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

# Add project root to path
project_root = Path(__file__).resolve().parents[0]
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from skripsi_code.utils.utils import (
    create_experiment_directory,
    get_experiment_log_files,
    get_model_checkpoint_path,
)


def demo_directory_structure():
    """Demonstrate the directory structure creation."""
    console = Console()

    console.print(Panel.fit("📁 MoMLNIDS Directory Structure Demo", style="bold blue"))

    # Demo 1: Basic experiment directory
    console.print("\n🔧 [bold cyan]Creating Basic Experiment Directory[/bold cyan]")

    basic_dir = create_experiment_directory(
        base_dir="ProperTraining",
        target_domain="NF-CSE-CIC-IDS2018-v2",
        experiment_name="Demo",
        use_clustering=False,
        epochs=10,
    )

    console.print(f"✅ Created: {basic_dir}")

    # Demo 2: Clustering experiment directory
    console.print(
        "\n🔧 [bold cyan]Creating Clustering Experiment Directory[/bold cyan]"
    )

    cluster_dir = create_experiment_directory(
        base_dir="ProperTraining",
        target_domain="NF-ToN-IoT-v2",
        experiment_name="PseudoLabelling",
        use_clustering=True,
        cluster_num=4,
        epochs=50,
    )

    console.print(f"✅ Created: {cluster_dir}")

    # Demo 3: Custom timestamp experiment
    console.print("\n🔧 [bold cyan]Creating Custom Timestamp Experiment[/bold cyan]")

    custom_dir = create_experiment_directory(
        base_dir="Training_Results",
        target_domain="NF-UNSW-NB15-v2",
        experiment_name="Enhanced",
        use_clustering=True,
        cluster_num=3,
        epochs=25,
        timestamp="12M13M4M",
    )

    console.print(f"✅ Created: {custom_dir}")

    # Demo 4: Show log files structure
    console.print("\n📋 [bold cyan]Log Files Structure[/bold cyan]")

    log_files = get_experiment_log_files(basic_dir)

    log_table = Table(title="Standard Log Files")
    log_table.add_column("Log Type", style="cyan")
    log_table.add_column("File Path", style="green")

    for log_type, log_path in log_files.items():
        log_table.add_row(log_type, str(log_path))

    console.print(log_table)

    # Demo 5: Show model checkpoint paths
    console.print("\n💾 [bold cyan]Model Checkpoint Paths[/bold cyan]")

    checkpoint_table = Table(title="Model Checkpoints")
    checkpoint_table.add_column("Checkpoint Type", style="cyan")
    checkpoint_table.add_column("File Path", style="green")

    # Best model
    best_model = get_model_checkpoint_path(basic_dir, is_best=True)
    checkpoint_table.add_row("Best Model", str(best_model))

    # Epoch checkpoints
    for epoch in [0, 2, 4, 6, 8]:
        epoch_model = get_model_checkpoint_path(basic_dir, epoch=epoch)
        checkpoint_table.add_row(f"Epoch {epoch}", str(epoch_model))

    console.print(checkpoint_table)

    # Demo 6: Show directory tree structure
    console.print("\n🌳 [bold cyan]Generated Directory Tree[/bold cyan]")

    # Create the tree visualization
    tree = Tree("📁 MoMLNIDS Experiments")

    # Add ProperTraining directories
    proper_training = tree.add("📁 ProperTraining10Epoch...")
    nf_cse = proper_training.add("📁 NF-CSE-CIC-IDS2018-v2")
    demo_exp = nf_cse.add("📁 NF-CSE-CIC-IDS2018-v2_N|Demo")
    demo_exp.add("📄 source_trained.log")
    demo_exp.add("📄 val_performance.log")
    demo_exp.add("📄 target_performance.log")
    demo_exp.add("📄 clustering.log")
    demo_exp.add("💾 model_best.pt")
    demo_exp.add("💾 model_0.pt")
    demo_exp.add("💾 model_2.pt")
    demo_exp.add("💾 model_4.pt")

    # Add ProperTraining50Epoch
    proper_50 = tree.add("📁 ProperTraining50Epoch...")
    nf_ton = proper_50.add("📁 NF-ToN-IoT-v2")
    pseudo_exp = nf_ton.add("📁 NF-ToN-IoT-v2_N|PseudoLabelling|Cluster_4")
    pseudo_exp.add("📄 clustering.log")
    pseudo_exp.add("📄 source_trained.log")
    pseudo_exp.add("📄 val_performance.log")
    pseudo_exp.add("📄 target_performance.log")
    pseudo_exp.add("💾 model_best.pt")
    for i in range(0, 50, 2):
        pseudo_exp.add(f"💾 model_{i}.pt")

    # Add Training_Results
    training_results = tree.add("📁 Training_Results25Epoch12M13M4M")
    nf_unsw = training_results.add("📁 NF-UNSW-NB15-v2")
    enhanced_exp = nf_unsw.add("📁 NF-UNSW-NB15-v2_N|Enhanced|Cluster_3")
    enhanced_exp.add("📄 source_trained.log")
    enhanced_exp.add("📄 val_performance.log")
    enhanced_exp.add("📄 target_performance.log")
    enhanced_exp.add("📄 clustering.log")
    enhanced_exp.add("💾 model_best.pt")

    console.print(tree)

    # Demo 7: Show naming convention examples
    console.print("\n📝 [bold cyan]Naming Convention Examples[/bold cyan]")

    naming_table = Table(title="Directory Naming Convention")
    naming_table.add_column("Component", style="cyan")
    naming_table.add_column("Pattern", style="yellow")
    naming_table.add_column("Example", style="green")

    naming_table.add_row(
        "Base Directory",
        "[base_dir][epochs]Epoch[timestamp]",
        "ProperTraining50Epoch12M13M4M",
    )
    naming_table.add_row("Target Domain", "[target_domain]/", "NF-CSE-CIC-IDS2018-v2/")
    naming_table.add_row(
        "Experiment",
        "[target]_N|[experiment]|[cluster]",
        "NF-CSE-CIC-IDS2018-v2_N|PseudoLabelling|Cluster_4",
    )
    naming_table.add_row(
        "Log Files", "*.log", "source_trained.log, val_performance.log"
    )
    naming_table.add_row(
        "Models", "model_[epoch].pt", "model_0.pt, model_2.pt, model_best.pt"
    )

    console.print(naming_table)

    console.print("\n✨ [bold green]Directory structure demo completed![/bold green]")
    console.print(f"📁 View created directories at: {Path.cwd()}")


if __name__ == "__main__":
    demo_directory_structure()
