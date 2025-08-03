from skripsi_code.model.MoMLNIDS import momlnids
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
import torch
from torch import nn, optim
from copy import deepcopy
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import numpy as np


def split_domain(domains, split_idx, print_domain=True):
    source_domain = deepcopy(domains)
    target_domain = [source_domain.pop(split_idx)]
    if print_domain:
        print("Source domain: ", end="")
        for domain in source_domain:
            print(domain, end=", ")
        print("Target domain: ", end="")
        for domain in target_domain:
            print(domain)
    return source_domain, target_domain


def get_model_learning_rate(
    model, extractor_weight=1.0, classifier_weight=1.0, discriminator_weight=1.0
):
    return [
        (model.FeatureExtractorLayer, 1.0 * extractor_weight),
        (model.DomainClassifier, 1.0 * discriminator_weight),
        (model.LabelClassifier, 1.0 * classifier_weight),
    ]


def get_optimizer(model, init_lr, weight_decay=0.01, amsgrad=True):
    if not amsgrad:
        return optim.SGD(
            model.parameters(),
            lr=init_lr,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    else:
        return optim.AdamW(
            model.parameters(), lr=init_lr, weight_decay=weight_decay, amsgrad=amsgrad
        )


def get_learning_rate_scheduler(optimizer, t_max=25):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)


def create_experiment_directory(
    base_dir="ProperTraining",
    target_domain=None,
    experiment_name=None,
    use_clustering=False,
    cluster_num=None,
    epochs=None,
    timestamp=None,
):
    """
    Create clean, organized experiment directory structure.

    New Format:
    Training_Sessions/ProperTraining_20Epochs_20250714_0930/NF-CSE-CIC-IDS2018-v2/Demo_Cluster4/

    Components:
    - Training_Sessions: Root directory for all training sessions
    - ProperTraining_20Epochs_20250714_0930: Session (Method_Epochs_Date_Time)
    - NF-CSE-CIC-IDS2018-v2: Target domain
    - Demo_Cluster4: Experiment name with cluster info

    Subdirectories created:
    - models/: Model checkpoints
    - logs/: Log files
    - results/: Analysis results
    """
    from pathlib import Path
    from datetime import datetime

    # Generate clean timestamp if not provided
    if timestamp is None:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M")  # Format: 20250714_0930

    # Create clear session directory name
    if epochs:
        session_name = f"{base_dir}_{epochs}Epochs_{timestamp}"
    else:
        session_name = f"{base_dir}_{timestamp}"

    # Create clean experiment name
    experiment_parts = []
    if experiment_name:
        experiment_parts.append(experiment_name)
    if use_clustering and cluster_num is not None:
        experiment_parts.append(f"Cluster{cluster_num}")

    if experiment_parts:
        experiment_dir_name = "_".join(experiment_parts)
    else:
        experiment_dir_name = "Default"

    # Create organized path structure
    full_path = (
        Path("Training_Sessions") / session_name / target_domain / experiment_dir_name
    )

    # Create directory structure with subdirectories
    full_path.mkdir(parents=True, exist_ok=True)

    # Create standard subdirectories for organization
    (full_path / "models").mkdir(exist_ok=True)
    (full_path / "logs").mkdir(exist_ok=True)
    (full_path / "results").mkdir(exist_ok=True)

    return full_path


def get_experiment_log_files(experiment_dir):
    """Get organized log file paths for an experiment directory."""
    from pathlib import Path

    experiment_dir = Path(experiment_dir)
    logs_dir = experiment_dir / "logs"

    return {
        "source_trained": logs_dir / "source_trained.log",
        "val_performance": logs_dir / "val_performance.log",
        "target_performance": logs_dir / "target_performance.log",
        "clustering": logs_dir / "clustering.log",
        "training": logs_dir / "training.log",
        "final_val": logs_dir / "final_val_performance.log",
        "final_target": logs_dir / "final_target_performance.log",
    }


def get_model_checkpoint_path(experiment_dir, epoch=None, is_best=False):
    """Get model checkpoint path in organized structure."""
    from pathlib import Path

    experiment_dir = Path(experiment_dir)
    models_dir = experiment_dir / "models"

    if is_best:
        return models_dir / "best_model.pth"
    elif epoch is not None:
        return models_dir / f"model_epoch_{epoch:03d}.pth"
    else:
        return models_dir / "model_latest.pth"


def demo_utils():
    """Demo function to test utility functions."""
    console = Console()

    console.print(Panel.fit("🔧 Utilities Demo", style="bold blue"))

    # Test domain splitting
    console.print("🌐 Testing domain splitting...")
    domains = ["Domain_A", "Domain_B", "Domain_C", "Domain_D"]

    for i in range(len(domains)):
        source, target = split_domain(domains, i, print_domain=False)
        console.print(f"Split {i}: Source={source}, Target={target}")

    # Test model creation and utilities
    console.print("\n🧠 Testing model utilities...")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"Device: {device}")

        # Create a test model
        model = momlnids(
            input_nodes=10,
            hidden_nodes=[64, 32, 16],
            classifier_nodes=[32, 16],
            num_domains=3,
            num_class=2,
            single_layer=False,
        ).to(device)

        console.print("✅ Model created successfully")

        # Test learning rate configuration
        lr_config = get_model_learning_rate(model, 1.0, 1.0, 0.5)
        console.print(f"✅ Learning rate config: {len(lr_config)} components")

        # Test optimizer creation
        optimizers = []
        for module, weight in lr_config:
            optimizer = get_optimizer(module, 0.001 * weight)
            optimizers.append(optimizer)

        console.print(f"✅ Created {len(optimizers)} optimizers")

        # Test scheduler creation
        schedulers = [get_learning_rate_scheduler(opt) for opt in optimizers]
        console.print(f"✅ Created {len(schedulers)} schedulers")

        # Create results table
        results_table = Table(title="Utility Functions Test Results")
        results_table.add_column("Function", style="cyan")
        results_table.add_column("Status", style="green")
        results_table.add_column("Details", style="yellow")

        results_table.add_row(
            "split_domain", "✅ Working", f"Tested with {len(domains)} domains"
        )
        results_table.add_row(
            "get_model_learning_rate", "✅ Working", f"{len(lr_config)} components"
        )
        results_table.add_row(
            "get_optimizer", "✅ Working", f"{len(optimizers)} optimizers created"
        )
        results_table.add_row(
            "get_learning_rate_scheduler",
            "✅ Working",
            f"{len(schedulers)} schedulers created",
        )

        console.print(results_table)

    except Exception as e:
        console.print(f"❌ Error in model testing: {e}")

    console.print("✨ Utilities demo completed!")


@click.command()
@click.option("--demo", is_flag=True, help="Run utilities demonstration")
@click.option("--test-domain", is_flag=True, help="Test domain splitting functionality")
@click.option(
    "--test-optimizer", is_flag=True, help="Test optimizer and scheduler creation"
)
def main(demo, test_domain, test_optimizer):
    """
    Test and demonstrate utility functions.
    """
    console = Console()

    if demo:
        demo_utils()
    elif test_domain:
        console.print(Panel.fit("🌐 Testing Domain Split", style="bold blue"))

        domains = ["NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v2", "NF-UNSW-NB15-v2"]

        for i, domain in enumerate(domains):
            console.print(f"\n--- Testing with {domain} as target ---")
            source, target = split_domain(domains, i)

    elif test_optimizer:
        console.print(Panel.fit("⚙️ Testing Optimizers", style="bold blue"))

        # Create dummy model
        dummy_model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))

        # Test different optimizers
        adam_opt = get_optimizer(dummy_model, 0.001, amsgrad=True)
        sgd_opt = get_optimizer(dummy_model, 0.001, amsgrad=False)

        # Test scheduler
        scheduler = get_learning_rate_scheduler(adam_opt)

        console.print(f"✅ AdamW optimizer: {type(adam_opt).__name__}")
        console.print(f"✅ SGD optimizer: {type(sgd_opt).__name__}")
        console.print(f"✅ Scheduler: {type(scheduler).__name__}")

    else:
        console.print("Use --demo, --test-domain, or --test-optimizer")
