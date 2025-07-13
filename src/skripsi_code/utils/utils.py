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
    return CosineAnnealingLR(optimizer, t_max, eta_min=1e-5)


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


