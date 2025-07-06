#!/usr/bin/env python3
"""
Enhanced MoMLNIDS Demo Script with Click, Rich, and OmegaConf
Features: Beautiful CLI, Model Testing, Flexible Configuration
"""

import click
import torch
import numpy as np
import yaml
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from omegaconf import DictConfig, OmegaConf

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich.prompt import Confirm, Prompt
from rich.layout import Layout
from rich.live import Live
from rich import box

# Initialize Rich console
console = Console()

# Default configuration with small batches for fast testing
DEFAULT_CONFIG = {
    "project": {"name": "MoMLNIDS_Demo"},
    "wandb": {"enabled": False},
    "training": {
        "batch_size": 16,  # Small batch size for fast testing
        "epochs": 1,
        "learning_rate": 0.001,
        "target_index": 0,
        "experiment_num": "|Demo|",
        "use_cluster": False,
        "eval_step": 1,
        "save_step": 1,
        "max_batches": 5,  # Very small for demo
    },
    "data": {"processed_data_path": "./skripsi_code/data/parquet/"},
    "device": {"num_workers": 0},
}


class DemoRunner:
    """Enhanced demo runner with rich UI and comprehensive functionality."""

    def __init__(self):
        self.console = console
        self.results = {}

    def run_command_with_progress(
        self, command: str, description: str, timeout: int = 300
    ) -> tuple[bool, List[str]]:
        """Run command with rich progress bar."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(description, total=None)

            try:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )

                output_lines = []
                start_time = time.time()

                for line in process.stdout:
                    output_lines.append(line.strip())
                    progress.update(
                        task, description=f"{description} - {line.strip()[:50]}..."
                    )

                    if time.time() - start_time > timeout:
                        process.terminate()
                        self.console.print(
                            f"[red]Command timed out after {timeout}s[/red]"
                        )
                        return False, output_lines

                process.wait()

                if process.returncode == 0:
                    progress.update(task, description=f"✅ {description} - Completed")
                    return True, output_lines
                else:
                    progress.update(task, description=f"❌ {description} - Failed")
                    return False, output_lines

            except Exception as e:
                self.console.print(f"[red]Error running command: {e}[/red]")
                return False, []

    def test_environment(self) -> Dict[str, bool]:
        """Test environment with rich formatting."""
        self.console.print(Panel("🔧 Testing Environment Setup", style="blue"))

        tests = [
            ("Python imports", "python tests/test_imports.py"),
            ("Smoke test", "python tests/smoke_test.py"),
            (
                "Data loader",
                "python -c \"from src.skripsi_code.utils.dataloader import random_split_dataloader; print('✅ Data loader working')\"",
            ),
            (
                "Model import",
                "python -c \"from src.skripsi_code.model.MoMLNIDS import MoMLDNIDS; print('✅ Model import working')\"",
            ),
        ]

        results = {}
        table = Table(title="Environment Tests")
        table.add_column("Test", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")

        for test_name, command in tests:
            success, output = self.run_command_with_progress(
                command, f"Running {test_name}", timeout=60
            )
            results[test_name] = success

            status = "✅ PASS" if success else "❌ FAIL"
            details = "OK" if success else "Check logs"
            table.add_row(test_name, status, details)

        self.console.print(table)
        return results

    def test_explainability(self) -> bool:
        """Test explainable AI with rich output."""
        self.console.print(Panel("🧠 Testing Explainable AI Features", style="magenta"))

        explainability_code = """
import torch
import numpy as np
from src.skripsi_code.model.MoMLNIDS import MoMLDNIDS
from src.skripsi_code.explainability.explainer import ModelExplainer

# Create model
model = MoMLDNIDS(
    input_nodes=39,
    hidden_nodes=[64, 32, 16, 10],
    classifier_nodes=[64, 32, 16],
    num_domains=3,
    num_class=2,
    single_layer=True
).double()

# Test explainer
feature_names = [f'feature_{i}' for i in range(39)]
explainer = ModelExplainer(model, feature_names)

# Test explanation
test_instance = np.random.randn(39)
explanation = explainer.explain_instance(test_instance, method='feature_ablation')
print("✅ Explainability test passed!")
print(f"Explanation keys: {list(explanation.keys())}")
"""

        with open("temp_explainability_test.py", "w") as f:
            f.write(explainability_code)

        success, output = self.run_command_with_progress(
            "python temp_explainability_test.py",
            "Testing explainability features",
            timeout=120,
        )

        Path("temp_explainability_test.py").unlink(missing_ok=True)

        if success:
            self.console.print("✅ [green]Explainability test passed![/green]")
        else:
            self.console.print("❌ [red]Explainability test failed![/red]")

        return success

    def test_model_loading(self, model_path: str) -> bool:
        """Test loading a trained model."""
        self.console.print(
            Panel(f"🔄 Testing Model Loading: {model_path}", style="yellow")
        )

        if not Path(model_path).exists():
            self.console.print(f"[red]Model file not found: {model_path}[/red]")
            return False

        try:
            with self.console.status("Loading model..."):
                # Load model state dict
                state_dict = torch.load(model_path, map_location="cpu")

                # Create model instance (you may need to adjust these parameters)
                model = torch.load(model_path, map_location="cpu")

                self.console.print("✅ [green]Model loaded successfully![/green]")

                # Display model info
                table = Table(title="Model Information")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="green")

                if hasattr(model, "state_dict"):
                    num_params = sum(p.numel() for p in model.parameters())
                    table.add_row("Total Parameters", f"{num_params:,}")
                    table.add_row("Model Type", str(type(model).__name__))
                    table.add_row("Device", str(next(model.parameters()).device))

                self.console.print(table)
                return True

        except Exception as e:
            self.console.print(f"[red]Error loading model: {e}[/red]")
            return False

    def run_inference_test(
        self, model_path: str, data_path: Optional[str] = None
    ) -> bool:
        """Run inference test on a trained model."""
        self.console.print(Panel("🎯 Running Inference Test", style="green"))

        inference_code = f'''
import torch
import numpy as np
from pathlib import Path

# Load model
model_path = "{model_path}"
try:
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    # Create dummy test data (replace with real data if available)
    test_data = torch.randn(10, 39).double()  # Batch of 10 samples
    
    with torch.no_grad():
        if hasattr(model, '__call__'):
            output = model(test_data)
            if isinstance(output, tuple):
                class_output, domain_output = output
                print(f"✅ Inference successful!")
                print(f"Class output shape: {{class_output.shape}}")
                print(f"Domain output shape: {{domain_output.shape}}")
                print(f"Sample predictions: {{class_output.argmax(dim=1)[:5].tolist()}}")
            else:
                print(f"✅ Inference successful!")
                print(f"Output shape: {{output.shape}}")
        else:
            print("❌ Model is not callable")
            
except Exception as e:
    print(f"❌ Inference failed: {{e}}")
'''

        with open("temp_inference_test.py", "w") as f:
            f.write(inference_code)

        success, output = self.run_command_with_progress(
            "python temp_inference_test.py", "Running inference test", timeout=60
        )

        Path("temp_inference_test.py").unlink(missing_ok=True)
        return success

    def create_quick_config(self, config_overrides: Dict[str, Any] = None) -> str:
        """Create a quick test configuration."""
        config = OmegaConf.create(DEFAULT_CONFIG)

        if config_overrides:
            config = OmegaConf.merge(config, OmegaConf.create(config_overrides))

        config_path = "config/quick_demo_config.yaml"
        Path("config").mkdir(exist_ok=True)

        with open(config_path, "w") as f:
            OmegaConf.save(config, f)

        return config_path

    def run_quick_training(self, max_batches: int = 5, epochs: int = 1) -> bool:
        """Run quick training test."""
        self.console.print(
            Panel(
                f"🚀 Quick Training Test ({epochs} epoch, {max_batches} batches)",
                style="blue",
            )
        )

        config_overrides = {
            "training": {
                "epochs": epochs,
                "max_batches": max_batches,
                "batch_size": 16,
            }
        }

        config_path = self.create_quick_config(config_overrides)

        success, output = self.run_command_with_progress(
            f"python src/main_config.py --config {config_path}",
            "Running quick training",
            timeout=600,
        )

        return success

    def display_results_summary(self, results: Dict[str, Any]):
        """Display comprehensive results summary with rich formatting."""

        # Create main layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        # Header
        header_text = Text("🎉 MoMLNIDS Demo Results Summary", style="bold magenta")
        layout["header"].update(Panel(header_text, style="bright_blue"))

        # Body - Results table
        table = Table(title="Test Results", box=box.ROUNDED)
        table.add_column("Test Category", style="cyan", width=20)
        table.add_column("Status", style="bold", width=10)
        table.add_column("Details", style="dim", width=30)

        overall_success = True

        for category, result in results.items():
            if isinstance(result, dict):
                # Environment tests
                success_count = sum(1 for v in result.values() if v)
                total_count = len(result)
                status = "✅ PASS" if success_count == total_count else "⚠️ PARTIAL"
                details = f"{success_count}/{total_count} tests passed"

                if success_count != total_count:
                    overall_success = False
            else:
                # Single test
                status = "✅ PASS" if result else "❌ FAIL"
                details = (
                    "Completed successfully" if result else "Check logs for errors"
                )

                if not result:
                    overall_success = False

            table.add_row(category.replace("_", " ").title(), status, details)

        layout["body"].update(table)

        # Footer
        if overall_success:
            footer_text = Text(
                "🎉 All tests passed! MoMLNIDS is ready for demonstration.",
                style="bold green",
            )
        else:
            footer_text = Text(
                "⚠️ Some tests failed. Please check the logs above.", style="bold yellow"
            )

        layout["footer"].update(
            Panel(
                footer_text,
                style="bright_green" if overall_success else "bright_yellow",
            )
        )

        self.console.print(layout)
        return overall_success


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """🚀 Enhanced MoMLNIDS Demo Script with Rich UI"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    # Display welcome banner
    console.print(
        Panel.fit(
            "[bold blue]MoMLNIDS Enhanced Demo Script[/bold blue]\n"
            "[dim]Multi-objective Multi-Layer Network Intrusion Detection System[/dim]\n"
            "[green]✨ Powered by Click, Rich, and OmegaConf ✨[/green]",
            border_style="bright_blue",
        )
    )


@cli.command()
@click.option("--functions", "-f", help="Comma-separated list of test functions to run")
@click.option(
    "--max-batches", "-b", default=5, help="Maximum batches for training tests"
)
@click.option("--config", "-c", help="Custom configuration file")
@click.pass_context
def test(ctx, functions, max_batches, config):
    """🧪 Run comprehensive functionality tests"""

    runner = DemoRunner()

    # Available test functions
    available_functions = {
        "environment": runner.test_environment,
        "explainability": runner.test_explainability,
        "quick_training": lambda: runner.run_quick_training(max_batches=max_batches),
    }

    # Determine which functions to run
    if functions:
        selected_functions = [f.strip() for f in functions.split(",")]
        console.print(
            f"[cyan]Running selected functions: {', '.join(selected_functions)}[/cyan]"
        )
    else:
        selected_functions = list(available_functions.keys())
        console.print("[cyan]Running all test functions[/cyan]")

    # Run tests
    results = {}

    for func_name in selected_functions:
        if func_name in available_functions:
            console.print(f" [bold]Running: {func_name}[/bold]")
            results[func_name] = available_functions[func_name]()
        else:
            console.print(f"[red]Unknown test function: {func_name}[/red]")
            results[func_name] = False

    # Display results
    runner.display_results_summary(results)


@cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--data-path", "-d", help="Path to test data (optional)")
@click.option("--batch-size", "-b", default=32, help="Batch size for inference")
def test_model(model_path, data_path, batch_size):
    """🎯 Test a trained model (.pth file)"""

    runner = DemoRunner()

    console.print(Panel(f"Testing Model: {model_path}", style="green"))

    # Test model loading
    load_success = runner.test_model_loading(model_path)

    if load_success:
        # Run inference test
        inference_success = runner.run_inference_test(model_path, data_path)

        if inference_success:
            console.print(
                "✅ [bold green]Model testing completed successfully![/bold green]"
            )
        else:
            console.print("❌ [bold red]Inference test failed![/bold red]")
    else:
        console.print("❌ [bold red]Model loading failed![/bold red]")


@cli.command()
@click.option("--test-mode", is_flag=True, help="Run in quick test mode")
@click.option("--max-batches", "-b", default=10, help="Maximum batches per epoch")
@click.option("--epochs", "-e", default=1, help="Number of epochs")
@click.option("--config", "-c", help="Custom configuration file")
def train(test_mode, max_batches, epochs, config):
    """🚀 Run training experiments"""

    runner = DemoRunner()

    if test_mode:
        max_batches = min(max_batches, 5)
        epochs = 1
        console.print(
            "[yellow]Running in test mode (limited batches and epochs)[/yellow]"
        )

    console.print(
        Panel(
            f"Training Configuration: {epochs} epochs, max {max_batches} batches",
            style="blue",
        )
    )

    if config:
        config_path = config
    else:
        # Create quick config
        config_overrides = {
            "training": {
                "epochs": epochs,
                "max_batches": max_batches,
                "batch_size": 16 if test_mode else 32,
            }
        }
        config_path = runner.create_quick_config(config_overrides)

    success = runner.run_quick_training(max_batches=max_batches, epochs=epochs)

    if success:
        console.print("✅ [bold green]Training completed successfully![/bold green]")
    else:
        console.print("❌ [bold red]Training failed![/bold red]")


@cli.command()
@click.option(
    "--scenario",
    "-s",
    type=click.Choice(["baseline", "pseudo_labeling", "all"]),
    default="baseline",
    help="Experiment scenario to run",
)
@click.option("--test-mode", is_flag=True, help="Run in quick test mode")
@click.option("--max-batches", "-b", default=10, help="Maximum batches per epoch")
def experiment(scenario, test_mode, max_batches):
    """🔬 Run full experiments (baseline or pseudo-labeling)"""

    runner = DemoRunner()

    if test_mode:
        max_batches = min(max_batches, 5)
        console.print("[yellow]Running experiments in test mode[/yellow]")

    scenarios_to_run = []
    if scenario == "all":
        scenarios_to_run = ["baseline", "pseudo_labeling"]
    else:
        scenarios_to_run = [scenario]

    results = {}

    for exp_scenario in scenarios_to_run:
        console.print(Panel(f"Running {exp_scenario} experiment", style="magenta"))

        if exp_scenario == "baseline":
            config_file = "config/baseline_config.yaml"
        else:
            config_file = "config/experiment_config.yaml"

        # Create modified config for test mode
        if test_mode:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            config["training"]["epochs"] = 1
            config["training"]["max_batches"] = max_batches
            config["wandb"]["enabled"] = False

            temp_config = f"config/temp_{exp_scenario}_config.yaml"
            with open(temp_config, "w") as f:
                yaml.dump(config, f)

            config_file = temp_config

        success, output = runner.run_command_with_progress(
            f"python src/main_config.py --config {config_file}",
            f"Running {exp_scenario} experiment",
            timeout=1800 if test_mode else 7200,
        )

        results[exp_scenario] = success

        # Cleanup temp config
        if test_mode and Path(f"config/temp_{exp_scenario}_config.yaml").exists():
            Path(f"config/temp_{exp_scenario}_config.yaml").unlink()

    # Display experiment results
    runner.display_results_summary(results)


@cli.command()
def config():
    """⚙️ Show current configuration and create custom configs"""

    console.print(Panel("Configuration Management", style="cyan"))

    # Display default config
    console.print("[bold]Default Configuration:[/bold]")
    config_text = OmegaConf.to_yaml(OmegaConf.create(DEFAULT_CONFIG))
    console.print(f"[dim]{config_text}[/dim]")

    # Option to create custom config
    if Confirm.ask("Would you like to create a custom configuration?"):
        config_name = Prompt.ask("Enter configuration name", default="custom_demo")

        # Get user preferences
        batch_size = int(Prompt.ask("Batch size", default="16"))
        epochs = int(Prompt.ask("Number of epochs", default="1"))
        max_batches = int(Prompt.ask("Max batches per epoch", default="5"))

        custom_config = OmegaConf.create(DEFAULT_CONFIG)
        custom_config.training.batch_size = batch_size
        custom_config.training.epochs = epochs
        custom_config.training.max_batches = max_batches
        custom_config.project.name = f"MoMLNIDS_{config_name}"

        config_path = f"config/{config_name}_config.yaml"
        Path("config").mkdir(exist_ok=True)

        with open(config_path, "w") as f:
            OmegaConf.save(custom_config, f)

        console.print(f"✅ [green]Custom configuration saved to: {config_path}[/green]")


if __name__ == "__main__":
    cli()
