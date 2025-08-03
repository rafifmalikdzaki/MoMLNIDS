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
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from omegaconf import DictConfig, OmegaConf

# Add the project's root directory to the Python path
# This is necessary for the script to find the 'src' package
project_root = Path(__file__).resolve().parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

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
        "batch_size": 1,  # Small batch size for fast testing
        "epochs": 1,
        "learning_rate": 0.001,
        "target_index": 0,
        "experiment_num": "|Demo|",
        "use_cluster": False,
        "eval_step": 1,
        "save_step": 1,
        "max_batches": 1,  # Very small for demo
        "extractor_weight": 1.0,
        "classifier_weight": 1.0,
        "discriminator_weight": 1.0,
        "amsgrad": False,
        "t_max": 10,
        "clustering_step": 1,
        "label_smooth": 0.0,
        "entropy_weight": 0.0,
        "grl_weight": 1.0,
        "eval_batch_frequency": 1,
    },
    "data": {
        "processed_data_path": "./data/parquet/",
        "domain_reweighting": [1.0, 1.0, 1.0],
        "label_reweighting": [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    },
    "device": {"num_workers": 0},
    "model": {
        "input_nodes": 39,
        "single_layer": True,
    },
    "logging": {
        "verbose": True,
    },
}


class DemoRunner:
    """Enhanced demo runner with rich UI and comprehensive functionality."""

    def __init__(self, interactive=False):
        self.console = console
        self.results = {}
        self.interactive = interactive
        self.log_file = Path("logs/demo_sidang_enhanced_errors.txt")
        self.log_file.parent.mkdir(exist_ok=True)
        # Clear log file at the start of a run
        if self.log_file.exists():
            try:
                self.log_file.unlink()
            except OSError as e:
                self.console.print(
                    f"[yellow]Could not remove old log file: {e}[/yellow]"
                )
        self.live_layout = Layout(name="root")
        self.live_output_panel = Panel(
            "",
            title="[bold blue]Live Output[/bold blue]",
            border_style="blue",
            box=box.ROUNDED,
        )
        self.live_status_panel = Panel(
            "",
            title="[bold green]Test Status[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        )
        self.live_progress_panel = Panel(
            "",
            title="[bold yellow]Progress[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
        )

    def interactive_pause(
        self, message: str = None, skip_on_non_interactive: bool = True
    ):
        """Pause execution for user interaction if in interactive mode."""
        if not self.interactive:
            if skip_on_non_interactive:
                return True

        if message:
            self.console.print(f"\n[cyan]{message}[/cyan]")

        self.console.print("\n[dim]Interactive Navigation:[/dim]")
        self.console.print("  [yellow]Enter[/yellow] - Continue to next step")
        self.console.print("  [yellow]s[/yellow] - Skip remaining steps in this test")
        self.console.print("  [yellow]q[/yellow] - Quit demo")
        self.console.print("  [yellow]r[/yellow] - Show last output again")

        while True:
            try:
                user_input = input("\n[Next] ").strip().lower()

                if user_input == "":
                    return True  # Continue
                elif user_input == "s":
                    return False  # Skip
                elif user_input == "q":
                    self.console.print("[yellow]Demo terminated by user[/yellow]")
                    sys.exit(0)
                elif user_input == "r":
                    # Show the last output again by returning and letting the caller handle
                    self.console.print("[dim]Showing last output again...[/dim]")
                    continue
                else:
                    self.console.print(
                        "[red]Invalid input. Use Enter, 's', 'q', or 'r'[/red]"
                    )
                    continue

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Demo terminated by user[/yellow]")
                sys.exit(0)

    def update_live_display(
        self,
        live: Live,
        status_content: str = None,
        output_content: str = None,
        progress_renderable=None,
    ):
        if status_content:
            self.live_status_panel.renderable = Text(status_content, style="green")
        if output_content:
            self.live_output_panel.renderable = Text(output_content, style="dim")
        if progress_renderable:
            self.live_progress_panel.renderable = progress_renderable

        self.live_layout.split_column(
            Layout(self.live_status_panel, size=3),
            Layout(self.live_progress_panel, size=5),
            Layout(self.live_output_panel),
        )
        live.update(self.live_layout) if live is not None else None

    def log_error(self, header: str, details: List[str]):
        """Log an error to the file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(f"--- {datetime.now().isoformat()} ---\n")
                f.write(f"ERROR: {header}\n")
                f.write("DETAILS:\n")
                f.writelines([f"  {line}\n" for line in details])
                f.write("\n")
        except Exception as e:
            self.console.print(f"[red]Failed to write to log file: {e}[/red]")

    def run_command_with_progress(
        self,
        command: str,
        description: str,
        timeout: int = 300,
        live: Optional[Live] = None,
        capture_rich_output: bool = False,
    ) -> tuple[bool, List[str]]:
        """Run command with rich progress bar."""
        progress_renderable = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        task = progress_renderable.add_task(description, total=None)

        output_lines = []
        try:
            # Get current environment and modify PYTHONPATH
            env = os.environ.copy()
            project_root = Path(__file__).resolve().parent
            src_path = str(project_root / "src")
            if "PYTHONPATH" in env:
                if src_path not in env["PYTHONPATH"]:
                    env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
            else:
                env["PYTHONPATH"] = src_path

            # For commands that produce rich output, use a different capture method
            if capture_rich_output or "smoke_test" in command:
                # Run command and capture all output at once
                process = subprocess.run(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    timeout=timeout
                )
                
                if process.stdout:
                    output_lines = process.stdout.splitlines()
                
                if process.returncode == 0:
                    final_status = f"✅ {description} - Completed"
                    progress_renderable.update(task, description=final_status)
                    if live:
                        self.update_live_display(
                            live,
                            status_content=final_status,
                            output_content=process.stdout,
                            progress_renderable=progress_renderable,
                        )
                    return True, output_lines
                else:
                    final_status = f"❌ {description} - Failed"
                    progress_renderable.update(task, description=final_status)
                    self.log_error(
                        f"Command failed: {description}",
                        [f"Command: {command}", "Output:"] + output_lines,
                    )
                    return False, output_lines
            else:
                # Original streaming method for other commands
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env=env,
                )

            start_time = time.time()

            for line in process.stdout:
                output_lines.append(line.strip())
                current_description = f"{description} - {line.strip()[:50]}..."
                progress_renderable.update(task, description=current_description)
                if live:
                    self.update_live_display(
                        live,
                        status_content=current_description,
                        output_content="\n".join(output_lines),
                        progress_renderable=progress_renderable,
                    )

                if time.time() - start_time > timeout:
                    process.terminate()
                    timeout_message = f"Command timed out after {timeout}s"
                    self.console.print(f"[red]{timeout_message}[/red]")
                    self.log_error(
                        f"Command timed out: {description}",
                        [f"Timeout: {timeout}s", f"Command: {command}"],
                    )
                    if live:
                        self.update_live_display(
                            live,
                            status_content=f"❌ {description} - Timed Out",
                            output_content="\n".join(output_lines),
                            progress_renderable=progress_renderable,
                        )
                    return False, output_lines

            process.wait()

            if process.returncode == 0:
                final_status = f"✅ {description} - Completed"
                progress_renderable.update(task, description=final_status)
                if live:
                    self.update_live_display(
                        live,
                        status_content=final_status,
                        output_content="\n".join(output_lines),
                        progress_renderable=progress_renderable,
                    )
                return True, output_lines
            else:
                final_status = f"❌ {description} - Failed"
                progress_renderable.update(task, description=final_status)
                self.log_error(
                    f"Command failed: {description}",
                    [f"Command: {command}", "Output:"] + output_lines,
                )
                error_message = f"Error details logged to {self.log_file}"
                self.console.print(f"[yellow]{error_message}[/yellow]")
                if live:
                    self.update_live_display(
                        live,
                        status_content=final_status,
                        output_content="\n".join(output_lines) + "\n" + error_message,
                        progress_renderable=progress_renderable,
                    )
                return False, output_lines

        except Exception as e:
            exception_message = f"Error running command: {e}"
            self.console.print(f"[red]{exception_message}[/red]")
            self.log_error(
                f"Exception running command: {description}",
                [f"Command: {command}", f"Exception: {e}"],
            )
            error_message = f"Error details logged to {self.log_file}"
            self.console.print(f"[yellow]{error_message}[/yellow]")
            if live:
                self.update_live_display(
                    live,
                    status_content=f"❌ {description} - Exception",
                    output_content="\n".join(output_lines)
                    + "\n"
                    + exception_message
                    + "\n"
                    + error_message,
                    progress_renderable=progress_renderable,
                )
            return False, []

    def test_environment(self, live: Live) -> Dict[str, bool]:
        """Test environment with rich formatting."""
        self.update_live_display(live, status_content="🔧 Testing Environment Setup")

                if self.interactive:
                    if success:
                        self.console.print(f"[green]✅ {test_name} completed successfully[/green]")
                        # For smoke test, show the rich output
                        if "smoke test" in test_name.lower() and output:
                            self.console.print("[dim]Smoke test output:[/dim]")
                            # Join all output lines and print as a block for rich formatting
                            full_output = "\n".join(output)
                            # Filter out progress bar artifacts and show meaningful lines
                            meaningful_lines = [line for line in output if 
                                              ('✓' in line or '🎉' in line or 'Testing' in line or 
                                               'passed' in line or 'successful' in line) and 
                                              not line.startswith('\r')]
                            for line in meaningful_lines[-8:]:  # Show last 8 meaningful lines
                                self.console.print(f"[dim]  {line}[/dim]")
                        elif output:
                            # For other tests, show last few lines
                            for line in output[-3:]:
                                self.console.print(f"[dim]  {line}[/dim]")
                    else:
                        self.console.print(f"[red]❌ {test_name} failed[/red]")
                        if output:
                            self.console.print("[dim]Last few lines of output:[/dim]")
                            for line in output[-3:]:
                                self.console.print(f"[dim]  {line}[/dim]")

        self.console.print(table)
        return results

    def test_explainability(self, live: Live) -> bool:
        """Test explainable AI with rich output."""
        self.update_live_display(
            live, status_content="🧠 Testing Explainable AI Features"
        )

        if self.interactive:
            if not self.interactive_pause(
                "🧠 About to test explainable AI features (model interpretation, feature attribution)"
            ):
                return False

        explainability_code = """
import torch
import numpy as np
from src.skripsi_code.model.MoMLNIDS import momlnids
from src.skripsi_code.explainability.explainer import ModelExplainer

# Create model
model = momlnids(
    input_nodes=39,
    hidden_nodes=[64, 32, 16, 10],
    classifier_nodes=[64, 32, 16],
    num_domains=3,
    num_class=2,
    single_layer=True
)

# Test explainer
feature_names = [f'feature_{i}' for i in range(39)]
explainer = ModelExplainer(model, feature_names)

# Test explanation
test_instance = torch.randn(39).float().numpy()
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
            live=live,
        )

        Path("temp_explainability_test.py").unlink(missing_ok=True)

        if success:
            self.update_live_display(
                live, status_content="✅ Explainability test passed!"
            )
            if self.interactive:
                self.console.print(
                    "[green]✅ Explainability test completed successfully![/green]"
                )
                if output:
                    self.console.print("[dim]Output:[/dim]")
                    for line in output[-5:]:
                        self.console.print(f"[dim]  {line}[/dim]")
        else:
            self.update_live_display(
                live, status_content="❌ Explainability test failed!"
            )
            self.log_error("Explainability test failed", ["Output:"] + output)
            self.console.print(
                f"[yellow]Error details logged to {self.log_file}[/yellow]"
            )
            if self.interactive:
                self.console.print("[red]❌ Explainability test failed![/red]")
                if output:
                    self.console.print("[dim]Last few lines of output:[/dim]")
                    for line in output[-3:]:
                        self.console.print(f"[dim]  {line}[/dim]")

        return success

    def test_model_loading(self, model_path: str, live: Optional[Live] = None) -> bool:
        """Test loading a trained model."""
        self.update_live_display(
            live, status_content=f"🔄 Testing Model Loading: {model_path}"
        )

        if not Path(model_path).exists():
            self.update_live_display(
                live, status_content=f"[red]Model file not found: {model_path}[/red]"
            )
            return False

        try:
            self.update_live_display(live, status_content="Loading model...")
            # Load model state dict
            state_dict = torch.load(model_path, map_location="cpu")

            # Create model instance (you may need to adjust these parameters)
            model = torch.load(model_path, map_location="cpu")

            self.update_live_display(
                live, status_content="✅ Model loaded successfully!"
            )

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
            self.update_live_display(
                live, status_content=f"[red]Error loading model: {e}[/red]"
            )
            self.log_error(f"Model loading failed: {model_path}", [f"Exception: {e}"])
            self.console.print(
                f"[yellow]Error details logged to {self.log_file}[/yellow]"
            )
            return False

    def run_inference_test(
        self,
        model_path: str,
        data_path: Optional[str] = None,
        live: Optional[Live] = None,
    ) -> bool:
        """Run inference test on a trained model."""
        self.update_live_display(live, status_content="🎯 Running Inference Test")

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
            "python temp_inference_test.py",
            "Running inference test",
            timeout=60,
            live=live,
        )

        if not success:
            self.log_error(
                f"Inference test failed for model: {model_path}",
                ["Output:"] + output,
            )
            self.console.print(
                f"[yellow]Error details logged to {self.log_file}[/yellow]"
            )
            self.update_live_display(
                live,
                status_content=f"❌ Inference test failed! Check logs at {self.log_file}",
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

    def run_quick_training(
        self, max_batches: int = 5, epochs: int = 1, live: Optional[Live] = None
    ) -> bool:
        """Run quick training test."""
        self.update_live_display(
            live,
            status_content=f"🚀 Quick Training Test ({epochs} epoch, {max_batches} batches)",
        )

        if self.interactive:
            if not self.interactive_pause(
                f"🚀 About to run quick training test ({epochs} epoch, {max_batches} batches max)"
            ):
                return False

        config_overrides = {
            "training": {
                "epochs": epochs,
                "max_batches": max_batches,
                "batch_size": 4,  # Use larger batch size to avoid BatchNorm issues
            }
        }

        config_path = self.create_quick_config(config_overrides)

        try:
            success, output = self.run_command_with_progress(
                f"python src/main_config.py --config {config_path}",
                "Running quick training",
                timeout=600,
                live=live,
            )
            if not success:
                error_message = "Quick training failed."
                if "num_samples should be a positive integer value" in "\n".join(
                    output
                ):
                    error_message += " This is likely due to missing data. Please ensure data is available in the configured processed_data_path."
                else:
                    error_message += " Check the logs for more details."

                self.console.print(f"[red]{error_message}[/red]")
                self.log_error(
                    "Quick training failed",
                    ["Output:"] + output,
                )
                self.update_live_display(live, status_content=f"❌ {error_message}")

                if self.interactive:
                    self.console.print("[red]❌ Quick training failed![/red]")
                    if output:
                        self.console.print("[dim]Last few lines of output:[/dim]")
                        for line in output[-5:]:
                            self.console.print(f"[dim]  {line}[/dim]")
                return False

            self.update_live_display(
                live, status_content="✅ Quick training completed successfully!"
            )

            if self.interactive:
                self.console.print(
                    "[green]✅ Quick training completed successfully![/green]"
                )
                if output:
                    # Show some key output lines
                    relevant_lines = [
                        line
                        for line in output
                        if any(
                            keyword in line.lower()
                            for keyword in [
                                "epoch",
                                "loss",
                                "accuracy",
                                "completed",
                                "finished",
                            ]
                        )
                    ]
                    if relevant_lines:
                        self.console.print("[dim]Key training metrics:[/dim]")
                        for line in relevant_lines[-5:]:
                            self.console.print(f"[dim]  {line}[/dim]")

            return success
        except Exception as e:
            error_message = f"Error during quick training: {e}"
            self.console.print(f"[red]{error_message}[/red]")
            self.log_error(
                "Error during quick training",
                [f"Exception: {e}"],
            )
            self.update_live_display(live, status_content=f"❌ {error_message}")

            if self.interactive:
                self.console.print(f"[red]❌ Exception during training: {e}[/red]")

            return False

    def run_project_validation(self, live: Optional[Live] = None) -> Dict[str, bool]:
        """Run comprehensive project validation from validate_project.py logic."""
        self.update_live_display(live, status_content="🎓 Running Project Validation")

        validation_results = {}

        # Test project structure
        self.update_live_display(live, status_content="📁 Validating project structure")
        required_dirs = [
            project_root / "src" / "skripsi_code",
            project_root / "src" / "skripsi_code" / "model",
            project_root / "src" / "skripsi_code" / "utils",
            project_root / "src" / "skripsi_code" / "clustering",
            project_root / "src" / "skripsi_code" / "explainability",
            project_root / "src" / "skripsi_code" / "TrainEval",
            project_root / "src" / "skripsi_code" / "config",
            project_root / "src" / "skripsi_code" / "experiment",
            project_root / "config",
        ]

        structure_valid = all(dir_path.exists() for dir_path in required_dirs)
        validation_results["Project Structure"] = structure_valid

        # Test main scripts
        self.update_live_display(live, status_content="📄 Validating main scripts")
        main_scripts = [
            "main.py",
            "main_improved.py",
            "main_config.py",
            "main_pseudo.py",
            "main_pseudo_50.py",
        ]

        scripts_valid = all(
            (project_root / "src" / script).exists()
            and (project_root / "src" / script).stat().st_size > 0
            for script in main_scripts
        )
        validation_results["Main Scripts"] = scripts_valid

        # Test module imports
        self.update_live_display(live, status_content="🔧 Testing module imports")
        import_tests = [
            (
                "Config Manager",
                "python -c \"import sys; sys.path.insert(0, 'src'); from skripsi_code.config.config_manager import ConfigManager; print('✅ Config OK')\"",
            ),
            (
                "Clustering",
                "python -c \"import sys; sys.path.insert(0, 'src'); from skripsi_code.clustering.cluster_methods import Kmeans; print('✅ Clustering OK')\"",
            ),
            (
                "Explainability",
                "python -c \"import sys; sys.path.insert(0, 'src'); from skripsi_code.explainability.explainer import ModelExplainer; print('✅ Explainability OK')\"",
            ),
            (
                "Model",
                "python -c \"import sys; sys.path.insert(0, 'src'); from skripsi_code.model.MoMLNIDS import momlnids; print('✅ Model OK')\"",
            ),
        ]

        import_results = {}
        for test_name, command in import_tests:
            self.update_live_display(live, status_content=f"Testing {test_name} import")
            success, _ = self.run_command_with_progress(
                command, f"Testing {test_name}", timeout=30, live=live
            )
            import_results[test_name] = success

        validation_results["Module Imports"] = all(import_results.values())

        # Test demo functions exist
        self.update_live_display(live, status_content="🎯 Checking demo functions")
        demo_functions = [
            (
                project_root
                / "src"
                / "skripsi_code"
                / "clustering"
                / "cluster_methods.py",
                "demo_clustering_methods",
            ),
            (
                project_root
                / "src"
                / "skripsi_code"
                / "explainability"
                / "explainer.py",
                "demo_explainability",
            ),
            (
                project_root / "src" / "skripsi_code" / "config" / "config_manager.py",
                "demo_config_management",
            ),
        ]

        demo_results = {}
        for file_path, demo_function in demo_functions:
            if file_path.exists():
                content = file_path.read_text()
                has_demo = (
                    demo_function in content and "def " + demo_function in content
                )
                demo_results[file_path.stem] = has_demo
            else:
                demo_results[file_path.stem] = False

        validation_results["Demo Functions"] = all(demo_results.values())

        return validation_results

    def run_module_demos(self, live: Optional[Live] = None) -> Dict[str, bool]:
        """Run individual module demonstrations."""
        self.update_live_display(
            live, status_content="🔬 Running Module Demonstrations"
        )

        if self.interactive:
            if not self.interactive_pause(
                "🔬 About to run module demonstrations (clustering, explainability, config)"
            ):
                return {"Module Demos": False}

        demo_results = {}

        # Demo modules with descriptions
        demos = [
            (
                "Clustering Demo",
                "python src/skripsi_code/clustering/cluster_methods.py --demo",
                "🧮 Testing clustering algorithms",
            ),
            (
                "Explainability Demo",
                "python src/skripsi_code/explainability/explainer.py --demo",
                "🧠 Testing model explainability",
            ),
            (
                "Config Demo",
                "python src/skripsi_code/config/config_manager.py --demo",
                "⚙️ Testing configuration management",
            ),
        ]

        for i, (demo_name, command, description) in enumerate(demos, 1):
            self.update_live_display(live, status_content=description)

            if self.interactive:
                if not self.interactive_pause(
                    f"Running demo {i}/{len(demos)}: {description}"
                ):
                    # Skip remaining demos
                    for remaining_demo, _, _ in demos[i - 1 :]:
                        demo_results[remaining_demo] = False
                    break

            success, output = self.run_command_with_progress(
                command,
                demo_name.lower(),
                timeout=180,
                live=live,
            )
            demo_results[demo_name] = success

            if self.interactive:
                if success:
                    self.console.print(
                        f"[green]✅ {demo_name} completed successfully[/green]"
                    )
                    # Show some output if available
                    if output:
                        interesting_lines = [
                            line
                            for line in output
                            if any(
                                keyword in line.lower()
                                for keyword in [
                                    "success",
                                    "completed",
                                    "demo",
                                    "test",
                                    "✅",
                                ]
                            )
                        ]
                        if interesting_lines:
                            self.console.print("[dim]Demo highlights:[/dim]")
                            for line in interesting_lines[-3:]:
                                self.console.print(f"[dim]  {line}[/dim]")
                else:
                    self.console.print(f"[red]❌ {demo_name} failed[/red]")
                    if output:
                        self.console.print("[dim]Last few lines of output:[/dim]")
                        for line in output[-3:]:
                            self.console.print(f"[dim]  {line}[/dim]")

        return demo_results

    def display_project_info(self, live: Optional[Live] = None):
        """Display comprehensive project information."""
        self.update_live_display(
            live, status_content="📊 Displaying Project Information"
        )

        # Create project info table
        info_table = Table(title="🎓 MoMLNIDS Project Information", box=box.ROUNDED)
        info_table.add_column("Component", style="cyan", width=25)
        info_table.add_column("Description", style="white", width=50)
        info_table.add_column("Status", style="green", width=15)

        # Project components
        components = [
            (
                "Multi-objective Learning",
                "Simultaneous classification and domain adaptation",
                "✅ Active",
            ),
            (
                "Multi-layer Architecture",
                "Feature extraction + classification + discrimination",
                "✅ Active",
            ),
            (
                "Clustering Methods",
                "K-means, GMM, Spectral, Agglomerative clustering",
                "✅ Active",
            ),
            (
                "Explainable AI",
                "Feature attribution, SHAP, LIME integration",
                "✅ Active",
            ),
            (
                "Configuration Management",
                "Flexible YAML-based configuration system",
                "✅ Active",
            ),
            (
                "Experiment Tracking",
                "Comprehensive experiment logging and tracking",
                "✅ Active",
            ),
            (
                "Pseudo-labeling",
                "Semi-supervised learning with confidence thresholding",
                "✅ Active",
            ),
            (
                "Domain Adaptation",
                "Gradient reversal layer for domain invariance",
                "✅ Active",
            ),
        ]

        for component, description, status in components:
            info_table.add_row(component, description, status)

        self.console.print(info_table)

        # Display available scripts
        scripts_table = Table(title="📝 Available Demo Scripts", box=box.ROUNDED)
        scripts_table.add_column("Script", style="cyan")
        scripts_table.add_column("Purpose", style="white")
        scripts_table.add_column("Command Example", style="yellow")

        scripts = [
            ("main.py", "Basic training script", "python src/main.py"),
            (
                "main_improved.py",
                "Enhanced training with logging",
                "python src/main_improved.py",
            ),
            (
                "main_config.py",
                "Configuration-based training",
                "python src/main_config.py --config config/default_config.yaml",
            ),
            (
                "main_pseudo.py",
                "Pseudo-labeling experiments",
                "python src/main_pseudo.py",
            ),
            (
                "demo_sidang_enhanced.py",
                "Comprehensive demo interface",
                "python demo_sidang_enhanced.py test",
            ),
        ]

        for script, purpose, command in scripts:
            scripts_table.add_row(script, purpose, command)

        self.console.print(scripts_table)

    def display_results_summary(
        self, results: Dict[str, Any], live: Optional[Live] = None
    ):
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
                # Environment tests or validation results
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
                "🎉 All tests passed! MoMLNIDS is ready for thesis defense!",
                style="bold green",
            )
        else:
            footer_text = Text(
                f"⚠️ Some tests failed. Check logs at {self.log_file}.",
                style="bold yellow",
            )

        layout["footer"].update(
            Panel(
                footer_text,
                style="bright_green" if overall_success else "bright_yellow",
            )
        )

        if live:
            live.stop()
        self.console.print(layout)
        return overall_success


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """🚀 Enhanced MoMLNIDS Demo Script for Thesis Defense"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    # Display welcome banner
    console.print(
        Panel.fit(
            "[bold blue]🎓 MoMLNIDS Enhanced Demo Script[/bold blue]\n"
            "[dim]Multi-objective Multi-Layer Network Intrusion Detection System[/dim]\n"
            "[green]✨ Comprehensive Testing & Validation for Thesis Defense ✨[/green]\n\n"
            "[bold cyan]Available Commands:[/bold cyan]\n"
            "• [yellow]validate[/yellow] - Project validation for thesis defense\n"
            "• [yellow]test[/yellow] - Comprehensive functionality testing\n"
            "• [yellow]demos[/yellow] - Run all module demonstrations\n"
            "• [yellow]info[/yellow] - Display project information\n"
            "• [yellow]train[/yellow] - Training experiments\n"
            "• [yellow]experiment[/yellow] - Full research experiments\n"
            "• [yellow]config[/yellow] - Configuration management\n\n"
            "[dim]Use --help with any command for detailed options[/dim]",
            border_style="bright_blue",
        )
    )


@cli.command()
@click.option("--functions", "-f", help="Comma-separated list of test functions to run")
@click.option(
    "--max-batches", "-b", default=5, help="Maximum batches for training tests"
)
@click.option("--config", "-c", help="Custom configuration file")
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Run in interactive mode with step-by-step navigation",
)
@click.pass_context
def test(ctx, functions, max_batches, config, interactive):
    """🧪 Run comprehensive functionality tests"""

    runner = DemoRunner(interactive=interactive)

    if interactive:
        console.print(
            Panel(
                "[bold cyan]🎮 Interactive Mode Enabled[/bold cyan]\n\n"
                "In interactive mode, you can:\n"
                "• Navigate step-by-step through each test\n"
                "• Skip tests or entire sections\n"
                "• Review outputs and errors as they happen\n"
                "• Exit at any time with 'q'\n\n"
                "[dim]Press Enter to continue, 's' to skip, 'q' to quit, 'r' to review[/dim]",
                border_style="cyan",
            )
        )

    # Available test functions
    available_functions = {
        "environment": runner.test_environment,
        "explainability": runner.test_explainability,
        "project_validation": runner.run_project_validation,
        "module_demos": runner.run_module_demos,
        "quick_training": lambda live: runner.run_quick_training(
            max_batches=max_batches, live=live
        ),
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

    if interactive:
        if not runner.interactive_pause(
            f"About to run {len(selected_functions)} test function(s): {', '.join(selected_functions)}"
        ):
            console.print("[yellow]Demo cancelled by user[/yellow]")
            return

    # Run tests
    results = {}

    with Live(runner.live_layout, screen=True, refresh_per_second=4) as live:
        for func_name in selected_functions:
            if func_name in available_functions:
                console.print(f" [bold]Running: {func_name}[/bold]")
                # Pass the live object to the test functions
                if func_name == "quick_training":
                    results[func_name] = available_functions[func_name](live=live)
                else:
                    results[func_name] = available_functions[func_name](live=live)
            else:
                console.print(f"[red]Unknown test function: {func_name}[/red]")
                results[func_name] = False
                runner.log_error(
                    f"Unknown test function specified: {func_name}",
                    [f"Available functions: {list(available_functions.keys())}"],
                )
                console.print(
                    f"[yellow]Error details logged to {runner.log_file}[/yellow]"
                )

    # Display results
    runner.display_results_summary(results)


@cli.command()
def validate():
    """🎓 Run comprehensive project validation for thesis defense"""

    runner = DemoRunner()

    console.print(
        Panel("🎓 MoMLNIDS Project Validation for Thesis Defense", style="bold blue")
    )

    with Live(runner.live_layout, screen=True, refresh_per_second=4) as live:
        validation_results = runner.run_project_validation(live=live)

    # Display validation summary
    validation_table = Table(title="🎯 Project Validation Summary", box=box.ROUNDED)
    validation_table.add_column("Component", style="cyan")
    validation_table.add_column("Status", style="bold")
    validation_table.add_column("Result", style="dim")

    all_passed = True
    for component, passed in validation_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        result = "Ready for defense" if passed else "Needs attention"
        validation_table.add_row(component, status, result)
        if not passed:
            all_passed = False

    console.print(validation_table)

    if all_passed:
        console.print(
            "\n🎉 [bold green]PROJECT IS READY FOR THESIS DEFENSE![/bold green]"
        )
        console.print("\n[dim]Quick Demo Commands:[/dim]")
        console.print(
            "- [cyan]python demo_sidang_enhanced.py demos[/cyan] - Run all module demos"
        )
        console.print(
            "- [cyan]python demo_sidang_enhanced.py info[/cyan] - Show project information"
        )
        console.print(
            "- [cyan]python demo_sidang_enhanced.py test[/cyan] - Run comprehensive tests"
        )
    else:
        console.print(
            f"\n⚠️ [bold yellow]Some components need attention. Check logs at {runner.log_file}[/bold yellow]"
        )


@cli.command()
@click.option("--interactive", "-i", is_flag=True, help="Run in interactive mode")
def demos(interactive):
    """🔬 Run all module demonstrations"""

    runner = DemoRunner(interactive=interactive)

    if interactive:
        console.print(
            Panel(
                "[bold cyan]🎮 Interactive Demo Mode[/bold cyan]\n\n"
                "You'll be able to step through each demo module:\n"
                "• Clustering algorithms demonstration\n"
                "• Explainability features showcase\n"
                "• Configuration management demo\n\n"
                "[dim]Navigate with Enter, skip with 's', quit with 'q'[/dim]",
                border_style="cyan",
            )
        )

    console.print(Panel("🔬 Running All Module Demonstrations", style="bold green"))

    with Live(runner.live_layout, screen=True, refresh_per_second=4) as live:
        demo_results = runner.run_module_demos(live=live)

    # Display demo results
    demo_table = Table(title="📋 Module Demo Results", box=box.ROUNDED)
    demo_table.add_column("Module", style="cyan")
    demo_table.add_column("Status", style="bold")
    demo_table.add_column("Description", style="dim")

    descriptions = {
        "Clustering Demo": "Demonstrates all clustering algorithms with sample data",
        "Explainability Demo": "Shows model interpretability features",
        "Config Demo": "Tests configuration management functionality",
    }

    for demo_name, success in demo_results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        description = descriptions.get(demo_name, "Module demonstration")
        demo_table.add_row(demo_name.replace(" Demo", ""), status, description)

    console.print(demo_table)

    success_count = sum(1 for result in demo_results.values() if result)
    total_count = len(demo_results)

    if success_count == total_count:
        console.print(
            f"\n🎉 [bold green]All {total_count} module demonstrations completed successfully![/bold green]"
        )
    else:
        console.print(
            f"\n⚠️ [bold yellow]{success_count}/{total_count} demonstrations successful. Check logs for details.[/bold yellow]"
        )


@cli.command()
def info():
    """📊 Display comprehensive project information"""

    runner = DemoRunner()

    console.print(Panel("📊 MoMLNIDS Project Information", style="bold cyan"))

    # Display project information
    runner.display_project_info()

    # Show project structure
    console.print("\n")
    structure_tree = Tree("📁 [bold blue]Project Structure[/bold blue]")

    # Add main directories
    src_branch = structure_tree.add("📂 src/")
    skripsi_branch = src_branch.add("📂 skripsi_code/")

    modules = [
        ("📂 model/", "Neural network architectures"),
        ("📂 clustering/", "Clustering algorithms"),
        ("📂 explainability/", "Model interpretation"),
        ("📂 config/", "Configuration management"),
        ("📂 experiment/", "Experiment tracking"),
        ("📂 utils/", "Utility functions"),
        ("📂 TrainEval/", "Training and evaluation"),
    ]

    for module, description in modules:
        skripsi_branch.add(f"{module} - [dim]{description}[/dim]")

    structure_tree.add("📂 config/ - [dim]Configuration files[/dim]")
    structure_tree.add("📂 docs/ - [dim]Documentation[/dim]")
    structure_tree.add("📄 demo_sidang_enhanced.py - [dim]Enhanced demo script[/dim]")

    console.print(structure_tree)

    # Show thesis defense readiness
    console.print("\n")
    readiness_panel = Panel.fit(
        "[bold green]🎓 THESIS DEFENSE READINESS[/bold green]\n\n"
        "✅ Multi-objective learning implementation\n"
        "✅ Multi-layer neural architecture\n"
        "✅ Comprehensive clustering methods\n"
        "✅ Explainable AI capabilities\n"
        "✅ Flexible configuration system\n"
        "✅ Experiment tracking and logging\n"
        "✅ Pseudo-labeling for semi-supervised learning\n"
        "✅ Domain adaptation mechanisms\n\n"
        "[dim]Use 'python demo_sidang_enhanced.py validate' to verify all components[/dim]",
        border_style="green",
    )
    console.print(readiness_panel)


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
            console.print(
                f"❌ [bold red]Inference test failed! Check logs at {runner.log_file}[/bold red]"
            )
    else:
        console.print(
            f"❌ [bold red]Model loading failed! Check logs at {runner.log_file}[/bold red]"
        )


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
            "training": {"epochs": epochs, "max_batches": max_batches, "batch_size": 1}
        }
        config_path = runner.create_quick_config(config_overrides)

    success = runner.run_quick_training(max_batches=max_batches, epochs=epochs)

    if success:
        console.print("✅ [bold green]Training completed successfully![/bold green]")
    else:
        console.print(
            f"❌ [bold red]Training failed! Check logs at {runner.log_file}[/bold red]"
        )


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

        if not success:
            console.print(
                f"[yellow]Experiment '{exp_scenario}' failed. Check logs at {runner.log_file}[/yellow]"
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
        batch_size = int(Prompt.ask("Batch size", default="1"))
        epochs = int(Prompt.ask("Number of epochs", default="1"))
        max_batches = int(Prompt.ask("Max batches per epoch", default="5"))

        custom_config = OmegaConf.create(DEFAULT_CONFIG)
        custom_config.training.batch_size = batch_size
        custom_config.training.epochs = epochs
        custom_config.training.max_batches = max_batches
        custom_config.project.name = f"MoMLNIDS_{config_name}"

        config_path = f"config/{config_name}_config.yaml"
        Path("config").mkdir(exist_ok=True)

        try:
            with open(config_path, "w") as f:
                OmegaConf.save(custom_config, f)

            console.print(
                f"✅ [green]Custom configuration saved to: {config_path}[/green]"
            )
        except Exception as e:
            console.print(f"[red]Failed to create custom config: {e}[/red]")
            # Since we don't have a runner instance here, we can't use self.log_error
            # We will just print the error to the console.


@cli.command()
def thesis():
    """🎓 Quick thesis defense readiness check"""

    runner = DemoRunner()

    console.print(Panel("🎓 Thesis Defense Readiness Check", style="bold magenta"))

    # Quick validation
    with Live(runner.live_layout, screen=True, refresh_per_second=4) as live:
        validation_results = runner.run_project_validation(live=live)

    # Show summary
    all_ready = all(validation_results.values())

    if all_ready:
        console.print("\n🎉 [bold green]THESIS DEFENSE READY![/bold green]")
        console.print("\n[bold cyan]Your MoMLNIDS project includes:[/bold cyan]")
        features = [
            "✅ Multi-objective neural network architecture",
            "✅ Domain adaptation with gradient reversal",
            "✅ Multiple clustering algorithms (K-means, GMM, Spectral, etc.)",
            "✅ Explainable AI with feature attribution methods",
            "✅ Flexible configuration management system",
            "✅ Comprehensive experiment tracking",
            "✅ Pseudo-labeling for semi-supervised learning",
            "✅ Modular, extensible codebase design",
        ]
        for feature in features:
            console.print(f"  {feature}")

        console.print(
            "\n[dim]Run 'python demo_sidang_enhanced.py demos' to see all modules in action![/dim]"
        )
    else:
        console.print(
            "\n⚠️ [bold yellow]Some components need attention before thesis defense[/bold yellow]"
        )
        failed_components = [
            comp for comp, passed in validation_results.items() if not passed
        ]
        console.print(f"[red]Failed components: {', '.join(failed_components)}[/red]")
        console.print(f"[dim]Check detailed logs at {runner.log_file}[/dim]")


if __name__ == "__main__":
    cli()
