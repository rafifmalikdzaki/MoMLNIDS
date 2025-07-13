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

