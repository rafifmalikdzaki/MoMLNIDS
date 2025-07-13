#!/usr/bin/env python3
"""
Enhanced MoMLNIDS Results Visualizer
A comprehensive tool for visualizing and analyzing MoMLNIDS experiment results with interactive grouping.
"""

import re
import json
import numpy as np
import signal
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
from datetime import datetime
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.columns import Columns
from rich.tree import Tree
from rich.rule import Rule
from rich.syntax import Syntax
from rich.align import Align
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm

console = Console()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    console.print(
        "\n[yellow]👋 Received interrupt signal. Shutting down gracefully...[/yellow]"
    )
    console.print(
        "[bold blue]Thank you for using Enhanced MoMLNIDS Results Visualizer![/bold blue]"
    )
    sys.exit(0)


# Set up signal handler for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)


class EnhancedMoMLNIDSAnalyzer:
    """Enhanced analyzer for MoMLNIDS experiment results with comprehensive grouping."""

    def __init__(self, base_dir: Path = None):
        self.base_dir = Path(base_dir) if base_dir else Path(".")
        self.experiment_groups = {}
        self.all_experiments = {}

    def discover_experiments(self) -> Dict[str, Dict]:
        """Discover all experiment directories and parse their results."""
        experiment_dirs = [
            "Training_results",
            "ProperTraining",
            "ProperTraining50Epoch",
            "results",
        ]

        discovered = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Discovering experiments...", total=None)

            for exp_dir in experiment_dirs:
                exp_path = self.base_dir / exp_dir
                if exp_path.exists():
                    discovered[exp_dir] = self._parse_experiment_directory(exp_path)

        self.experiment_groups = discovered
        self._flatten_experiments()
        return discovered

    def _parse_experiment_directory(self, exp_path: Path) -> Dict[str, Any]:
        """Parse an experiment directory and extract all results."""
        experiments = {}

        # Handle different directory structures
        if exp_path.name == "results":
            # Handle JSON results format
            for json_file in exp_path.rglob("*.json"):
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        experiment_name = f"{json_file.parent.name}_{json_file.stem}"
                        experiments[experiment_name] = {
                            "type": "json",
                            "path": json_file,
                            "data": data,
                            "modified": datetime.fromtimestamp(
                                json_file.stat().st_mtime
                            ),
                        }
                except Exception as e:
                    console.print(f"[red]Error loading {json_file}: {e}[/red]")
        else:
            # Handle log-based results format
            for log_dir in exp_path.rglob("*"):
                if log_dir.is_dir() and any(log_dir.glob("*.log")):
                    experiment_name = log_dir.name
                    experiment_data = self._parse_log_directory(log_dir)
                    if experiment_data:
                        experiments[experiment_name] = experiment_data

        return experiments

    def _parse_log_directory(self, log_dir: Path) -> Optional[Dict[str, Any]]:
        """Parse a directory containing training logs."""
        try:
            experiment = {
                "type": "logs",
                "path": log_dir,
                "modified": datetime.fromtimestamp(log_dir.stat().st_mtime),
                "source_training": None,
                "target_performance": None,
                "val_performance": None,
                "clustering": None,
                "config": self._extract_config_from_path(log_dir),
            }

            # Parse different log files
            if (log_dir / "source_trained.log").exists():
                experiment["source_training"] = self._parse_source_log(
                    log_dir / "source_trained.log"
                )

            if (log_dir / "target_performance.log").exists():
                experiment["target_performance"] = self._parse_performance_log(
                    log_dir / "target_performance.log"
                )

            if (log_dir / "val_performance.log").exists():
                experiment["val_performance"] = self._parse_performance_log(
                    log_dir / "val_performance.log"
                )

            if (log_dir / "clustering.log").exists():
                experiment["clustering"] = self._parse_clustering_log(
                    log_dir / "clustering.log"
                )

            return experiment

        except Exception as e:
            console.print(f"[red]Error parsing {log_dir}: {e}[/red]")
            return None

    def _extract_config_from_path(self, path: Path) -> Dict[str, Any]:
        """Extract detailed configuration from directory path."""
        path_str = str(path)
        config = {
            "experiment_name": path.name,
            "dataset": None,
            "dataset_short": None,
            "method": None,
            "method_detailed": None,
            "clusters": None,
            "architecture": None,
            "weighting": None,
            "epochs": None,
            "source_dataset": None,
            "target_dataset": None,
            "full_method_description": None,
        }

        # Extract dataset with improved detection
        dataset_mappings = {
            "NF-CSE-CIC-IDS2018-v2": "CSE-CIC-IDS2018",
            "NF-ToN-IoT-v2": "ToN-IoT",
            "NF-UNSW-NB15-v2": "UNSW-NB15",
            "NF-BoT-IoT-v2": "BoT-IoT",
        }

        for full_dataset, short_dataset in dataset_mappings.items():
            if full_dataset in path_str:
                config["dataset"] = full_dataset
                config["dataset_short"] = short_dataset
                break

        # Extract method with more detail
        if "PseudoLabelling" in path_str:
            config["method"] = "PseudoLabelling"
            # Look for cluster information
            cluster_match = re.search(r"Cluster_(\d+)", path_str)
            if cluster_match:
                config["clusters"] = int(cluster_match.group(1))
                config["method_detailed"] = (
                    f"PseudoLabelling (K={cluster_match.group(1)})"
                )
                config["full_method_description"] = (
                    f"PseudoLabelling with {cluster_match.group(1)} clusters"
                )
            else:
                config["method_detailed"] = "PseudoLabelling"
                config["full_method_description"] = "PseudoLabelling (clusters unknown)"
        elif "SingleLayer" in path_str:
            config["method"] = "SingleLayer"
            config["method_detailed"] = "SingleLayer"
            config["full_method_description"] = "SingleLayer domain adaptation"

        # Extract architecture with more patterns
        arch_patterns = [
            r"(\d+-\d+-\d+-\d+)",  # Standard 4-layer
            r"(\d+-\d+-\d+)",  # 3-layer
            r"(\d+-\d+)",  # 2-layer
        ]
        for pattern in arch_patterns:
            arch_match = re.search(pattern, path_str)
            if arch_match:
                config["architecture"] = arch_match.group(1)
                break

        # Extract weighting strategy
        if "DomainWeight" in path_str:
            config["weighting"] = "DomainWeight"
        elif "No-Weighting" in path_str:
            config["weighting"] = "No-Weighting"
        elif "Equal" in path_str:
            config["weighting"] = "Equal"

        # Extract epoch information from path
        epoch_match = re.search(r"(\d+)Epoch", path_str)
        if epoch_match:
            config["epochs"] = int(epoch_match.group(1))

        # Try to extract source and target from complex paths
        # Look for patterns like Source_Dataset_Target_Dataset
        source_target_match = re.search(r"(NF-[^_]+).*?(NF-[^_/]+)", path_str)
        if source_target_match:
            config["source_dataset"] = source_target_match.group(1)
            config["target_dataset"] = source_target_match.group(2)

        return config

    def _parse_source_log(self, log_file: Path) -> List[Dict[str, Any]]:
        """Parse source training log file."""
        training_data = []
        with open(log_file, "r") as f:
            content = f.read()

        # Parse training epochs
        epochs = re.findall(
            r"Train: Epoch: (\d+)/\d+ \| Alpha: ([\d.]+) \|.*?"
            r"LClass: ([\d.]+) \| Acc Class: ([\d.]+) \|.*?"
            r"LDomain: ([\d.]+) \| Acc Domain: ([\d.]+) \|.*?"
            r"Loss Entropy: ([\d.]+) \|.*?"
            r"F1 Score: ([\d.]+) Precision: ([\d.]+) Recall: ([\d.]+)",
            content,
            re.DOTALL,
        )

        for epoch_data in epochs:
            training_data.append(
                {
                    "epoch": int(epoch_data[0]),
                    "alpha": float(epoch_data[1]),
                    "loss_class": float(epoch_data[2]),
                    "acc_class": float(epoch_data[3]),
                    "loss_domain": float(epoch_data[4]),
                    "acc_domain": float(epoch_data[5]),
                    "loss_entropy": float(epoch_data[6]),
                    "f1_score": float(epoch_data[7]),
                    "precision": float(epoch_data[8]),
                    "recall": float(epoch_data[9]),
                }
            )

        return training_data

    def _parse_performance_log(self, log_file: Path) -> List[Dict[str, Any]]:
        """Parse target/validation performance log file."""
        performance_data = []
        with open(log_file, "r") as f:
            for line in f:
                match = re.match(
                    r"Eval: Epoch: (\d+) Loss: ([\d.]+) Acc: ([\d.]+) F1 Score: ([\d.]+) Precision: ([\d.]+) Recall: ([\d.]+)",
                    line.strip(),
                )
                if match:
                    performance_data.append(
                        {
                            "epoch": int(match.group(1)),
                            "loss": float(match.group(2)),
                            "accuracy": float(match.group(3)),
                            "f1_score": float(match.group(4)),
                            "precision": float(match.group(5)),
                            "recall": float(match.group(6)),
                        }
                    )

        return performance_data

    def _parse_clustering_log(self, log_file: Path) -> Dict[str, Any]:
        """Parse clustering log file."""
        clustering_data = {}
        with open(log_file, "r") as f:
            content = f.read()

        # Extract clustering information
        cluster_match = re.search(r"Number of clusters: (\d+)", content)
        if cluster_match:
            clustering_data["num_clusters"] = int(cluster_match.group(1))

        return clustering_data

    def _flatten_experiments(self):
        """Flatten experiment groups into a single dictionary."""
        for group_name, experiments in self.experiment_groups.items():
            for exp_name, exp_data in experiments.items():
                full_name = f"{group_name}::{exp_name}"
                self.all_experiments[full_name] = {**exp_data, "group": group_name}

    def create_experiment_tree(self) -> Tree:
        """Create a tree view of all experiments."""
        tree = Tree("[bold blue]MoMLNIDS Experiment Results[/bold blue]")

        for group_name, experiments in self.experiment_groups.items():
            group_node = tree.add(
                f"[bold green]{group_name}[/bold green] ({len(experiments)} experiments)"
            )

            # Group by dataset
            dataset_groups = defaultdict(list)
            for exp_name, exp_data in experiments.items():
                if exp_data["type"] == "logs" and exp_data.get("config"):
                    dataset = exp_data["config"].get("dataset", "Unknown")
                else:
                    dataset = "JSON Results"
                dataset_groups[dataset].append((exp_name, exp_data))

            for dataset, exp_list in dataset_groups.items():
                dataset_node = group_node.add(
                    f"[cyan]{dataset}[/cyan] ({len(exp_list)} experiments)"
                )
                for exp_name, exp_data in exp_list:
                    exp_node = dataset_node.add(f"[white]{exp_name}[/white]")

                    # Add performance summary if available
                    if exp_data["type"] == "logs" and exp_data.get(
                        "target_performance"
                    ):
                        best_perf = max(
                            exp_data["target_performance"], key=lambda x: x["accuracy"]
                        )
                        exp_node.add(
                            f"[dim]Best Acc: {best_perf['accuracy']:.3f}, F1: {best_perf['f1_score']:.3f}[/dim]"
                        )
                    elif exp_data["type"] == "json":
                        # Add JSON results summary
                        exp_node.add(f"[dim]JSON format results[/dim]")

        return tree

    def create_comparison_table(self, experiments: List[str]) -> Table:
        """Create a detailed comparison table for selected experiments."""
        table = Table(
            title="Detailed Experiment Comparison",
            show_header=True,
            header_style="bold blue",
        )
        table.add_column("Experiment", style="cyan bold", width=25)
        table.add_column("Dataset", style="yellow", width=15)
        table.add_column("Method", style="green", width=20)
        table.add_column("Clusters", style="purple", justify="center", width=8)
        table.add_column("Best Acc", style="red", justify="center", width=10)
        table.add_column("Best F1", style="magenta", justify="center", width=10)
        table.add_column("Precision", style="orange3", justify="center", width=10)
        table.add_column("Recall", style="blue", justify="center", width=10)
        table.add_column("Final Loss", style="cyan", justify="center", width=10)
        table.add_column("Epochs", style="white", justify="center", width=8)

        for exp_name in experiments:
            if exp_name in self.all_experiments:
                exp_data = self.all_experiments[exp_name]

                # Extract metrics based on type
                if exp_data["type"] == "logs":
                    config = exp_data.get("config", {})
                    dataset = config.get("dataset_short", config.get("dataset", "N/A"))
                    method = config.get("method_detailed", config.get("method", "N/A"))
                    clusters = (
                        str(config.get("clusters", "N/A"))
                        if config.get("clusters")
                        else "N/A"
                    )

                    if exp_data.get("target_performance"):
                        perfs = exp_data["target_performance"]
                        best_acc = max(perfs, key=lambda x: x["accuracy"])
                        best_f1 = max(perfs, key=lambda x: x["f1_score"])
                        final_perf = perfs[-1] if perfs else None

                        best_acc_val = f"{best_acc['accuracy']:.3f}"
                        best_f1_val = f"{best_f1['f1_score']:.3f}"
                        precision_val = f"{best_acc['precision']:.3f}"  # Use precision from best accuracy performance
                        recall_val = f"{best_acc['recall']:.3f}"  # Use recall from best accuracy performance
                        final_loss_val = (
                            f"{final_perf['loss']:.3f}" if final_perf else "N/A"
                        )
                        epochs = str(len(perfs))
                    else:
                        best_acc_val = best_f1_val = precision_val = recall_val = (
                            final_loss_val
                        ) = epochs = "N/A"

                elif exp_data["type"] == "json":
                    dataset = "JSON"
                    method = "JSON Results"
                    clusters = "N/A"
                    # Try to extract from JSON structure
                    data = exp_data.get("data", {})
                    if "evaluation_results" in data:
                        # Extract metrics from JSON
                        best_acc_val = best_f1_val = precision_val = recall_val = (
                            final_loss_val
                        ) = epochs = "JSON"
                    else:
                        best_acc_val = best_f1_val = precision_val = recall_val = (
                            final_loss_val
                        ) = epochs = "N/A"

                # Color code performance
                def format_metric(value_str: str) -> str:
                    if value_str == "N/A" or value_str == "JSON":
                        return value_str
                    try:
                        value = float(value_str)
                        if value >= 0.9:
                            return f"[bold green]{value_str}[/bold green]"
                        elif value >= 0.8:
                            return f"[green]{value_str}[/green]"
                        elif value >= 0.6:
                            return f"[yellow]{value_str}[/yellow]"
                        else:
                            return f"[red]{value_str}[/red]"
                    except:
                        return value_str

                # Color code loss (lower is better)
                def format_loss(value_str: str) -> str:
                    if value_str == "N/A" or value_str == "JSON":
                        return value_str
                    try:
                        value = float(value_str)
                        if value <= 0.1:
                            return f"[bold green]{value_str}[/bold green]"
                        elif value <= 0.3:
                            return f"[green]{value_str}[/green]"
                        elif value <= 0.5:
                            return f"[yellow]{value_str}[/yellow]"
                        else:
                            return f"[red]{value_str}[/red]"
                    except:
                        return value_str

                # Color code clusters
                def format_clusters(clusters_str: str) -> str:
                    if clusters_str == "N/A":
                        return "[dim]N/A[/dim]"
                    else:
                        return f"[bold purple]{clusters_str}[/bold purple]"

                table.add_row(
                    exp_name.split("::")[-1][:22] + "..."
                    if len(exp_name.split("::")[-1]) > 25
                    else exp_name.split("::")[-1],
                    dataset.replace("NF-", "").replace("-v2", ""),
                    method,
                    format_clusters(clusters),
                    format_metric(best_acc_val),
                    format_metric(best_f1_val),
                    format_metric(precision_val),
                    format_metric(recall_val),
                    format_loss(final_loss_val),
                    epochs,
                )

        return table

    def create_training_progress_chart(self, experiment: str) -> Panel:
        """Create a training progress visualization."""
        if experiment not in self.all_experiments:
            return Panel("Experiment not found", title="Error", border_style="red")

        exp_data = self.all_experiments[experiment]

        if exp_data["type"] != "logs" or not exp_data.get("target_performance"):
            return Panel(
                "No training data available",
                title="Training Progress",
                border_style="yellow",
            )

        perfs = exp_data["target_performance"]
        if not perfs:
            return Panel(
                "No performance data", title="Training Progress", border_style="yellow"
            )

        # Create ASCII chart
        chart_lines = []
        chart_lines.append("[bold]Training Progress (Target Performance)[/bold]")
        chart_lines.append("")

        # Get accuracy progression
        accuracies = [p["accuracy"] for p in perfs]
        f1_scores = [p["f1_score"] for p in perfs]
        losses = [p["loss"] for p in perfs]

        max_acc = max(accuracies)
        max_f1 = max(f1_scores)
        min_loss = min(losses)

        chart_lines.append(
            f"Best Accuracy: [bold green]{max_acc:.4f}[/bold green] (Epoch {accuracies.index(max_acc)})"
        )
        chart_lines.append(
            f"Best F1 Score: [bold blue]{max_f1:.4f}[/bold blue] (Epoch {f1_scores.index(max_f1)})"
        )
        chart_lines.append(
            f"Best Loss: [bold green]{min_loss:.4f}[/bold green] (Epoch {losses.index(min_loss)})"
        )
        chart_lines.append(f"Final Accuracy: [yellow]{accuracies[-1]:.4f}[/yellow]")
        chart_lines.append(f"Final Loss: [yellow]{losses[-1]:.4f}[/yellow]")
        chart_lines.append(f"Total Epochs: {len(perfs)}")
        chart_lines.append("")

        # Simple progress visualization
        chart_lines.append("[bold]Last 10 Epochs Accuracy:[/bold]")
        recent_perfs = perfs[-10:] if len(perfs) > 10 else perfs
        for i, perf in enumerate(recent_perfs):
            epoch_num = len(perfs) - len(recent_perfs) + i
            bar_length = int(perf["accuracy"] * 50)  # Scale to 50 chars
            bar = "█" * bar_length + "░" * (50 - bar_length)
            chart_lines.append(
                f"Epoch {epoch_num:2d}: [{self._get_color(perf['accuracy'])}]{bar}[/{self._get_color(perf['accuracy'])}] {perf['accuracy']:.3f}"
            )

        chart_lines.append("")
        chart_lines.append("[bold]Last 10 Epochs F1-Score:[/bold]")
        for i, perf in enumerate(recent_perfs):
            epoch_num = len(perfs) - len(recent_perfs) + i
            bar_length = int(perf["f1_score"] * 50)  # Scale to 50 chars
            bar = "█" * bar_length + "░" * (50 - bar_length)
            chart_lines.append(
                f"Epoch {epoch_num:2d}: [{self._get_f1_color(perf['f1_score'])}]{bar}[/{self._get_f1_color(perf['f1_score'])}] {perf['f1_score']:.3f}"
            )

        return Panel(
            "\n".join(chart_lines),
            title=f"[bold]Training Progress: {experiment.split('::')[-1]}[/bold]",
            border_style="green",
        )

    def create_loss_progression_chart(self, experiment: str) -> Panel:
        """Create a comprehensive loss progression visualization for train and validation sets."""
        if experiment not in self.all_experiments:
            return Panel("Experiment not found", title="Error", border_style="red")

        exp_data = self.all_experiments[experiment]

        if exp_data["type"] != "logs":
            return Panel(
                "No training data available",
                title="Loss Progression",
                border_style="yellow",
            )

        # Get training data (source training)
        train_data = exp_data.get("source_training", [])
        val_data = exp_data.get("target_performance", [])

        if not train_data and not val_data:
            return Panel(
                "No loss data available",
                title="Loss Progression",
                border_style="yellow",
            )

        chart_lines = []
        chart_lines.append("[bold]Loss Progression Analysis[/bold]")
        chart_lines.append("")

        # Training Loss Analysis
        if train_data:
            train_class_losses = [d["loss_class"] for d in train_data]
            train_domain_losses = [d["loss_domain"] for d in train_data]
            train_entropy_losses = [d["loss_entropy"] for d in train_data]

            chart_lines.append("[bold cyan]📈 Training Loss Summary:[/bold cyan]")
            chart_lines.append(
                f"• Classification Loss - Initial: {train_class_losses[0]:.4f}, Final: {train_class_losses[-1]:.4f}, Best: {min(train_class_losses):.4f}"
            )
            chart_lines.append(
                f"• Domain Loss - Initial: {train_domain_losses[0]:.4f}, Final: {train_domain_losses[-1]:.4f}, Best: {min(train_domain_losses):.4f}"
            )
            chart_lines.append(
                f"• Entropy Loss - Initial: {train_entropy_losses[0]:.4f}, Final: {train_entropy_losses[-1]:.4f}, Best: {min(train_entropy_losses):.4f}"
            )
            chart_lines.append("")

            # Calculate improvement
            class_improvement = train_class_losses[0] - train_class_losses[-1]
            domain_improvement = train_domain_losses[0] - train_domain_losses[-1]
            entropy_improvement = train_entropy_losses[0] - train_entropy_losses[-1]

            chart_lines.append(
                "[bold green]📊 Training Loss Improvements:[/bold green]"
            )
            chart_lines.append(
                f"• Classification: {'+' if class_improvement > 0 else ''}{class_improvement:.4f} ({'✅ Improved' if class_improvement > 0 else '❌ Worsened'})"
            )
            chart_lines.append(
                f"• Domain: {'+' if domain_improvement > 0 else ''}{domain_improvement:.4f} ({'✅ Improved' if domain_improvement > 0 else '❌ Worsened'})"
            )
            chart_lines.append(
                f"• Entropy: {'+' if entropy_improvement > 0 else ''}{entropy_improvement:.4f} ({'✅ Improved' if entropy_improvement > 0 else '❌ Worsened'})"
            )
            chart_lines.append("")

        # Validation Loss Analysis
        if val_data:
            val_losses = [d["loss"] for d in val_data]
            chart_lines.append("[bold blue]📉 Validation Loss Summary:[/bold blue]")
            chart_lines.append(f"• Initial: {val_losses[0]:.4f}")
            chart_lines.append(f"• Final: {val_losses[-1]:.4f}")
            chart_lines.append(
                f"• Best: {min(val_losses):.4f} (Epoch {val_losses.index(min(val_losses))})"
            )
            chart_lines.append(
                f"• Worst: {max(val_losses):.4f} (Epoch {val_losses.index(max(val_losses))})"
            )

            val_improvement = val_losses[0] - val_losses[-1]
            chart_lines.append(
                f"• Overall Improvement: {'+' if val_improvement > 0 else ''}{val_improvement:.4f}"
            )
            chart_lines.append("")

        # Loss Progression Visualization
        if val_data:
            chart_lines.append(
                "[bold]📈 Validation Loss Progression (Last 15 Epochs):[/bold]"
            )
            recent_val = val_data[-15:] if len(val_data) > 15 else val_data

            # Normalize losses for visualization (0-50 chars)
            val_losses_recent = [d["loss"] for d in recent_val]
            if val_losses_recent:
                max_loss = max(val_losses_recent)
                min_loss = min(val_losses_recent)
                loss_range = max_loss - min_loss if max_loss != min_loss else 1

                for i, perf in enumerate(recent_val):
                    epoch_num = len(val_data) - len(recent_val) + i
                    # Invert the bar length (shorter bar = better loss)
                    normalized_loss = (perf["loss"] - min_loss) / loss_range
                    bar_length = int(
                        (1 - normalized_loss) * 40
                    )  # Scale to 40 chars, inverted
                    bar = "█" * bar_length + "░" * (40 - bar_length)

                    # Color based on loss value
                    loss_color = self._get_loss_color(perf["loss"])
                    chart_lines.append(
                        f"Epoch {epoch_num:2d}: [{loss_color}]{bar}[/{loss_color}] {perf['loss']:.4f}"
                    )
                chart_lines.append("")

        # Training loss comparison visualization
        if train_data:
            chart_lines.append(
                "[bold]🔄 Training Loss Components (Last 10 Epochs):[/bold]"
            )
            recent_train = train_data[-10:] if len(train_data) > 10 else train_data

            for i, train_perf in enumerate(recent_train):
                epoch_num = len(train_data) - len(recent_train) + i

                # Visualize classification loss
                class_bar_length = max(
                    1, int((1 - min(1, train_perf["loss_class"])) * 25)
                )
                class_bar = "█" * class_bar_length + "░" * (25 - class_bar_length)

                # Visualize domain loss
                domain_bar_length = max(
                    1, int((1 - min(1, train_perf["loss_domain"])) * 25)
                )
                domain_bar = "█" * domain_bar_length + "░" * (25 - domain_bar_length)

                chart_lines.append(f"Epoch {epoch_num:2d}:")
                chart_lines.append(
                    f"  Class:  [green]{class_bar}[/green] {train_perf['loss_class']:.4f}"
                )
                chart_lines.append(
                    f"  Domain: [blue]{domain_bar}[/blue] {train_perf['loss_domain']:.4f}"
                )
                chart_lines.append("")

        # Loss stability analysis
        if val_data and len(val_data) > 5:
            recent_losses = [d["loss"] for d in val_data[-10:]]
            loss_std = np.std(recent_losses)
            loss_trend = recent_losses[-1] - recent_losses[0]

            chart_lines.append(
                "[bold yellow]📊 Loss Stability Analysis (Last 10 Epochs):[/bold yellow]"
            )
            chart_lines.append(f"• Standard Deviation: {loss_std:.4f}")
            chart_lines.append(
                f"• Trend: {'+' if loss_trend > 0 else ''}{loss_trend:.4f}"
            )

            if loss_std < 0.01:
                chart_lines.append("• Status: [green]✅ Very Stable[/green]")
            elif loss_std < 0.05:
                chart_lines.append("• Status: [yellow]⚠️ Moderately Stable[/yellow]")
            else:
                chart_lines.append("• Status: [red]⚡ Unstable[/red]")

            if abs(loss_trend) < 0.01:
                chart_lines.append("• Convergence: [green]✅ Converged[/green]")
            elif loss_trend < 0:
                chart_lines.append("• Convergence: [blue]📉 Still Improving[/blue]")
            else:
                chart_lines.append("• Convergence: [red]📈 Diverging[/red]")

        return Panel(
            "\n".join(chart_lines),
            title=f"[bold]Loss Progression: {experiment.split('::')[-1]}[/bold]",
            border_style="cyan",
        )

    def _get_color(self, value: float) -> str:
        """Get color based on performance value."""
        if value >= 0.8:
            return "green"
        elif value >= 0.6:
            return "yellow"
        else:
            return "red"

    def _get_f1_color(self, value: float) -> str:
        """Get blue-based color for F1-Score performance value."""
        if value >= 0.8:
            return "blue"
        elif value >= 0.6:
            return "cyan"
        else:
            return "bright_blue"

    def _get_loss_color(self, value: float) -> str:
        """Get color based on loss value (lower is better)."""
        if value <= 0.1:
            return "bold green"
        elif value <= 0.3:
            return "green"
        elif value <= 0.5:
            return "yellow"
        elif value <= 1.0:
            return "orange3"
        else:
            return "red"

    def create_dataset_performance_breakdown(self) -> List[Panel]:
        """Create dataset-specific performance breakdown panels."""
        panels = []

        # Group experiments by dataset
        dataset_groups = defaultdict(list)
        for exp_name, exp_data in self.all_experiments.items():
            if exp_data["type"] == "logs":
                config = exp_data.get("config", {})
                dataset = config.get("dataset_short", config.get("dataset", "Unknown"))
                if exp_data.get("target_performance"):
                    perfs = exp_data["target_performance"]
                    best_perf = max(perfs, key=lambda x: x["accuracy"])
                    final_perf = perfs[-1] if perfs else best_perf
                    dataset_groups[dataset].append(
                        {
                            "name": exp_name.split("::")[-1],
                            "method": config.get(
                                "full_method_description",
                                config.get("method", "Unknown"),
                            ),
                            "clusters": config.get("clusters"),
                            "accuracy": best_perf["accuracy"],
                            "f1_score": best_perf["f1_score"],
                            "precision": best_perf["precision"],
                            "recall": best_perf["recall"],
                            "final_loss": final_perf["loss"],
                            "epochs": len(perfs),
                        }
                    )

        # Create panel for each dataset
        for dataset, experiments in dataset_groups.items():
            if not experiments:
                continue

            # Sort by accuracy
            experiments.sort(key=lambda x: x["accuracy"], reverse=True)

            # Create table for this dataset
            table = Table(
                title=f"{dataset} Performance Results",
                show_header=True,
                header_style="bold cyan",
                title_style="bold yellow",
            )
            table.add_column("Rank", style="bold", width=5)
            table.add_column("Method", style="green", width=25)
            table.add_column("Clusters", style="purple", justify="center", width=8)
            table.add_column("Accuracy", style="red", justify="center", width=10)
            table.add_column("F1 Score", style="blue", justify="center", width=10)
            table.add_column("Precision", style="yellow", justify="center", width=10)
            table.add_column("Recall", style="orange3", justify="center", width=10)
            table.add_column("Final Loss", style="cyan", justify="center", width=10)

            for i, exp in enumerate(experiments[:10], 1):  # Top 10
                rank_style = "bold gold1" if i <= 3 else "white"
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""

                clusters_str = str(exp["clusters"]) if exp["clusters"] else "N/A"
                if clusters_str != "N/A":
                    clusters_display = f"[bold purple]{clusters_str}[/bold purple]"
                else:
                    clusters_display = "[dim]N/A[/dim]"

                # Color code loss (lower is better)
                def format_loss(loss_val: float) -> str:
                    if loss_val <= 0.1:
                        return f"[bold green]{loss_val:.3f}[/bold green]"
                    elif loss_val <= 0.3:
                        return f"[green]{loss_val:.3f}[/green]"
                    elif loss_val <= 0.5:
                        return f"[yellow]{loss_val:.3f}[/yellow]"
                    else:
                        return f"[red]{loss_val:.3f}[/red]"

                table.add_row(
                    f"[{rank_style}]{i}{medal}[/{rank_style}]",
                    exp["method"][:22] + "..."
                    if len(exp["method"]) > 25
                    else exp["method"],
                    clusters_display,
                    f"[bold green]{exp['accuracy']:.4f}[/bold green]"
                    if exp["accuracy"] >= 0.9
                    else f"[green]{exp['accuracy']:.4f}[/green]"
                    if exp["accuracy"] >= 0.8
                    else f"[yellow]{exp['accuracy']:.4f}[/yellow]"
                    if exp["accuracy"] >= 0.6
                    else f"[red]{exp['accuracy']:.4f}[/red]",
                    f"{exp['f1_score']:.4f}",
                    f"{exp['precision']:.4f}",
                    f"{exp['recall']:.4f}",
                    format_loss(exp["final_loss"]),
                )

            # Add summary statistics
            avg_acc = sum(exp["accuracy"] for exp in experiments) / len(experiments)
            max_acc = max(exp["accuracy"] for exp in experiments)
            min_acc = min(exp["accuracy"] for exp in experiments)
            avg_loss = sum(exp["final_loss"] for exp in experiments) / len(experiments)
            min_loss = min(exp["final_loss"] for exp in experiments)

            summary_text = f"""
[bold]Dataset Summary:[/bold]
• Total Experiments: {len(experiments)}
• Best Accuracy: [bold green]{max_acc:.4f}[/bold green]
• Average Accuracy: {avg_acc:.4f}
• Worst Accuracy: [red]{min_acc:.4f}[/red]
• Performance Range: {max_acc - min_acc:.4f}
• Best Final Loss: [bold green]{min_loss:.4f}[/bold green]
• Average Final Loss: {avg_loss:.4f}
            """

            # Combine table and summary
            panel_content = Columns(
                [
                    table,
                    Panel(
                        summary_text.strip(), title="Statistics", border_style="blue"
                    ),
                ]
            )
            panels.append(
                Panel(
                    panel_content,
                    title=f"📊 {dataset} Dataset Results",
                    border_style="cyan",
                )
            )

        return panels

    def interactive_explorer(self):
        """Interactive explorer for experiments."""
        try:
            self.discover_experiments()

            if not self.all_experiments:
                console.print("[red]No experiments found![/red]")
                return

            while True:
                try:
                    console.clear()
                    console.print(
                        Rule(
                            "[bold blue]Enhanced MoMLNIDS Results Explorer[/bold blue]"
                        )
                    )
                    console.print()

                    # Show experiment tree
                    tree = self.create_experiment_tree()
                    console.print(tree)
                    console.print()

                    console.print("[bold blue]Available Actions:[/bold blue]")
                    console.print("1. 📊 Compare experiments by group")
                    console.print("2. 🔍 Analyze specific experiment")
                    console.print("3. 📈 Show training progress")
                    console.print("4. 📉 Show loss progression")
                    console.print("5. 🏆 Show best performers")
                    console.print("6. 📋 Dataset-specific performance breakdown")
                    console.print("7. 🔬 Advanced method comparison")
                    console.print("8. 💾 Export comparison report")
                    console.print("9. 🌳 Refresh experiment tree")
                    console.print("10. 🚪 Exit")
                    console.print()

                    choice = Prompt.ask(
                        "Enter your choice",
                        choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                    )

                    if choice == "1":
                        self._compare_by_group()
                    elif choice == "2":
                        self._analyze_specific_experiment()
                    elif choice == "3":
                        self._show_training_progress()
                    elif choice == "4":
                        self._show_loss_progression()
                    elif choice == "5":
                        self._show_best_performers()
                    elif choice == "6":
                        self._show_dataset_breakdown()
                    elif choice == "7":
                        self._show_method_comparison()
                    elif choice == "8":
                        self._export_comparison_report()
                    elif choice == "9":
                        console.print("[yellow]Refreshing experiments...[/yellow]")
                        self.discover_experiments()
                        console.print("[green]✅ Refreshed![/green]")
                    elif choice == "10":
                        console.print("[bold green]👋 Goodbye![/bold green]")
                        break

                    if choice != "10":
                        try:
                            Prompt.ask("\nPress Enter to continue")
                        except (KeyboardInterrupt, EOFError):
                            console.print(
                                "\n[yellow]Returning to main menu...[/yellow]"
                            )
                            continue

                except (KeyboardInterrupt, EOFError):
                    console.print(
                        "\n[yellow]👋 Received interrupt signal. Exiting gracefully...[/yellow]"
                    )
                    break
                except Exception as e:
                    console.print(f"\n[red]❌ An error occurred: {e}[/red]")
                    console.print("[yellow]Returning to main menu...[/yellow]")
                    try:
                        Prompt.ask("\nPress Enter to continue")
                    except (KeyboardInterrupt, EOFError):
                        break

        except KeyboardInterrupt:
            console.print(
                "\n[yellow]👋 Received interrupt signal. Shutting down gracefully...[/yellow]"
            )
        except Exception as e:
            console.print(f"\n[red]❌ A critical error occurred: {e}[/red]")
        finally:
            console.print(
                "[bold blue]Thank you for using Enhanced MoMLNIDS Results Visualizer![/bold blue]"
            )

    def _compare_by_group(self):
        """Compare experiments by group."""
        try:
            console.print(
                "\n[bold blue]Select experiment group to compare:[/bold blue]"
            )
            groups = list(self.experiment_groups.keys())

            for i, group in enumerate(groups, 1):
                count = len(self.experiment_groups[group])
                console.print(f"{i}. {group} ({count} experiments)")

            try:
                choice = int(Prompt.ask("Enter group number")) - 1
                if 0 <= choice < len(groups):
                    selected_group = groups[choice]
                    experiments = [
                        f"{selected_group}::{exp}"
                        for exp in self.experiment_groups[selected_group].keys()
                    ]

                    console.print(
                        f"\n[bold green]Comparing {selected_group} experiments:[/bold green]"
                    )
                    table = self.create_comparison_table(experiments)
                    console.print(table)
                else:
                    console.print("[red]Invalid choice![/red]")
            except ValueError:
                console.print("[red]Please enter a valid number![/red]")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Operation cancelled.[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Error comparing experiments: {e}[/red]")

    def _analyze_specific_experiment(self):
        """Analyze a specific experiment in detail."""
        try:
            console.print(
                "\n[bold blue]Enter experiment name or part of it:[/bold blue]"
            )
            search_term = Prompt.ask("Search").lower()

            matches = [
                name
                for name in self.all_experiments.keys()
                if search_term in name.lower()
            ]

            if not matches:
                console.print("[red]No matching experiments found![/red]")
                return

            if len(matches) == 1:
                selected = matches[0]
            else:
                console.print("\n[bold blue]Multiple matches found:[/bold blue]")
                for i, match in enumerate(matches[:10], 1):  # Limit to 10
                    console.print(f"{i}. {match}")

                try:
                    choice = int(Prompt.ask("Select experiment number")) - 1
                    if 0 <= choice < len(matches):
                        selected = matches[choice]
                    else:
                        console.print("[red]Invalid choice![/red]")
                        return
                except ValueError:
                    console.print("[red]Please enter a valid number![/red]")
                    return
                except (KeyboardInterrupt, EOFError):
                    console.print("\n[yellow]Operation cancelled.[/yellow]")
                    return

            # Show detailed analysis
            console.print(f"\n[bold green]Detailed Analysis: {selected}[/bold green]")

            exp_data = self.all_experiments[selected]

            if exp_data["type"] == "logs":
                # Show enhanced configuration
                config = exp_data.get("config", {})
                console.print("\n[bold blue]📋 Configuration Details:[/bold blue]")

                config_table = Table(show_header=False, box=None, padding=(0, 2))
                config_table.add_column("Property", style="cyan bold")
                config_table.add_column("Value", style="white")

                config_table.add_row(
                    "Experiment Name", config.get("experiment_name", "N/A")
                )
                config_table.add_row("Dataset", config.get("dataset", "N/A"))
                config_table.add_row(
                    "Method",
                    config.get("full_method_description", config.get("method", "N/A")),
                )

                if config.get("clusters"):
                    config_table.add_row(
                        "Cluster Count",
                        f"[bold purple]{config['clusters']}[/bold purple]",
                    )

                if config.get("weighting"):
                    config_table.add_row("Weighting Strategy", config["weighting"])

                if config.get("epochs"):
                    config_table.add_row("Expected Epochs", str(config["epochs"]))

                if config.get("source_dataset"):
                    config_table.add_row("Source Dataset", config["source_dataset"])

                if config.get("target_dataset"):
                    config_table.add_row("Target Dataset", config["target_dataset"])

                console.print(config_table)

                # Show performance summary with more details
                if exp_data.get("target_performance"):
                    perfs = exp_data["target_performance"]
                    best_perf = max(perfs, key=lambda x: x["accuracy"])
                    best_f1_perf = max(perfs, key=lambda x: x["f1_score"])
                    final_perf = perfs[-1]

                    console.print(f"\n[bold blue]🎯 Performance Summary:[/bold blue]")

                    perf_table = Table(show_header=True, header_style="bold green")
                    perf_table.add_column("Metric", style="cyan bold")
                    perf_table.add_column("Best", style="green")
                    perf_table.add_column(
                        "Best Epoch", style="yellow", justify="center"
                    )
                    perf_table.add_column("Final", style="blue")
                    perf_table.add_column("Improvement", style="purple")

                    # Calculate improvements
                    acc_improvement = (
                        best_perf["accuracy"] - perfs[0]["accuracy"]
                        if len(perfs) > 1
                        else 0
                    )
                    f1_improvement = (
                        best_f1_perf["f1_score"] - perfs[0]["f1_score"]
                        if len(perfs) > 1
                        else 0
                    )

                    # Color code loss (lower is better)
                    def format_loss(loss_val: float) -> str:
                        if loss_val <= 0.1:
                            return f"[bold green]{loss_val:.4f}[/bold green]"
                        elif loss_val <= 0.3:
                            return f"[green]{loss_val:.4f}[/green]"
                        elif loss_val <= 0.5:
                            return f"[yellow]{loss_val:.4f}[/yellow]"
                        else:
                            return f"[red]{loss_val:.4f}[/red]"

                    perf_table.add_row(
                        "Accuracy",
                        f"{best_perf['accuracy']:.4f}",
                        str(best_perf["epoch"]),
                        f"{final_perf['accuracy']:.4f}",
                        f"+{acc_improvement:.4f}"
                        if acc_improvement > 0
                        else f"{acc_improvement:.4f}",
                    )

                    perf_table.add_row(
                        "F1 Score",
                        f"{best_f1_perf['f1_score']:.4f}",
                        str(best_f1_perf["epoch"]),
                        f"{final_perf['f1_score']:.4f}",
                        f"+{f1_improvement:.4f}"
                        if f1_improvement > 0
                        else f"{f1_improvement:.4f}",
                    )

                    perf_table.add_row(
                        "Precision",
                        f"{best_perf['precision']:.4f}",
                        str(best_perf["epoch"]),
                        f"{final_perf['precision']:.4f}",
                        "N/A",
                    )

                    perf_table.add_row(
                        "Recall",
                        f"{best_perf['recall']:.4f}",
                        str(best_perf["epoch"]),
                        f"{final_perf['recall']:.4f}",
                        "N/A",
                    )

                    perf_table.add_row(
                        "Loss",
                        format_loss(best_perf["loss"]),
                        str(best_perf["epoch"]),
                        format_loss(final_perf["loss"]),
                        "N/A",
                    )

                    console.print(perf_table)

                    # Show training stability metrics
                    if len(perfs) > 10:
                        recent_accs = [p["accuracy"] for p in perfs[-10:]]
                        recent_losses = [p["loss"] for p in perfs[-10:]]
                        acc_std = np.std(recent_accs)
                        acc_trend = recent_accs[-1] - recent_accs[0]
                        loss_trend = recent_losses[-1] - recent_losses[0]

                        console.print(
                            f"\n[bold blue]📈 Training Stability (Last 10 Epochs):[/bold blue]"
                        )
                        console.print(f"  Accuracy Std Dev: {acc_std:.4f}")
                        console.print(
                            f"  Accuracy Trend: {'+' if acc_trend > 0 else ''}{acc_trend:.4f}"
                        )
                        console.print(
                            f"  Loss Trend: {'+' if loss_trend > 0 else ''}{loss_trend:.4f}"
                        )

                        if acc_std < 0.01:
                            console.print("  [green]✅ Stable training[/green]")
                        elif acc_std < 0.02:
                            console.print("  [yellow]⚠️ Moderately stable[/yellow]")
                        else:
                            console.print("  [red]⚡ Unstable training[/red]")

            # Show training progress chart
            progress_chart = self.create_training_progress_chart(selected)
            console.print("\n")
            console.print(progress_chart)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Operation cancelled.[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Error analyzing experiment: {e}[/red]")

    def _show_dataset_breakdown(self):
        """Show dataset-specific performance breakdown."""
        console.print(
            "\n[bold green]📊 Dataset-Specific Performance Breakdown[/bold green]"
        )

        panels = self.create_dataset_performance_breakdown()

        if not panels:
            console.print("[red]No dataset performance data found![/red]")
            return

        for panel in panels:
            console.print(panel)
            console.print()

    def _show_method_comparison(self):
        """Show advanced method comparison."""
        console.print("\n[bold green]🔬 Advanced Method Comparison[/bold green]")

        # Group experiments by method and dataset
        method_performance = defaultdict(lambda: defaultdict(list))

        for exp_name, exp_data in self.all_experiments.items():
            if exp_data["type"] == "logs" and exp_data.get("target_performance"):
                config = exp_data.get("config", {})
                method = config.get(
                    "full_method_description", config.get("method", "Unknown")
                )
                dataset = config.get("dataset_short", config.get("dataset", "Unknown"))

                perfs = exp_data["target_performance"]
                best_perf = max(perfs, key=lambda x: x["accuracy"])

                method_performance[method][dataset].append(
                    {
                        "experiment": exp_name.split("::")[-1],
                        "accuracy": best_perf["accuracy"],
                        "f1_score": best_perf["f1_score"],
                        "clusters": config.get("clusters"),
                    }
                )

        # Create comparison table
        table = Table(
            title="Method Performance Across Datasets",
            show_header=True,
            header_style="bold blue",
        )
        table.add_column("Method", style="cyan bold", width=30)
        table.add_column("Dataset", style="yellow", width=15)
        table.add_column("Count", style="white", justify="center", width=7)
        table.add_column("Best Acc", style="green", justify="center", width=10)
        table.add_column("Avg Acc", style="blue", justify="center", width=10)
        table.add_column("Std Dev", style="purple", justify="center", width=10)
        table.add_column("Best F1", style="red", justify="center", width=10)

        for method, datasets in method_performance.items():
            for dataset, experiments in datasets.items():
                if experiments:
                    accuracies = [exp["accuracy"] for exp in experiments]
                    f1_scores = [exp["f1_score"] for exp in experiments]

                    best_acc = max(accuracies)
                    avg_acc = np.mean(accuracies)
                    std_acc = np.std(accuracies)
                    best_f1 = max(f1_scores)

                    def format_performance(
                        value: float, threshold_high=0.9, threshold_med=0.8
                    ) -> str:
                        if value >= threshold_high:
                            return f"[bold green]{value:.4f}[/bold green]"
                        elif value >= threshold_med:
                            return f"[green]{value:.4f}[/green]"
                        elif value >= 0.6:
                            return f"[yellow]{value:.4f}[/yellow]"
                        else:
                            return f"[red]{value:.4f}[/red]"

                    table.add_row(
                        method[:27] + "..." if len(method) > 30 else method,
                        dataset,
                        str(len(experiments)),
                        format_performance(best_acc),
                        format_performance(avg_acc),
                        f"{std_acc:.4f}",
                        format_performance(best_f1),
                    )

        console.print(table)

        # Show clustering analysis for PseudoLabelling methods
        console.print("\n[bold blue]🔍 PseudoLabelling Clustering Analysis[/bold blue]")

        clustering_table = Table(
            title="PseudoLabelling Performance by Cluster Count",
            show_header=True,
            header_style="bold purple",
        )
        clustering_table.add_column("Dataset", style="yellow", width=15)
        clustering_table.add_column(
            "Clusters", style="purple bold", justify="center", width=10
        )
        clustering_table.add_column(
            "Experiments", style="white", justify="center", width=12
        )
        clustering_table.add_column(
            "Best Acc", style="green", justify="center", width=12
        )
        clustering_table.add_column("Avg Acc", style="blue", justify="center", width=12)

        # Collect PseudoLabelling experiments
        clustering_data = defaultdict(lambda: defaultdict(list))

        for exp_name, exp_data in self.all_experiments.items():
            if (
                exp_data["type"] == "logs"
                and exp_data.get("target_performance")
                and exp_data.get("config", {}).get("method") == "PseudoLabelling"
            ):
                config = exp_data.get("config", {})
                dataset = config.get("dataset_short", "Unknown")
                clusters = config.get("clusters", "Unknown")

                perfs = exp_data["target_performance"]
                best_perf = max(perfs, key=lambda x: x["accuracy"])
                clustering_data[dataset][clusters].append(best_perf["accuracy"])

        for dataset, cluster_groups in clustering_data.items():
            for clusters, accuracies in cluster_groups.items():
                if accuracies:
                    best_acc = max(accuracies)
                    avg_acc = np.mean(accuracies)

                    clustering_table.add_row(
                        dataset,
                        str(clusters)
                        if clusters != "Unknown"
                        else "[dim]Unknown[/dim]",
                        str(len(accuracies)),
                        f"[bold green]{best_acc:.4f}[/bold green]"
                        if best_acc >= 0.9
                        else f"[green]{best_acc:.4f}[/green]"
                        if best_acc >= 0.8
                        else f"[yellow]{best_acc:.4f}[/yellow]"
                        if best_acc >= 0.6
                        else f"[red]{best_acc:.4f}[/red]",
                        f"{avg_acc:.4f}",
                    )

        console.print(clustering_table)

    def _show_training_progress(self):
        """Show training progress for selected experiment."""
        self._analyze_specific_experiment()  # Reuse the same logic

    def _show_loss_progression(self):
        """Show loss progression for selected experiment."""
        try:
            console.print(
                "\n[bold blue]Enter experiment name or part of it:[/bold blue]"
            )
            search_term = Prompt.ask("Search").lower()

            matches = [
                name
                for name in self.all_experiments.keys()
                if search_term in name.lower()
            ]

            if not matches:
                console.print("[red]No matching experiments found![/red]")
                return

            if len(matches) == 1:
                selected = matches[0]
            else:
                console.print("\n[bold blue]Multiple matches found:[/bold blue]")
                for i, match in enumerate(matches[:10], 1):  # Limit to 10
                    console.print(f"{i}. {match}")

                try:
                    choice = int(Prompt.ask("Select experiment number")) - 1
                    if 0 <= choice < len(matches):
                        selected = matches[choice]
                    else:
                        console.print("[red]Invalid choice![/red]")
                        return
                except ValueError:
                    console.print("[red]Please enter a valid number![/red]")
                    return
                except (KeyboardInterrupt, EOFError):
                    console.print("\n[yellow]Operation cancelled.[/yellow]")
                    return

            # Show loss progression chart
            console.print(
                f"\n[bold green]Loss Progression Analysis: {selected}[/bold green]"
            )
            loss_chart = self.create_loss_progression_chart(selected)
            console.print("\n")
            console.print(loss_chart)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Operation cancelled.[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Error showing loss progression: {e}[/red]")

    def _show_best_performers(self):
        """Show best performing experiments across all groups with detailed breakdown."""
        console.print("\n[bold green]🏆 Best Performing Experiments[/bold green]")

        # Collect all experiments with performance data
        performers = []
        for exp_name, exp_data in self.all_experiments.items():
            if exp_data["type"] == "logs" and exp_data.get("target_performance"):
                perfs = exp_data["target_performance"]
                if perfs:
                    best_perf = max(perfs, key=lambda x: x["accuracy"])
                    final_perf = perfs[-1] if perfs else best_perf
                    config = exp_data.get("config", {})
                    performers.append(
                        {
                            "name": exp_name,
                            "short_name": exp_name.split("::")[-1],
                            "accuracy": best_perf["accuracy"],
                            "f1_score": best_perf["f1_score"],
                            "precision": best_perf["precision"],
                            "recall": best_perf["recall"],
                            "final_loss": final_perf["loss"],
                            "epoch": best_perf["epoch"],
                            "dataset": config.get(
                                "dataset_short", config.get("dataset", "Unknown")
                            ),
                            "method": config.get(
                                "method_detailed", config.get("method", "Unknown")
                            ),
                            "full_method": config.get(
                                "full_method_description",
                                config.get("method", "Unknown"),
                            ),
                            "clusters": config.get("clusters"),
                            "group": exp_data.get("group", "Unknown"),
                        }
                    )

        # Sort by accuracy
        performers.sort(key=lambda x: x["accuracy"], reverse=True)

        # Show overall top performers
        table = Table(
            title="🥇 Overall Top Performers (by Accuracy)",
            show_header=True,
            header_style="bold gold1",
        )
        table.add_column("Rank", style="bold", width=5)
        table.add_column("Experiment", style="cyan", width=22)
        table.add_column("Dataset", style="yellow", width=12)
        table.add_column("Method", style="green", width=18)
        table.add_column("K", style="purple", justify="center", width=5)
        table.add_column("Acc", style="green", justify="center", width=8)
        table.add_column("F1", style="blue", justify="center", width=8)
        table.add_column("Prec", style="orange3", justify="center", width=8)
        table.add_column("Rec", style="yellow", justify="center", width=8)
        table.add_column("Loss", style="cyan", justify="center", width=8)

        for i, perf in enumerate(performers[:15], 1):  # Top 15
            rank_style = "bold gold1" if i <= 3 else "white"
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""

            clusters_str = str(perf["clusters"]) if perf["clusters"] else "N/A"
            clusters_display = (
                f"[bold purple]{clusters_str}[/bold purple]"
                if clusters_str != "N/A"
                else "[dim]-[/dim]"
            )

            # Color code loss (lower is better)
            def format_loss(loss_val: float) -> str:
                if loss_val <= 0.1:
                    return f"[bold green]{loss_val:.3f}[/bold green]"
                elif loss_val <= 0.3:
                    return f"[green]{loss_val:.3f}[/green]"
                elif loss_val <= 0.5:
                    return f"[yellow]{loss_val:.3f}[/yellow]"
                else:
                    return f"[red]{loss_val:.3f}[/red]"

            table.add_row(
                f"[{rank_style}]{i}{medal}[/{rank_style}]",
                perf["short_name"][:19] + "..."
                if len(perf["short_name"]) > 22
                else perf["short_name"],
                perf["dataset"].replace("NF-", "").replace("-v2", ""),
                perf["method"][:15] + "..."
                if len(perf["method"]) > 18
                else perf["method"],
                clusters_display,
                f"[bold green]{perf['accuracy']:.3f}[/bold green]",
                f"{perf['f1_score']:.3f}",
                f"{perf['precision']:.3f}",
                f"{perf['recall']:.3f}",
                format_loss(perf["final_loss"]),
            )

        console.print(table)

        # Show dataset-specific champions
        console.print(f"\n[bold blue]📊 Dataset Champions[/bold blue]")

        dataset_champions = {}
        for perf in performers:
            dataset = perf["dataset"]
            if (
                dataset not in dataset_champions
                or perf["accuracy"] > dataset_champions[dataset]["accuracy"]
            ):
                dataset_champions[dataset] = perf

        champions_table = Table(
            title="Best Performer per Dataset",
            show_header=True,
            header_style="bold cyan",
        )
        champions_table.add_column("Dataset", style="yellow bold", width=15)
        champions_table.add_column("Champion Method", style="green", width=25)
        champions_table.add_column("K", style="purple", justify="center", width=5)
        champions_table.add_column(
            "Accuracy", style="green bold", justify="center", width=10
        )
        champions_table.add_column("F1", style="blue", justify="center", width=8)
        champions_table.add_column(
            "Precision", style="orange3", justify="center", width=10
        )
        champions_table.add_column("Recall", style="yellow", justify="center", width=8)
        champions_table.add_column("Loss", style="cyan", justify="center", width=8)

        for dataset, champion in sorted(dataset_champions.items()):
            clusters_display = (
                f"[bold purple]{champion['clusters']}[/bold purple]"
                if champion["clusters"]
                else "[dim]-[/dim]"
            )

            def format_loss(loss_val: float) -> str:
                if loss_val <= 0.1:
                    return f"[bold green]{loss_val:.3f}[/bold green]"
                elif loss_val <= 0.3:
                    return f"[green]{loss_val:.3f}[/green]"
                elif loss_val <= 0.5:
                    return f"[yellow]{loss_val:.3f}[/yellow]"
                else:
                    return f"[red]{loss_val:.3f}[/red]"

            champions_table.add_row(
                dataset,
                champion["full_method"][:22] + "..."
                if len(champion["full_method"]) > 25
                else champion["full_method"],
                clusters_display,
                f"[bold green]{champion['accuracy']:.4f}[/bold green]",
                f"{champion['f1_score']:.3f}",
                f"{champion['precision']:.3f}",
                f"{champion['recall']:.3f}",
                format_loss(champion["final_loss"]),
            )

        console.print(champions_table)

        # Show method performance summary
        console.print(f"\n[bold blue]🔬 Method Performance Summary[/bold blue]")

        method_stats = defaultdict(list)
        for perf in performers:
            method_key = perf["full_method"]
            method_stats[method_key].append(perf["accuracy"])

        method_summary_table = Table(
            title="Method Performance Statistics",
            show_header=True,
            header_style="bold green",
        )
        method_summary_table.add_column("Method", style="cyan", width=30)
        method_summary_table.add_column(
            "Experiments", style="white", justify="center", width=12
        )
        method_summary_table.add_column(
            "Best Acc", style="green", justify="center", width=10
        )
        method_summary_table.add_column(
            "Avg Acc", style="blue", justify="center", width=10
        )
        method_summary_table.add_column(
            "Std Dev", style="purple", justify="center", width=10
        )

        for method, accuracies in sorted(
            method_stats.items(), key=lambda x: max(x[1]), reverse=True
        ):
            if len(accuracies) >= 1:  # Only show methods with at least 1 experiment
                best_acc = max(accuracies)
                avg_acc = np.mean(accuracies)
                std_acc = np.std(accuracies) if len(accuracies) > 1 else 0.0

                method_summary_table.add_row(
                    method[:27] + "..." if len(method) > 30 else method,
                    str(len(accuracies)),
                    f"[bold green]{best_acc:.4f}[/bold green]"
                    if best_acc >= 0.9
                    else f"[green]{best_acc:.4f}[/green]"
                    if best_acc >= 0.8
                    else f"[yellow]{best_acc:.4f}[/yellow]"
                    if best_acc >= 0.6
                    else f"[red]{best_acc:.4f}[/red]",
                    f"{avg_acc:.4f}",
                    f"{std_acc:.4f}",
                )

        console.print(method_summary_table)

    def _export_comparison_report(self):
        """Export comparison report to file."""
        filename = Prompt.ask(
            "Enter filename for report", default="experiment_comparison_report.txt"
        )

        with open(filename, "w") as f:
            f.write("MoMLNIDS Experiment Comparison Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for group_name, experiments in self.experiment_groups.items():
                f.write(f"\n{group_name}\n")
                f.write("-" * len(group_name) + "\n")

                for exp_name, exp_data in experiments.items():
                    f.write(f"\nExperiment: {exp_name}\n")

                    if exp_data["type"] == "logs":
                        config = exp_data.get("config", {})
                        f.write(f"  Dataset: {config.get('dataset', 'N/A')}\n")
                        f.write(f"  Method: {config.get('method', 'N/A')}\n")
                        f.write(f"  Clusters: {config.get('clusters', 'N/A')}\n")

                        if exp_data.get("target_performance"):
                            perfs = exp_data["target_performance"]
                            best_perf = max(perfs, key=lambda x: x["accuracy"])
                            f.write(
                                f"  Best Accuracy: {best_perf['accuracy']:.4f} (Epoch {best_perf['epoch']})\n"
                            )
                            f.write(f"  Best F1 Score: {best_perf['f1_score']:.4f}\n")
                            f.write(f"  Final Accuracy: {perfs[-1]['accuracy']:.4f}\n")

        console.print(f"[green]✅ Report exported to: {filename}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Enhanced MoMLNIDS Results Visualizer")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=".",
        help="Base directory containing experiment results (default: current directory)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive explorer (default mode)",
    )

    args = parser.parse_args()

    analyzer = EnhancedMoMLNIDSAnalyzer(args.base_dir)

    # Default to interactive mode
    analyzer.interactive_explorer()


if __name__ == "__main__":
    main()
