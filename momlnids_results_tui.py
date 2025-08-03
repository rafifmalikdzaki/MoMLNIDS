#!/usr/bin/env python3
"""
MoMLNIDS Results TUI - Advanced Terminal User Interface
A modern, interactive terminal interface for exploring MoMLNIDS experiment results using Textual.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
import json
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    DataTable,
    Tree,
    Static,
    Button,
    TabbedContent,
    TabPane,
    ProgressBar,
    Label,
    Input,
    SelectionList,
    Pretty,
    Collapsible,
    Sparkline,
    Digits,
)
from textual.screen import Screen, ModalScreen
from textual.binding import Binding
from textual.reactive import reactive
from textual.message import Message
from textual import events
from rich.text import Text
from rich.console import Console
from rich.table import Table as RichTable
from rich.panel import Panel
from rich.syntax import Syntax

# Import our enhanced analyzer
from enhanced_results_visualizer import EnhancedMoMLNIDSAnalyzer

console = Console()


class ExperimentDetailModal(ModalScreen[None]):
    """Modal screen for detailed experiment analysis."""

    def __init__(self, experiment_name: str, experiment_data: Dict[str, Any]):
        super().__init__()
        self.experiment_name = experiment_name
        self.experiment_data = experiment_data

    def compose(self) -> ComposeResult:
        """Create the modal layout."""
        with Container(id="modal-container"):
            yield Static(
                f"[bold blue]Detailed Analysis: {self.experiment_name}[/bold blue]",
                id="modal-title",
            )

            with TabbedContent():
                with TabPane("Configuration", id="config-tab"):
                    yield self.create_config_view()

                with TabPane("Performance", id="performance-tab"):
                    yield self.create_performance_view()

                with TabPane("Training Progress", id="training-tab"):
                    yield self.create_training_view()

                with TabPane("Raw Data", id="raw-tab"):
                    yield Pretty(self.experiment_data, expand=True)

            with Horizontal():
                yield Button("Close", variant="primary", id="close-modal")
                yield Button("Export", variant="default", id="export-data")

    def create_config_view(self) -> Container:
        """Create configuration display."""
        container = Container()

        if self.experiment_data.get("type") == "logs":
            config = self.experiment_data.get("config", {})
            config_text = "\n".join(
                [f"[cyan]{k}:[/cyan] {v}" for k, v in config.items()]
            )
            container.mount(Static(config_text or "No configuration data available"))
        else:
            config = self.experiment_data.get("data", {}).get("config", {})
            config_text = "\n".join(
                [f"[cyan]{k}:[/cyan] {v}" for k, v in config.items()]
            )
            container.mount(Static(config_text or "No configuration data available"))

        return container

    def create_performance_view(self) -> Container:
        """Create performance metrics display."""
        container = Container()

        if self.experiment_data.get("type") == "logs" and self.experiment_data.get(
            "target_performance"
        ):
            perfs = self.experiment_data["target_performance"]
            if perfs:
                # Create performance summary
                best_acc = max(perfs, key=lambda x: x["accuracy"])
                latest_perf = perfs[-1]

                summary = f"""[bold green]Best Performance (Epoch {best_acc["epoch"]}):[/bold green]
  Accuracy: {best_acc["accuracy"]:.4f}
  F1 Score: {best_acc["f1_score"]:.4f}
  Precision: {best_acc["precision"]:.4f}
  Recall: {best_acc["recall"]:.4f}

[bold yellow]Latest Performance (Epoch {latest_perf["epoch"]}):[/bold yellow]
  Accuracy: {latest_perf["accuracy"]:.4f}
  F1 Score: {latest_perf["f1_score"]:.4f}
  
[bold blue]Training Summary:[/bold blue]
  Total Epochs: {len(perfs)}
  Accuracy Range: {min(p["accuracy"] for p in perfs):.3f} - {max(p["accuracy"] for p in perfs):.3f}
"""
                container.mount(Static(summary))

                # Add sparkline for accuracy trend
                accuracies = [p["accuracy"] for p in perfs[-20:]]  # Last 20 epochs
                container.mount(Static("[bold]Accuracy Trend (Last 20 epochs):[/bold]"))
                container.mount(Sparkline(accuracies, summary_function=max))
        else:
            container.mount(Static("No performance data available"))

        return container

    def create_training_view(self) -> Container:
        """Create training progress visualization."""
        container = Container()

        if self.experiment_data.get("type") == "logs" and self.experiment_data.get(
            "target_performance"
        ):
            perfs = self.experiment_data["target_performance"]

            # Create data table for epoch details
            table = DataTable()
            table.add_columns(
                "Epoch", "Accuracy", "F1 Score", "Precision", "Recall", "Loss"
            )

            for perf in perfs[-10:]:  # Show last 10 epochs
                table.add_row(
                    str(perf["epoch"]),
                    f"{perf['accuracy']:.3f}",
                    f"{perf['f1_score']:.3f}",
                    f"{perf['precision']:.3f}",
                    f"{perf['recall']:.3f}",
                    f"{perf['loss']:.3f}",
                )

            container.mount(Static("[bold]Recent Training Progress:[/bold]"))
            container.mount(table)
        else:
            container.mount(Static("No training data available"))

        return container

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "close-modal":
            self.dismiss()
        elif event.button.id == "export-data":
            # Export functionality
            filename = f"export_{self.experiment_name.replace('::', '_')}.json"
            with open(filename, "w") as f:
                json.dump(self.experiment_data, f, indent=2, default=str)
            self.notify(f"Exported to {filename}")


class ComparisonScreen(Screen):
    """Screen for comparing multiple experiments."""

    def __init__(self, experiments: Dict[str, Any], group_name: str):
        super().__init__()
        self.experiments = experiments
        self.group_name = group_name

    def compose(self) -> ComposeResult:
        """Create the comparison screen layout."""
        yield Header()

        with Container():
            yield Static(
                f"[bold blue]Experiment Comparison: {self.group_name}[/bold blue]",
                id="comparison-title",
            )

            # Create comparison table
            table = DataTable()
            table.add_columns(
                "Experiment",
                "Dataset",
                "Method",
                "Best Acc",
                "Best F1",
                "Final Acc",
                "Epochs",
            )

            for exp_name, exp_data in list(self.experiments.items())[
                :20
            ]:  # Limit to 20 for display
                if exp_data.get("type") == "logs" and exp_data.get(
                    "target_performance"
                ):
                    config = exp_data.get("config", {})
                    perfs = exp_data["target_performance"]

                    if perfs:
                        best_acc = max(perfs, key=lambda x: x["accuracy"])
                        best_f1 = max(perfs, key=lambda x: x["f1_score"])
                        final_perf = perfs[-1]

                        table.add_row(
                            exp_name.split("::")[-1][:25],
                            config.get("dataset", "N/A")
                            .replace("NF-", "")
                            .replace("-v2", ""),
                            config.get("method", "N/A"),
                            f"{best_acc['accuracy']:.3f}",
                            f"{best_f1['f1_score']:.3f}",
                            f"{final_perf['accuracy']:.3f}",
                            str(len(perfs)),
                        )

            yield table

            with Horizontal():
                yield Button("Back", variant="primary", id="back-button")
                yield Button(
                    "Export Comparison", variant="default", id="export-comparison"
                )

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "back-button":
            self.app.pop_screen()
        elif event.button.id == "export-comparison":
            # Export comparison data
            filename = f"comparison_{self.group_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w") as f:
                json.dump(self.experiments, f, indent=2, default=str)
            self.notify(f"Comparison exported to {filename}")


class MoMLNIDSTUI(App):
    """Main TUI application for MoMLNIDS results visualization."""

    CSS = """
    #main-container {
        layout: grid;
        grid-size: 2 1;
        grid-columns: 1fr 2fr;
    }
    
    #left-panel {
        border: solid $primary;
        margin: 1;
    }
    
    #right-panel {
        border: solid $secondary;
        margin: 1;
    }
    
    #modal-container {
        align: center middle;
        background: $surface;
        border: thick $primary;
        width: 80%;
        height: 80%;
    }
    
    #modal-title {
        text-align: center;
        margin: 1;
    }
    
    #comparison-title {
        text-align: center;
        margin: 1;
    }
    
    .status-panel {
        height: 4;
        border: solid $accent;
        margin: 1;
    }
    
    #experiment-tree {
        height: 100%;
    }
    
    #details-panel {
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("c", "compare", "Compare"),
        Binding("e", "export", "Export"),
        Binding("h", "help", "Help"),
    ]

    def __init__(self):
        super().__init__()
        self.analyzer = EnhancedMoMLNIDSAnalyzer(".")
        self.experiments = {}
        self.selected_experiment = None

    def compose(self) -> ComposeResult:
        """Create the main application layout."""
        yield Header()

        with Container(id="main-container"):
            # Left panel - Experiment tree and controls
            with Vertical(id="left-panel"):
                yield Static(
                    "[bold blue]📊 MoMLNIDS Experiments[/bold blue]", classes="title"
                )

                # Status panel
                with Container(classes="status-panel"):
                    yield Static("Loading experiments...", id="status-text")
                    yield ProgressBar(id="progress-bar")

                # Control buttons
                with Horizontal():
                    yield Button("🔄 Refresh", id="refresh-btn", variant="primary")
                    yield Button("📊 Compare", id="compare-btn", variant="default")
                    yield Button("📈 Best", id="best-btn", variant="success")

                # Experiment tree
                yield Tree("Experiments", id="experiment-tree")

            # Right panel - Details and analysis
            with Vertical(id="right-panel"):
                with TabbedContent():
                    with TabPane("Overview", id="overview-tab"):
                        yield Static(
                            "[bold]Select an experiment to view details[/bold]",
                            id="overview-content",
                        )

                    with TabPane("Performance", id="perf-tab"):
                        yield Static("No experiment selected", id="perf-content")

                    with TabPane("Configuration", id="config-tab"):
                        yield Static("No experiment selected", id="config-content")

                    with TabPane("Analysis", id="analysis-tab"):
                        yield Static("No experiment selected", id="analysis-content")

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the application."""
        await self.load_experiments()

    async def load_experiments(self) -> None:
        """Load and discover all experiments."""
        status_text = self.query_one("#status-text", Static)
        progress_bar = self.query_one("#progress-bar", ProgressBar)

        status_text.update("🔍 Discovering experiments...")
        progress_bar.progress = 0

        # Load experiments
        self.experiments = self.analyzer.discover_experiments()

        progress_bar.progress = 50
        status_text.update("🌳 Building experiment tree...")

        # Populate the tree
        await self.populate_experiment_tree()

        progress_bar.progress = 100
        total_experiments = sum(len(exps) for exps in self.experiments.values())
        status_text.update(
            f"✅ Loaded {total_experiments} experiments from {len(self.experiments)} directories"
        )

    async def populate_experiment_tree(self) -> None:
        """Populate the experiment tree widget."""
        tree = self.query_one("#experiment-tree", Tree)
        tree.clear()

        # Root node
        root = tree.root
        root.expand()

        for group_name, experiments in self.experiments.items():
            if not experiments:  # Skip empty groups
                continue

            group_node = root.add(f"📁 {group_name} ({len(experiments)})", expand=False)
            group_node.data = {
                "type": "group",
                "name": group_name,
                "experiments": experiments,
            }

            # Group by dataset
            dataset_groups = {}
            for exp_name, exp_data in experiments.items():
                if exp_data.get("type") == "logs" and exp_data.get("config"):
                    dataset = exp_data["config"].get("dataset", "Unknown")
                else:
                    dataset = "JSON Results"

                if dataset not in dataset_groups:
                    dataset_groups[dataset] = []
                dataset_groups[dataset].append((exp_name, exp_data))

            for dataset, exp_list in dataset_groups.items():
                dataset_name = dataset.replace("NF-", "").replace("-v2", "")
                dataset_node = group_node.add(
                    f"🗂️ {dataset_name} ({len(exp_list)})", expand=False
                )
                dataset_node.data = {
                    "type": "dataset",
                    "name": dataset,
                    "experiments": dict(exp_list),
                }

                for exp_name, exp_data in exp_list:
                    # Create experiment node with performance indicator
                    if exp_data.get("type") == "logs" and exp_data.get(
                        "target_performance"
                    ):
                        perfs = exp_data["target_performance"]
                        if perfs:
                            best_acc = max(perfs, key=lambda x: x["accuracy"])[
                                "accuracy"
                            ]
                            icon = (
                                "🟢"
                                if best_acc >= 0.8
                                else "🟡"
                                if best_acc >= 0.6
                                else "🔴"
                            )
                            exp_display = f"{icon} {exp_name} (Acc: {best_acc:.3f})"
                        else:
                            exp_display = f"⚪ {exp_name}"
                    else:
                        exp_display = f"📄 {exp_name}"

                    exp_node = dataset_node.add(exp_display, expand=False)
                    exp_node.data = {
                        "type": "experiment",
                        "name": f"{group_name}::{exp_name}",
                        "data": exp_data,
                    }

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection."""
        node_data = event.node.data

        if node_data and node_data.get("type") == "experiment":
            self.selected_experiment = node_data["name"]
            self.update_experiment_details(node_data["data"])
        elif node_data and node_data.get("type") == "group":
            self.update_group_overview(node_data["name"], node_data["experiments"])

    def update_experiment_details(self, exp_data: Dict[str, Any]) -> None:
        """Update the right panel with experiment details."""
        # Update overview tab
        overview_content = self.query_one("#overview-content", Static)

        if exp_data.get("type") == "logs":
            config = exp_data.get("config", {})
            overview_text = f"""[bold blue]Experiment Details[/bold blue]

[cyan]Dataset:[/cyan] {config.get("dataset", "N/A")}
[cyan]Method:[/cyan] {config.get("method", "N/A")}
[cyan]Clusters:[/cyan] {config.get("clusters", "N/A")}
[cyan]Architecture:[/cyan] {config.get("architecture", "N/A")}
[cyan]Weighting:[/cyan] {config.get("weighting", "N/A")}

[yellow]Click for detailed analysis[/yellow]
"""
        else:
            overview_text = (
                "[bold blue]JSON Experiment[/bold blue]\n\nClick for detailed analysis"
            )

        overview_content.update(overview_text)

        # Update performance tab
        perf_content = self.query_one("#perf-content", Static)

        if exp_data.get("type") == "logs" and exp_data.get("target_performance"):
            perfs = exp_data["target_performance"]
            if perfs:
                best_perf = max(perfs, key=lambda x: x["accuracy"])
                latest_perf = perfs[-1]

                perf_text = f"""[bold green]Best Performance (Epoch {best_perf["epoch"]}):[/bold green]
  Accuracy: {best_perf["accuracy"]:.4f}
  F1 Score: {best_perf["f1_score"]:.4f}
  Precision: {best_perf["precision"]:.4f}
  Recall: {best_perf["recall"]:.4f}

[bold yellow]Latest Performance:[/bold yellow]
  Accuracy: {latest_perf["accuracy"]:.4f}
  F1 Score: {latest_perf["f1_score"]:.4f}

[bold blue]Progress:[/bold blue]
  Total Epochs: {len(perfs)}
  Best at Epoch: {best_perf["epoch"]}
"""
            else:
                perf_text = "No performance data available"
        else:
            perf_text = "No performance data available"

        perf_content.update(perf_text)

    def update_group_overview(
        self, group_name: str, experiments: Dict[str, Any]
    ) -> None:
        """Update the panel with group overview."""
        overview_content = self.query_one("#overview-content", Static)

        # Calculate group statistics
        total_experiments = len(experiments)
        log_experiments = sum(
            1 for exp in experiments.values() if exp.get("type") == "logs"
        )

        # Find best performer
        best_experiment = None
        best_accuracy = 0

        for exp_name, exp_data in experiments.items():
            if exp_data.get("type") == "logs" and exp_data.get("target_performance"):
                perfs = exp_data["target_performance"]
                if perfs:
                    best_acc = max(perfs, key=lambda x: x["accuracy"])["accuracy"]
                    if best_acc > best_accuracy:
                        best_accuracy = best_acc
                        best_experiment = exp_name

        overview_text = f"""[bold blue]{group_name} Overview[/bold blue]

[cyan]Total Experiments:[/cyan] {total_experiments}
[cyan]Log-based Experiments:[/cyan] {log_experiments}

[bold green]Best Performer:[/bold green]
  {best_experiment or "N/A"}
  Accuracy: {best_accuracy:.3f}

[yellow]Use Compare button to see detailed comparison[/yellow]
"""

        overview_content.update(overview_text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "refresh-btn":
            self.action_refresh()
        elif event.button.id == "compare-btn":
            self.action_compare()
        elif event.button.id == "best-btn":
            self.show_best_performers()

    def on_key(self, event: events.Key) -> None:
        """Handle key presses."""
        if event.key == "enter" and self.selected_experiment:
            # Show detailed modal for selected experiment
            exp_data = self.analyzer.all_experiments.get(self.selected_experiment)
            if exp_data:
                self.push_screen(
                    ExperimentDetailModal(self.selected_experiment, exp_data)
                )

    def action_refresh(self) -> None:
        """Refresh experiments."""
        self.run_worker(self.load_experiments())

    def action_compare(self) -> None:
        """Show comparison screen."""
        tree = self.query_one("#experiment-tree", Tree)
        selected_node = tree.cursor_node

        if selected_node and selected_node.data:
            node_data = selected_node.data
            if node_data.get("type") == "group":
                self.push_screen(
                    ComparisonScreen(node_data["experiments"], node_data["name"])
                )
            elif node_data.get("type") == "dataset":
                self.push_screen(
                    ComparisonScreen(node_data["experiments"], node_data["name"])
                )
            else:
                self.notify("Select a group or dataset to compare")
        else:
            self.notify("Select a group or dataset to compare")

    def show_best_performers(self) -> None:
        """Show best performing experiments."""
        # Collect all experiments with performance data
        performers = []
        for exp_name, exp_data in self.analyzer.all_experiments.items():
            if exp_data.get("type") == "logs" and exp_data.get("target_performance"):
                perfs = exp_data["target_performance"]
                if perfs:
                    best_perf = max(perfs, key=lambda x: x["accuracy"])
                    performers.append(
                        {
                            "name": exp_name,
                            "accuracy": best_perf["accuracy"],
                            "f1_score": best_perf["f1_score"],
                            "epoch": best_perf["epoch"],
                            "dataset": exp_data.get("config", {}).get(
                                "dataset", "Unknown"
                            ),
                        }
                    )

        # Sort by accuracy
        performers.sort(key=lambda x: x["accuracy"], reverse=True)

        # Create display text
        best_text = "[bold gold1]🏆 Top 10 Best Performers[/bold gold1]\n\n"

        for i, perf in enumerate(performers[:10], 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            dataset_name = perf["dataset"].replace("NF-", "").replace("-v2", "")
            best_text += f"{medal} [cyan]{perf['name'].split('::')[-1][:30]}[/cyan]\n"
            best_text += f"    Dataset: {dataset_name}\n"
            best_text += f"    Accuracy: [bold green]{perf['accuracy']:.4f}[/bold green] (Epoch {perf['epoch']})\n"
            best_text += (
                f"    F1 Score: [bold blue]{perf['f1_score']:.4f}[/bold blue]\n\n"
            )

        # Update overview with best performers
        overview_content = self.query_one("#overview-content", Static)
        overview_content.update(best_text)

    def action_export(self) -> None:
        """Export all results."""
        filename = (
            f"momlnids_results_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(filename, "w") as f:
            json.dump(self.analyzer.experiment_groups, f, indent=2, default=str)
        self.notify(f"Results exported to {filename}")

    def action_help(self) -> None:
        """Show help information."""
        help_text = """[bold blue]MoMLNIDS Results TUI - Help[/bold blue]

[bold yellow]Navigation:[/bold yellow]
• Use arrow keys to navigate the experiment tree
• Press Enter to view detailed analysis
• Select groups or datasets and press 'c' to compare

[bold yellow]Key Bindings:[/bold yellow]
• q - Quit application
• r - Refresh experiments
• c - Compare selected group/dataset
• e - Export all results
• h - Show this help

[bold yellow]Tree Icons:[/bold yellow]
• 🟢 High performance (≥80% accuracy)
• 🟡 Medium performance (60-79% accuracy)
• 🔴 Low performance (<60% accuracy)
• 📁 Experiment group
• 🗂️ Dataset group
• 📄 JSON results

[bold yellow]Buttons:[/bold yellow]
• Refresh - Reload all experiments
• Compare - Compare experiments in selected group
• Best - Show top performing experiments

[bold yellow]Tabs:[/bold yellow]
• Overview - General information
• Performance - Metrics and statistics
• Configuration - Experiment settings
• Analysis - Detailed analysis
"""

        overview_content = self.query_one("#overview-content", Static)
        overview_content.update(help_text)


def main():
    """Run the MoMLNIDS TUI application."""
    app = MoMLNIDSTUI()
    app.run()


if __name__ == "__main__":
    main()
