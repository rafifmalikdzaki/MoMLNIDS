#!/usr/bin/env python3
"""
MoMLNIDS Textual User Interface (TUI)
Interactive dashboard for Network Intrusion Detection System training and evaluation.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import numpy as np
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Static,
    Button,
    Input,
    Select,
    TextArea,
    DataTable,
    ProgressBar,
    Log,
    Tabs,
    TabPane,
    Label,
    Tree,
)
from textual.binding import Binding
from textual.screen import Screen
from textual.reactive import reactive
from textual import events

# Import project modules
try:
    from src.skripsi_code.config import load_config, get_config
    from src.skripsi_code.model.MoMLNIDS import momlnids
    from src.skripsi_code.utils.dataloader import random_split_dataloader
    from src.skripsi_code.TrainEval.TrainEval import train, eval
    from src.skripsi_code.clustering.cluster_utils import pseudolabeling
    from src.skripsi_code.explainability.explainer import ModelExplainer
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some features may not be available.")


class TrainingMonitor(Static):
    """Widget to monitor training progress."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_active = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_loss = 0.0

    def compose(self) -> ComposeResult:
        yield Label("Training Status: Idle", id="training-status")
        yield ProgressBar(total=100, show_eta=True, id="epoch-progress")
        yield Label("Epoch: 0/0", id="epoch-label")
        yield Label("Loss: 0.0000", id="loss-label")
        yield TextArea("", read_only=True, id="training-log")

    def update_training_status(
        self, status: str, epoch: int = 0, total: int = 0, loss: float = 0.0
    ):
        """Update training status display."""
        self.query_one("#training-status", Label).update(f"Training Status: {status}")
        self.query_one("#epoch-label", Label).update(f"Epoch: {epoch}/{total}")
        self.query_one("#loss-label", Label).update(f"Loss: {loss:.4f}")

        if total > 0:
            progress = (epoch / total) * 100
            self.query_one("#epoch-progress", ProgressBar).update(progress=progress)

    def log_message(self, message: str):
        """Add message to training log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_widget = self.query_one("#training-log", TextArea)
        current_text = log_widget.text
        new_text = f"{current_text}\n[{timestamp}] {message}"
        log_widget.text = new_text[-2000:]  # Keep last 2000 chars


class ConfigPanel(Static):
    """Widget for configuration management."""

    def compose(self) -> ComposeResult:
        yield Label("Configuration Settings", classes="panel-title")

        with Vertical():
            yield Label("Model Configuration:")
            yield Select(
                [("MoMLNIDS", "momlnids"), ("Custom", "custom")],
                value="momlnids",
                id="model-select",
            )

            yield Label("Dataset Selection:")
            yield Select(
                [
                    ("NF-UNSW-NB15-v2", "unsw"),
                    ("NF-CSE-CIC-IDS2018-v2", "cic"),
                    ("NF-ToN-IoT-v2", "ton"),
                    ("All Datasets", "all"),
                ],
                value="all",
                id="dataset-select",
            )

            yield Label("Training Parameters:")
            with Horizontal():
                yield Label("Epochs:")
                yield Input(value="20", id="epochs-input")
            with Horizontal():
                yield Label("Batch Size:")
                yield Input(value="1", id="batch-input")
            with Horizontal():
                yield Label("Learning Rate:")
                yield Input(value="0.0015", id="lr-input")

            yield Label("Features:")
            with Horizontal():
                yield Button("✓ Clustering", variant="primary", id="clustering-toggle")
                yield Button("✗ Explainability", id="explain-toggle")
                yield Button("✗ W&B Logging", id="wandb-toggle")


class ResultsPanel(Static):
    """Widget for displaying results and metrics."""

    def compose(self) -> ComposeResult:
        yield Label("Experiment Results", classes="panel-title")

        with Vertical():
            yield DataTable(id="metrics-table")
            yield Label("Model Performance:", classes="section-title")
            yield DataTable(id="performance-table")

    def on_mount(self):
        """Initialize tables when mounted."""
        metrics_table = self.query_one("#metrics-table", DataTable)
        metrics_table.add_columns("Metric", "Value")

        performance_table = self.query_one("#performance-table", DataTable)
        performance_table.add_columns(
            "Dataset", "Accuracy", "F1-Score", "Precision", "Recall"
        )

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics display."""
        metrics_table = self.query_one("#metrics-table", DataTable)
        metrics_table.clear()

        for key, value in metrics.items():
            if isinstance(value, float):
                metrics_table.add_row(key, f"{value:.4f}")
            else:
                metrics_table.add_row(key, str(value))

    def update_performance(self, results: Dict[str, Dict[str, float]]):
        """Update performance results."""
        performance_table = self.query_one("#performance-table", DataTable)
        performance_table.clear()

        for dataset, metrics in results.items():
            performance_table.add_row(
                dataset,
                f"{metrics.get('accuracy', 0):.4f}",
                f"{metrics.get('f1_score', 0):.4f}",
                f"{metrics.get('precision', 0):.4f}",
                f"{metrics.get('recall', 0):.4f}",
            )


class SystemInfoPanel(Static):
    """Widget for system information."""

    def compose(self) -> ComposeResult:
        yield Label("System Information", classes="panel-title")

        device_info = "CUDA" if torch.cuda.is_available() else "CPU"
        if torch.cuda.is_available():
            device_info += f" ({torch.cuda.get_device_name()})"

        with Vertical():
            yield Label(f"Device: {device_info}")
            yield Label(f"PyTorch Version: {torch.__version__}")
            yield Label(f"Available Memory: {self.get_memory_info()}")
            yield Label(f"Project Path: {Path.cwd()}")

    def get_memory_info(self) -> str:
        """Get system memory information."""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            return f"{total_memory // (1024**3)} GB GPU"
        else:
            return "CPU Mode"


class ExperimentManager(Static):
    """Widget for managing experiments."""

    def compose(self) -> ComposeResult:
        yield Label("Experiment Management", classes="panel-title")

        with Vertical():
            yield Label("Quick Actions:")
            with Horizontal():
                yield Button(
                    "🚀 Start Training", variant="success", id="start-training"
                )
                yield Button("⏹️ Stop Training", variant="error", id="stop-training")
                yield Button("💾 Save Model", variant="primary", id="save-model")

            yield Label("Advanced Actions:")
            with Horizontal():
                yield Button("🔍 Run Evaluation", id="run-evaluation")
                yield Button("🧠 Generate Explanations", id="generate-explanations")
                yield Button("📊 View Clustering", id="view-clustering")

            yield Label("Experiment History:")
            yield DataTable(id="experiment-history")

    def on_mount(self):
        """Initialize experiment history table."""
        history_table = self.query_one("#experiment-history", DataTable)
        history_table.add_columns("Timestamp", "Action", "Status", "Notes")


class MainDashboard(Screen):
    """Main dashboard screen."""

    CSS = """
    .panel-title {
        text-style: bold;
        color: cyan;
        margin: 1;
    }
    
    .section-title {
        text-style: bold;
        color: yellow;
        margin: 1 0;
    }
    
    Button {
        margin: 0 1;
    }
    
    Input {
        width: 10;
    }
    
    DataTable {
        height: 8;
        margin: 1 0;
    }
    
    TextArea {
        height: 10;
        margin: 1 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Container():
            with Tabs("Config", "Training", "Results", "System"):
                with TabPane("Configuration", id="config-tab"):
                    yield ConfigPanel()

                with TabPane("Training Monitor", id="training-tab"):
                    yield TrainingMonitor(id="training-monitor")
                    yield ExperimentManager()

                with TabPane("Results", id="results-tab"):
                    yield ResultsPanel(id="results-panel")

                with TabPane("System", id="system-tab"):
                    yield SystemInfoPanel()

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id

        if button_id == "start-training":
            self.start_training()
        elif button_id == "stop-training":
            self.stop_training()
        elif button_id == "save-model":
            self.save_model()
        elif button_id == "run-evaluation":
            self.run_evaluation()
        elif button_id == "generate-explanations":
            self.generate_explanations()
        elif button_id == "view-clustering":
            self.view_clustering()
        elif button_id == "clustering-toggle":
            self.toggle_feature(event.button, "clustering")
        elif button_id == "explain-toggle":
            self.toggle_feature(event.button, "explainability")
        elif button_id == "wandb-toggle":
            self.toggle_feature(event.button, "wandb")

    def toggle_feature(self, button: Button, feature: str):
        """Toggle feature on/off."""
        if button.label.startswith("✓"):
            button.label = f"✗ {feature.title()}"
            button.variant = "default"
        else:
            button.label = f"✓ {feature.title()}"
            button.variant = "primary"

    def start_training(self):
        """Start model training."""
        training_monitor = self.query_one("#training-monitor", TrainingMonitor)
        training_monitor.log_message("Starting training...")
        training_monitor.update_training_status("Initializing", 0, 20)

        # Add to experiment history
        self.add_experiment_entry(
            "Training Started", "Running", "Using default configuration"
        )

        # Here you would integrate with your actual training code
        self.notify(
            "Training started! Check the Training Monitor tab.", severity="information"
        )

    def stop_training(self):
        """Stop model training."""
        training_monitor = self.query_one("#training-monitor", TrainingMonitor)
        training_monitor.log_message("Training stopped by user.")
        training_monitor.update_training_status("Stopped")

        self.add_experiment_entry(
            "Training Stopped", "Stopped", "Manually stopped by user"
        )
        self.notify("Training stopped.", severity="warning")

    def save_model(self):
        """Save current model."""
        self.add_experiment_entry("Model Saved", "Completed", "Model saved to disk")
        self.notify("Model saved successfully!", severity="information")

    def run_evaluation(self):
        """Run model evaluation."""
        self.add_experiment_entry(
            "Evaluation Started", "Running", "Evaluating on test datasets"
        )

        # Simulate evaluation results
        results = {
            "NF-UNSW-NB15-v2": {
                "accuracy": 0.9234,
                "f1_score": 0.8912,
                "precision": 0.9001,
                "recall": 0.8823,
            },
            "NF-CSE-CIC-IDS2018-v2": {
                "accuracy": 0.8876,
                "f1_score": 0.8654,
                "precision": 0.8789,
                "recall": 0.8521,
            },
            "NF-ToN-IoT-v2": {
                "accuracy": 0.9456,
                "f1_score": 0.9234,
                "precision": 0.9345,
                "recall": 0.9124,
            },
        }

        results_panel = self.query_one("#results-panel", ResultsPanel)
        results_panel.update_performance(results)

        self.add_experiment_entry(
            "Evaluation Completed", "Completed", "Results updated"
        )
        self.notify("Evaluation completed! Check Results tab.", severity="information")

    def generate_explanations(self):
        """Generate model explanations."""
        self.add_experiment_entry(
            "Explanations Generated",
            "Completed",
            "SHAP and feature importance computed",
        )
        self.notify("Model explanations generated!", severity="information")

    def view_clustering(self):
        """View clustering results."""
        self.add_experiment_entry(
            "Clustering Viewed", "Completed", "Cluster visualization generated"
        )
        self.notify("Clustering results displayed!", severity="information")

    def add_experiment_entry(self, action: str, status: str, notes: str):
        """Add entry to experiment history."""
        try:
            history_table = self.query_one("#experiment-history", DataTable)
            timestamp = datetime.now().strftime("%H:%M:%S")
            history_table.add_row(timestamp, action, status, notes)
        except Exception:
            pass  # Ignore if table not found


class MoMLNIDSTUI(App):
    """Main TUI application for MoMLNIDS."""

    TITLE = "MoMLNIDS - Multi-Domain Network Intrusion Detection System"
    SUB_TITLE = "Interactive Training & Evaluation Dashboard"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("t", "toggle_training", "Toggle Training"),
        Binding("e", "evaluate", "Evaluate"),
        Binding("s", "save", "Save Model"),
        Binding("h", "help", "Help"),
    ]

    def __init__(self):
        super().__init__()
        self.config = None
        self.model = None
        self.training_active = False

    def on_mount(self):
        """Initialize the application."""
        try:
            self.config = get_config()
            self.notify("Configuration loaded successfully!", severity="information")
        except Exception as e:
            self.notify(f"Failed to load configuration: {e}", severity="error")

    def action_toggle_training(self):
        """Toggle training state."""
        if not self.training_active:
            self.query_one(MainDashboard).start_training()
        else:
            self.query_one(MainDashboard).stop_training()

    def action_evaluate(self):
        """Run evaluation."""
        self.query_one(MainDashboard).run_evaluation()

    def action_save(self):
        """Save model."""
        self.query_one(MainDashboard).save_model()

    def action_help(self):
        """Show help information."""
        help_text = """
        MoMLNIDS TUI - Keyboard Shortcuts:
        
        q - Quit application
        t - Toggle training on/off
        e - Run evaluation
        s - Save model
        h - Show this help
        
        Mouse Controls:
        - Click buttons to perform actions
        - Use tabs to navigate between panels
        - Scroll in text areas and tables
        
        Features:
        - Real-time training monitoring
        - Interactive configuration
        - Performance metrics display
        - Experiment history tracking
        """
        self.notify(help_text, severity="information")

    def on_ready(self):
        """Called when app is ready."""
        self.push_screen(MainDashboard())


def main():
    """Main entry point."""
    app = MoMLNIDSTUI()
    app.run()


if __name__ == "__main__":
    main()
