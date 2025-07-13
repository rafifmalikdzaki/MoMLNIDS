"""Experiment tracking utilities using Weights & Biases (wandb)."""

import os
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path
import json

import wandb
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from omegaconf import DictConfig
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich.tree import Tree
import time
import tempfile

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Handles experiment tracking with wandb integration."""

    def __init__(self, config: DictConfig):
        """
        Initialize experiment tracker.

        Args:
            config: Configuration object
        """
        self.config = config
        self.wandb_enabled = config.wandb.get("enabled", False)
        self.run = None

    def init_experiment(
        self,
        experiment_name: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
    ) -> None:
        """
        Initialize experiment tracking.

        Args:
            experiment_name: Name of the experiment
            tags: Additional tags for the experiment
            notes: Notes about the experiment
        """
        if not self.wandb_enabled:
            logger.info("Wandb tracking is disabled")
            return

        try:
            # Prepare wandb configuration
            wandb_config = {
                "project": self.config.wandb.get("project", "nids-research"),
                "entity": self.config.wandb.get("entity"),
                "name": experiment_name,
                "tags": (self.config.wandb.get("tags", []) + (tags or [])),
                "notes": notes,
                "config": self._prepare_config_for_wandb(),
            }

            # Remove None values
            wandb_config = {k: v for k, v in wandb_config.items() if v is not None}

            # Initialize wandb run
            self.run = wandb.init(**wandb_config)

            logger.info(f"Initialized wandb experiment: {self.run.name}")

        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            self.wandb_enabled = False

    def log_metrics(
        self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None
    ) -> None:
        """
        Log metrics to wandb.

        Args:
            metrics: Dictionary of metrics to log
            step: Step number (optional)
        """
        if not self.wandb_enabled or self.run is None:
            return

        try:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_model_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Log model performance metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            prefix: Prefix for metric names

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        try:
            # Compute basic metrics
            metrics[f"{prefix}accuracy"] = accuracy_score(y_true, y_pred)
            metrics[f"{prefix}precision"] = precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            metrics[f"{prefix}recall"] = recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            metrics[f"{prefix}f1_score"] = f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            )

            # Compute AUC if probabilities are provided
            if y_prob is not None:
                try:
                    if y_prob.shape[1] == 2:  # Binary classification
                        metrics[f"{prefix}auc_roc"] = roc_auc_score(
                            y_true, y_prob[:, 1]
                        )
                    else:  # Multi-class
                        metrics[f"{prefix}auc_roc"] = roc_auc_score(
                            y_true, y_prob, multi_class="ovr"
                        )
                except Exception as e:
                    logger.warning(f"Could not compute AUC: {e}")

            # Log metrics
            self.log_metrics(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Failed to compute/log performance metrics: {e}")
            return {}

    def log_model_artifact(
        self,
        model: torch.nn.Module,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log model as wandb artifact.

        Args:
            model: PyTorch model to save
            name: Name for the model artifact
            metadata: Additional metadata
        """
        if not self.wandb_enabled or self.run is None:
            return

        try:
            # Create artifact
            artifact = wandb.Artifact(name=name, type="model", metadata=metadata or {})

            # Save model temporarily
            temp_path = Path(f"temp_{name}.pth")
            torch.save(model.state_dict(), temp_path)

            # Add to artifact
            artifact.add_file(str(temp_path))

            # Log artifact
            self.run.log_artifact(artifact)

            # Clean up
            temp_path.unlink()

            logger.info(f"Logged model artifact: {name}")

        except Exception as e:
            logger.error(f"Failed to log model artifact: {e}")

    def log_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[list] = None
    ) -> None:
        """
        Log confusion matrix to wandb.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes (optional)
        """
        if not self.wandb_enabled or self.run is None:
            return

        try:
            wandb.log(
                {
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None, y_true=y_true, preds=y_pred, class_names=class_names
                    )
                }
            )

        except Exception as e:
            logger.error(f"Failed to log confusion matrix: {e}")

    def log_feature_importance(
        self,
        feature_names: list,
        importance_scores: np.ndarray,
        title: str = "Feature Importance",
    ) -> None:
        """
        Log feature importance plot.

        Args:
            feature_names: Names of features
            importance_scores: Importance scores for each feature
            title: Title for the plot
        """
        if not self.wandb_enabled or self.run is None:
            return

        try:
            # Create data for plotting
            data = [
                [name, score] for name, score in zip(feature_names, importance_scores)
            ]

            # Log as table
            table = wandb.Table(data=data, columns=["Feature", "Importance"])

            wandb.log(
                {
                    f"{title.lower().replace(' ', '_')}": wandb.plot.bar(
                        table, "Feature", "Importance", title=title
                    )
                }
            )

        except Exception as e:
            logger.error(f"Failed to log feature importance: {e}")

    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters.

        Args:
            hyperparams: Dictionary of hyperparameters
        """
        if not self.wandb_enabled or self.run is None:
            return

        try:
            wandb.config.update(hyperparams)
        except Exception as e:
            logger.error(f"Failed to log hyperparameters: {e}")

    def save_experiment_results(
        self, results: Dict[str, Any], filename: str = "experiment_results.json"
    ) -> None:
        """
        Save experiment results to file and log as artifact.

        Args:
            results: Results dictionary
            filename: Filename for results
        """
        try:
            # Create results directory
            results_dir = Path(self.config.output.get("results_dir", "results"))
            results_dir.mkdir(parents=True, exist_ok=True)

            # Save results locally
            results_path = results_dir / filename
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Saved experiment results to: {results_path}")

            # Log as wandb artifact if enabled
            if self.wandb_enabled and self.run is not None:
                artifact = wandb.Artifact("experiment_results", type="results")
                artifact.add_file(str(results_path))
                self.run.log_artifact(artifact)

        except Exception as e:
            logger.error(f"Failed to save experiment results: {e}")

    def finish_experiment(self) -> None:
        """Finish the current experiment."""
        if self.wandb_enabled and self.run is not None:
            try:
                wandb.finish()
                logger.info("Finished wandb experiment")
            except Exception as e:
                logger.error(f"Failed to finish wandb experiment: {e}")

    def _prepare_config_for_wandb(self) -> Dict[str, Any]:
        """Prepare configuration for wandb logging."""
        # Create a simplified config dict for wandb
        config_dict = {
            "model_name": self.config.model.get("name"),
            "batch_size": self.config.training.get("batch_size"),
            "learning_rate": self.config.training.get("learning_rate"),
            "epochs": self.config.training.get("epochs"),
            "datasets": self.config.data.get("datasets"),
            "random_seed": self.config.random_seed,
        }

        # Remove None values
        return {k: v for k, v in config_dict.items() if v is not None}


class MetricsLogger:
    """Simple metrics logger for tracking training progress."""

    def __init__(self):
        """Initialize metrics logger."""
        self.metrics_history = {}

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Step number (optional)
        """
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []

            if step is not None:
                self.metrics_history[name].append((step, value))
            else:
                self.metrics_history[name].append(value)

    def get_history(self, metric_name: str) -> list:
        """Get history for a specific metric."""
        return self.metrics_history.get(metric_name, [])

    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric."""
        history = self.get_history(metric_name)
        if history:
            return (
                history[-1] if isinstance(history[-1], (int, float)) else history[-1][1]
            )
        return None

    def clear(self) -> None:
        """Clear all metrics history."""
        self.metrics_history.clear()


def demo_experiment_tracking():
    """Demonstrate experiment tracking functionality."""
    console = Console()

    console.print(Panel.fit("📊 Experiment Tracking Demo", style="bold blue"))

    # Create sample configuration
    sample_config = DictConfig(
        {
            "wandb": {
                "enabled": False,  # Disable for demo
                "project": "momlnids-demo",
                "tags": ["demo", "test"],
            },
            "model": {"name": "MoMLNIDS"},
            "training": {"batch_size": 32, "learning_rate": 0.001, "epochs": 10},
            "data": {"datasets": ["dataset1", "dataset2"]},
            "random_seed": 42,
            "output": {"results_dir": "demo_results"},
        }
    )

    # Test ExperimentTracker
    console.print("🔬 Testing ExperimentTracker...")

    tracker = ExperimentTracker(sample_config)
    tracker.init_experiment("demo_experiment", tags=["demo"], notes="Demo experiment")

    # Test MetricsLogger
    console.print("\n📈 Testing MetricsLogger...")

    metrics_logger = MetricsLogger()

    # Simulate training metrics
    console.print("Simulating training progress...")
    training_table = Table(title="Training Progress")
    training_table.add_column("Epoch", style="cyan")
    training_table.add_column("Loss", style="red")
    training_table.add_column("Accuracy", style="green")
    training_table.add_column("F1 Score", style="yellow")

    for epoch in track(range(5), description="Training epochs..."):
        # Simulate metrics
        loss = 1.0 - (epoch * 0.15) + np.random.normal(0, 0.05)
        accuracy = 0.5 + (epoch * 0.1) + np.random.normal(0, 0.02)
        f1 = 0.4 + (epoch * 0.12) + np.random.normal(0, 0.03)

        metrics = {"train_loss": loss, "train_accuracy": accuracy, "train_f1": f1}

        # Log to both trackers
        tracker.log_metrics(metrics, step=epoch)
        metrics_logger.log(metrics, step=epoch)

        training_table.add_row(
            str(epoch), f"{loss:.4f}", f"{accuracy:.4f}", f"{f1:.4f}"
        )

        time.sleep(0.1)  # Simulate training time

    console.print(training_table)

    # Test model performance logging
    console.print("\n🎯 Testing model performance logging...")

    # Generate sample predictions
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.dirichlet(np.ones(n_classes), n_samples)

    # Introduce some correlation for realistic metrics
    mask = np.random.random(n_samples) < 0.7  # 70% correct predictions
    y_pred[mask] = y_true[mask]

    performance_metrics = tracker.log_model_performance(
        y_true, y_pred, y_prob, prefix="test_"
    )

    perf_table = Table(title="Model Performance")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="green")

    for metric, value in performance_metrics.items():
        perf_table.add_row(metric, f"{value:.4f}")

    console.print(perf_table)

    # Test experiment results saving
    console.print("\n💾 Testing experiment results saving...")

    experiment_results = {
        "experiment_name": "demo_experiment",
        "final_metrics": performance_metrics,
        "training_history": metrics_logger.metrics_history,
        "config": dict(sample_config),
        "timestamp": str(time.time()),
    }

    tracker.save_experiment_results(experiment_results, "demo_experiment_results.json")

    # Display metrics history
    console.print("\n📊 Metrics History:")

    for metric_name in metrics_logger.metrics_history:
        latest_value = metrics_logger.get_latest(metric_name)
        console.print(f"  {metric_name}: {latest_value:.4f} (latest)")

    tracker.finish_experiment()
    console.print("\n✨ Experiment tracking demo completed!")


def test_metrics_logger():
    """Test the MetricsLogger functionality."""
    console = Console()

    console.print(Panel.fit("📈 MetricsLogger Testing", style="bold green"))

    logger = MetricsLogger()

    # Test logging
    test_table = Table(title="MetricsLogger Tests")
    test_table.add_column("Test", style="cyan")
    test_table.add_column("Status", style="green")
    test_table.add_column("Details", style="yellow")

    try:
        # Test basic logging
        logger.log({"loss": 0.5, "accuracy": 0.8})
        test_table.add_row("Basic Logging", "✅ Pass", "Logged metrics without step")

        # Test logging with steps
        for i in range(10):
            logger.log({"loss": 1.0 - i * 0.1}, step=i)
        test_table.add_row("Step Logging", "✅ Pass", "Logged 10 steps")

        # Test history retrieval
        loss_history = logger.get_history("loss")
        test_table.add_row(
            "History Retrieval", "✅ Pass", f"Retrieved {len(loss_history)} entries"
        )

        # Test latest value
        latest_loss = logger.get_latest("loss")
        test_table.add_row("Latest Value", "✅ Pass", f"Latest loss: {latest_loss}")

        # Test clear
        logger.clear()
        empty_history = logger.get_history("loss")
        test_table.add_row(
            "Clear History", "✅ Pass", f"History length: {len(empty_history)}"
        )

    except Exception as e:
        test_table.add_row("Error", "❌ Fail", str(e)[:50])

    console.print(test_table)


@click.command()
@click.option("--demo", is_flag=True, help="Run experiment tracking demonstration")
@click.option("--test-metrics", is_flag=True, help="Test MetricsLogger functionality")
@click.option(
    "--test-wandb", is_flag=True, help="Test wandb integration (requires wandb setup)"
)
@click.option(
    "--simulate-training", type=int, default=5, help="Simulate training for N epochs"
)
def main(demo, test_metrics, test_wandb, simulate_training):
    """
    Test and demonstrate experiment tracking functionality.

    This script provides comprehensive testing of experiment tracking,
    metrics logging, and integration with Weights & Biases.
    """
    console = Console()

    if demo:
        demo_experiment_tracking()
    elif test_metrics:
        test_metrics_logger()
    elif test_wandb:
        console.print(Panel.fit("🌐 Testing W&B Integration", style="bold blue"))

        # Create config with wandb enabled
        config = DictConfig(
            {
                "wandb": {
                    "enabled": True,
                    "project": "momlnids-test",
                    "tags": ["test"],
                },
                "model": {"name": "TestModel"},
                "training": {"epochs": simulate_training, "batch_size": 32},
                "random_seed": 42,
                "output": {"results_dir": "test_results"},
            }
        )

        try:
            tracker = ExperimentTracker(config)
            tracker.init_experiment("wandb_test")

            # Log some test metrics
            for epoch in range(simulate_training):
                metrics = {
                    "epoch": epoch,
                    "loss": 1.0 - epoch * 0.1,
                    "accuracy": 0.5 + epoch * 0.08,
                }
                tracker.log_metrics(metrics, step=epoch)

            tracker.finish_experiment()
            console.print("✅ W&B integration test completed")

        except Exception as e:
            console.print(f"❌ W&B test failed: {e}")
    else:
        console.print("Use --demo to run full demonstration")
        console.print("Use --test-metrics to test MetricsLogger")
        console.print("Use --test-wandb to test W&B integration")
        console.print("Use --help for more options")


