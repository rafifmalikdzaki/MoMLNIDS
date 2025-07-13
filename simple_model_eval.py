#!/usr/bin/env python3
"""
Simple MoMLNIDS Model Evaluation Script

A streamlined script to load any model and dataset, run evaluation, and display metrics.
Usage: python simple_model_eval.py --model-path <path> --dataset <name>
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import click
from pathlib import Path
from typing import Dict, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich.tree import Tree
import warnings
import polars as pl
import glob
import time
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

import sys

sys.path.append("src")

from skripsi_code.model.MoMLNIDS import momlnids


class SimpleModelEvaluator:
    """Simple model evaluator for MoMLNIDS."""

    def __init__(self, model_path: str, device: str = "auto"):
        self.console = Console()
        self.model_path = Path(model_path)

        # Auto-detect device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.console.print(f"🔧 Using device: {self.device}")

        # Load model
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        """Load the trained model."""
        self.console.print(f"📦 Loading model from: {self.model_path}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Extract model configuration from path or use defaults
            model_config = self._infer_model_config(checkpoint)

            # Create model
            model = (
                momlnids(
                    input_nodes=model_config["input_nodes"],
                    hidden_nodes=model_config["hidden_nodes"],
                    classifier_nodes=model_config["classifier_nodes"],
                    num_domains=model_config["num_domains"],
                    num_class=model_config["num_class"],
                    single_layer=model_config["single_layer"],
                )
                .double()
                .to(self.device)
            )

            # Load state dict
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model.eval()

            self.console.print("✅ Model loaded successfully")
            return model

        except Exception as e:
            self.console.print(f"❌ Error loading model: {e}")
            raise

    def _infer_model_config(self, checkpoint: Dict) -> Dict:
        """Infer model configuration from checkpoint or use defaults."""

        # Try to extract from checkpoint metadata
        if "config" in checkpoint:
            return checkpoint["config"]

        # Default configuration based on MoMLNIDS architecture
        config = {
            "input_nodes": 39,  # Standard for network features
            "hidden_nodes": [256, 128, 64],  # Standard hidden layers
            "classifier_nodes": [32, 16],  # Standard classifier layers
            "num_domains": 4,  # Typical domain count
            "num_class": 2,  # Binary classification
            "single_layer": False,
        }

        # Try to infer from model path
        path_str = str(self.model_path).lower()
        if "singlelayer" in path_str:
            config["single_layer"] = True

        return config

    def load_dataset(
        self,
        dataset_name: str,
        data_path: str = "src/data/parquet",
        num_samples: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load dataset from parquet files."""

        self.console.print(f"📂 Loading dataset: {dataset_name}")

        data_dir = Path(data_path) / dataset_name
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

        # Find parquet files
        parquet_files = list(data_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in: {data_dir}")

        self.console.print(f"📄 Found {len(parquet_files)} parquet files")

        # Load and combine data
        all_data = []
        for file_path in track(parquet_files, description="Loading files..."):
            try:
                df = pl.read_parquet(file_path)
                all_data.append(df)
            except Exception as e:
                self.console.print(f"⚠️  Warning: Could not load {file_path}: {e}")

        if not all_data:
            raise ValueError("No data could be loaded")

        # Combine all data
        combined_df = pl.concat(all_data)

        # Sample data if requested
        if num_samples and len(combined_df) > num_samples:
            combined_df = combined_df.sample(num_samples, shuffle=True)
            self.console.print(f"🔢 Sampled {num_samples:,} rows from dataset")
        else:
            self.console.print(f"📊 Using all {len(combined_df):,} rows")

        # Prepare features and labels
        features, labels = self._prepare_data(combined_df)

        return features, labels

    def _prepare_data(self, df: pl.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare features and labels from dataframe."""

        # Convert to pandas for easier processing
        df_pd = df.to_pandas()

        # Assume last column is label, rest are features
        if "Label" in df_pd.columns:
            label_col = "Label"
        elif "label" in df_pd.columns:
            label_col = "label"
        else:
            # Use last column
            label_col = df_pd.columns[-1]

        # Extract features and labels
        feature_cols = [col for col in df_pd.columns if col != label_col]

        X = df_pd[feature_cols].values.astype(np.float64)
        y = df_pd[label_col].values

        # Convert labels to binary if needed
        if y.dtype == "object" or len(np.unique(y)) > 2:
            # Convert to binary: 0 for normal, 1 for attack
            y_binary = (
                np.where(y == "BENIGN", 0, 1)
                if y.dtype == "object"
                else np.where(y == 0, 0, 1)
            )
            y = y_binary

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float64).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        self.console.print(
            f"✅ Prepared data: {X_tensor.shape[0]:,} samples, {X_tensor.shape[1]} features"
        )

        return X_tensor, y_tensor

    def evaluate_model(
        self, features: torch.Tensor, labels: torch.Tensor, batch_size: int = 1000
    ) -> Dict:
        """Evaluate model and compute metrics."""

        self.console.print("🔄 Running model evaluation...")

        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []

        # Process in batches
        num_batches = (len(features) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in track(range(num_batches), description="Evaluating..."):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(features))

                batch_features = features[start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx]

                # Forward pass
                outputs = self.model(batch_features)

                # Get class predictions and probabilities
                if isinstance(outputs, tuple):
                    class_outputs = outputs[0]  # First output is classifier
                else:
                    class_outputs = outputs

                probabilities = F.softmax(class_outputs, dim=1)
                predictions = torch.argmax(class_outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)

        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_prob)

        return metrics

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict:
        """Calculate comprehensive evaluation metrics."""

        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics["recall"] = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics["f1_score"] = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # Per-class metrics
        metrics["precision_per_class"] = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        metrics["recall_per_class"] = recall_score(
            y_true, y_pred, average=None, zero_division=0
        )
        metrics["f1_per_class"] = f1_score(
            y_true, y_pred, average=None, zero_division=0
        )

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

        # ROC AUC if probabilities available and binary classification
        if y_prob.shape[1] == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
            except:
                metrics["roc_auc"] = "N/A"
        else:
            metrics["roc_auc"] = "N/A"

        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        metrics["class_distribution"] = dict(zip(unique, counts))

        # Prediction distribution
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        metrics["prediction_distribution"] = dict(zip(unique_pred, counts_pred))

        return metrics

    def display_results(self, metrics: Dict, dataset_name: str):
        """Display evaluation results using rich formatting."""

        # Main results panel
        self.console.print(
            Panel.fit(f"📊 Evaluation Results: {dataset_name}", style="bold blue")
        )

        # Performance metrics table
        perf_table = Table(title="🎯 Performance Metrics")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")

        perf_table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
        perf_table.add_row("Precision (Weighted)", f"{metrics['precision']:.4f}")
        perf_table.add_row("Recall (Weighted)", f"{metrics['recall']:.4f}")
        perf_table.add_row("F1-Score (Weighted)", f"{metrics['f1_score']:.4f}")

        if metrics["roc_auc"] != "N/A":
            perf_table.add_row("ROC AUC", f"{metrics['roc_auc']:.4f}")
        else:
            perf_table.add_row("ROC AUC", "N/A")

        self.console.print(perf_table)

        # Per-class metrics table
        if (
            len(metrics["precision_per_class"]) <= 5
        ):  # Only show if not too many classes
            class_table = Table(title="📋 Per-Class Metrics")
            class_table.add_column("Class", style="cyan")
            class_table.add_column("Precision", style="yellow")
            class_table.add_column("Recall", style="magenta")
            class_table.add_column("F1-Score", style="green")

            for i in range(len(metrics["precision_per_class"])):
                class_name = "Normal" if i == 0 else f"Attack_{i}"
                class_table.add_row(
                    class_name,
                    f"{metrics['precision_per_class'][i]:.4f}",
                    f"{metrics['recall_per_class'][i]:.4f}",
                    f"{metrics['f1_per_class'][i]:.4f}",
                )

            self.console.print(class_table)

        # Confusion matrix
        cm = metrics["confusion_matrix"]
        cm_table = Table(title="🔄 Confusion Matrix")
        cm_table.add_column("", style="cyan")

        # Add columns for predicted classes
        for i in range(cm.shape[1]):
            pred_name = "Normal" if i == 0 else f"Attack_{i}"
            cm_table.add_column(f"Pred {pred_name}", style="yellow")

        # Add rows for true classes
        for i in range(cm.shape[0]):
            true_name = "Normal" if i == 0 else f"Attack_{i}"
            row = [f"True {true_name}"] + [str(cm[i, j]) for j in range(cm.shape[1])]
            cm_table.add_row(*row)

        self.console.print(cm_table)

        # Class distribution
        dist_table = Table(title="📊 Data Distribution")
        dist_table.add_column("Metric", style="cyan")
        dist_table.add_column("Class 0 (Normal)", style="green")
        dist_table.add_column("Class 1 (Attack)", style="red")

        true_dist = metrics["class_distribution"]
        pred_dist = metrics["prediction_distribution"]

        dist_table.add_row(
            "True Labels", str(true_dist.get(0, 0)), str(true_dist.get(1, 0))
        )
        dist_table.add_row(
            "Predictions", str(pred_dist.get(0, 0)), str(pred_dist.get(1, 0))
        )

        self.console.print(dist_table)


@click.command()
@click.option(
    "--model-path", "-m", required=True, help="Path to trained model (.pt file)"
)
@click.option(
    "--dataset", "-d", required=True, help="Dataset name (e.g., NF-CSE-CIC-IDS2018-v2)"
)
@click.option("--data-path", default="src/data/parquet", help="Path to data directory")
@click.option(
    "--num-samples", "-n", type=int, help="Number of samples to evaluate (default: all)"
)
@click.option("--batch-size", "-b", default=1000, help="Batch size for evaluation")
@click.option("--device", default="auto", help="Device to use (auto, cpu, cuda)")
def main(model_path, dataset, data_path, num_samples, batch_size, device):
    """
    Simple MoMLNIDS Model Evaluation

    Load any trained model and dataset, run evaluation, and display comprehensive metrics.

    Example:
        python simple_model_eval.py -m ProperTraining/NF-CSE-CIC-IDS2018-v2/model_best.pt -d NF-CSE-CIC-IDS2018-v2
    """
    console = Console()

    console.print(Panel.fit("🚀 Simple MoMLNIDS Model Evaluation", style="bold blue"))

    try:
        # Initialize evaluator
        start_time = time.time()
        evaluator = SimpleModelEvaluator(model_path, device)

        # Load dataset
        features, labels = evaluator.load_dataset(dataset, data_path, num_samples)

        # Run evaluation
        metrics = evaluator.evaluate_model(features, labels, batch_size)

        # Display results
        evaluator.display_results(metrics, dataset)

        # Summary
        elapsed_time = time.time() - start_time
        console.print(f"\n✅ Evaluation completed in {elapsed_time:.2f} seconds")

    except Exception as e:
        console.print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
