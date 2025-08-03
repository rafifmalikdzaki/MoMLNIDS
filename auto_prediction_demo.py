#!/usr/bin/env python3
"""
MoMLNIDS Automatic Prediction Demo with Real Data

This script automatically loads real network intrusion detection data from the
parquet files and demonstrates the prediction capabilities of trained MoMLNIDS models.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import click
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, track
from rich.tree import Tree
from rich.columns import Columns
import warnings
import polars as pl
import glob
import random
from collections import defaultdict

warnings.filterwarnings("ignore")

import sys

sys.path.append("src")

from skripsi_code.model.MoMLNIDS import momlnids
from skripsi_code.utils.dataloader import random_split_dataloader


class AutoMoMLNIDSDemo:
    """
    Automatic MoMLNIDS demonstration with real network intrusion data.

    Loads data from parquet files and performs comprehensive predictions
    with detailed analysis and visualization.
    """

    def __init__(
        self, model_path: str, data_path: str = "src/data/parquet", device: str = "auto"
    ):
        """
        Initialize automatic demo.

        Args:
            model_path: Path to the trained model (.pt file)
            data_path: Path to parquet data directory
            device: Device to run inference on ("cpu", "cuda", or "auto")
        """
        self.console = Console()
        self.device = self._setup_device(device)
        self.model = None
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        # Discover available datasets
        self.available_datasets = self._discover_datasets()
        self._load_model()

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        torch_device = torch.device(device)
        self.console.print(f"🔧 Using device: {torch_device}")
        return torch_device

    def _discover_datasets(self) -> Dict[str, List[str]]:
        """Discover available datasets in the parquet directory."""
        datasets = {}

        for dataset_dir in self.data_path.iterdir():
            if dataset_dir.is_dir():
                parquet_files = list(dataset_dir.glob("*.parquet"))
                if parquet_files:
                    datasets[dataset_dir.name] = sorted([str(f) for f in parquet_files])

        self.console.print(
            f"📊 Found {len(datasets)} datasets: {list(datasets.keys())}"
        )
        return datasets

    def _load_model(self):
        """Load the trained MoMLNIDS model."""
        self.console.print(f"📁 Loading model from: {self.model_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Extract model parameters from checkpoint structure
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Infer model architecture from state dict
            model_config = self._infer_model_config(state_dict)

            # Create model with inferred config
            self.model = momlnids(**model_config)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.console.print("✅ Model loaded successfully")
            self._print_model_info(model_config)

        except Exception as e:
            self.console.print(f"❌ Error loading model: {e}")
            raise

    def _infer_model_config(self, state_dict: dict) -> dict:
        """Infer model configuration from state dictionary."""
        config = {
            "input_nodes": 39,  # Default
            "hidden_nodes": [64, 32, 16, 10],
            "classifier_nodes": [64, 32, 16],
            "num_domains": 3,
            "num_class": 2,
            "single_layer": True,
        }

        # Infer input nodes from first layer
        for key in state_dict.keys():
            if "FeatureExtractorLayer.fc_modules.0.0.weight" in key:
                config["input_nodes"] = state_dict[key].shape[1]
                break

        # Infer hidden layer dimensions by examining FeatureExtractor layers
        hidden_nodes = []
        max_fe_layer = -1

        # Find max FeatureExtractor layer
        for key in state_dict.keys():
            if "FeatureExtractorLayer.fc_modules" in key and ".weight" in key:
                try:
                    layer_num = int(key.split(".")[2])
                    max_fe_layer = max(max_fe_layer, layer_num)
                except (ValueError, IndexError):
                    continue

        # Extract hidden layer dimensions
        for layer_num in range(max_fe_layer + 1):
            weight_key = f"FeatureExtractorLayer.fc_modules.{layer_num}.0.weight"
            if weight_key in state_dict:
                hidden_nodes.append(state_dict[weight_key].shape[0])

        if hidden_nodes:
            config["hidden_nodes"] = hidden_nodes

        # Infer number of classes from LabelClassifier output layer
        if "LabelClassifier.output_layer.0.weight" in state_dict:
            config["num_class"] = state_dict[
                "LabelClassifier.output_layer.0.weight"
            ].shape[0]

        # Infer number of domains from DomainClassifier output layer
        if "DomainClassifier.output_layer.0.weight" in state_dict:
            config["num_domains"] = state_dict[
                "DomainClassifier.output_layer.0.weight"
            ].shape[0]

        # Infer classifier nodes from DomainClassifier architecture
        classifier_nodes = []
        max_dc_layer = -1

        # Find max DomainClassifier layer
        for key in state_dict.keys():
            if "DomainClassifier.fc_modules" in key and ".weight" in key:
                try:
                    layer_num = int(key.split(".")[2])
                    max_dc_layer = max(max_dc_layer, layer_num)
                except (ValueError, IndexError):
                    continue

        # Extract classifier layer dimensions
        for layer_num in range(max_dc_layer + 1):
            weight_key = f"DomainClassifier.fc_modules.{layer_num}.0.weight"
            if weight_key in state_dict:
                classifier_nodes.append(state_dict[weight_key].shape[0])

        if classifier_nodes:
            config["classifier_nodes"] = classifier_nodes

        return config

    def _print_model_info(self, config: dict):
        """Print model configuration information."""
        info_table = Table(title="Model Configuration")
        info_table.add_column("Parameter", style="cyan")
        info_table.add_column("Value", style="green")

        for key, value in config.items():
            info_table.add_row(str(key), str(value))

        self.console.print(info_table)

    def load_sample_data(
        self, dataset_name: str, num_samples: int = 25000, percentage: float = None
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Load sample data from a specific dataset.

        Args:
            dataset_name: Name of the dataset to load
            num_samples: Number of samples to load (ignored if percentage is set)
            percentage: Percentage of dataset to load (0.0-1.0), overrides num_samples

        Returns:
            features, labels, metadata
        """
        if dataset_name not in self.available_datasets:
            raise ValueError(
                f"Dataset {dataset_name} not found. Available: {list(self.available_datasets.keys())}"
            )

        parquet_files = self.available_datasets[dataset_name]

        # Determine sampling strategy
        if percentage is not None:
            # Use percentage-based sampling
            total_files = len(parquet_files)
            files_to_sample = max(1, int(total_files * percentage))
            selected_files = random.sample(parquet_files, files_to_sample)
            samples_per_file = None  # Use all samples from selected files
            self.console.print(
                f"📊 Loading {percentage:.1%} of {dataset_name} ({files_to_sample}/{total_files} chunks)..."
            )
        else:
            # Use fixed number sampling
            selected_files = random.sample(parquet_files, min(5, len(parquet_files)))
            samples_per_file = num_samples // len(selected_files)
            self.console.print(
                f"📊 Loading {num_samples:,} samples from {dataset_name}..."
            )

        all_features = []
        all_labels = []
        all_attacks = []
        total_loaded = 0

        for file_path in track(selected_files, description="Loading chunks"):
            df = pl.read_parquet(file_path)

            # Sample from this chunk
            if samples_per_file is not None and len(df) > samples_per_file:
                indices = random.sample(range(len(df)), samples_per_file)
                df = df[indices]

            # Extract features (columns 4-42, which are indices 4:43)
            features = (
                df.select(pl.nth(range(4, 43)))
                .with_columns(pl.col("*").cast(pl.Float32))
                .to_numpy()
            )

            # Extract labels
            labels = df.select(pl.col("Label")).cast(pl.Int64).to_numpy().flatten()
            attacks = df.select(pl.col("Attack")).to_numpy().flatten()

            # Handle NaN values
            features = np.nan_to_num(features, nan=0.0, posinf=1e5, neginf=-1e5)

            all_features.append(features)
            all_labels.append(labels)
            all_attacks.append(attacks)
            total_loaded += len(features)

        # Combine all data
        features = np.vstack(all_features)
        labels = np.hstack(all_labels)
        attacks = np.hstack(all_attacks)

        # Final limit check for fixed number sampling
        if percentage is None and len(features) > num_samples:
            indices = random.sample(range(len(features)), num_samples)
            features = features[indices]
            labels = labels[indices]
            attacks = attacks[indices]

        self.console.print(f"✅ Loaded {len(features):,} samples from {dataset_name}")

        metadata = {
            "dataset_name": dataset_name,
            "num_samples": len(features),
            "num_features": features.shape[1],
            "label_distribution": pd.Series(labels).value_counts().to_dict(),
            "attack_distribution": pd.Series(attacks).value_counts().to_dict(),
            "sampling_method": f"{percentage:.1%} of dataset"
            if percentage
            else f"{len(features):,} samples",
            "feature_stats": {
                "mean": np.mean(features, axis=0),
                "std": np.std(features, axis=0),
                "min": np.min(features, axis=0),
                "max": np.max(features, axis=0),
            },
        }

        return features, labels, metadata

    def predict_batch(self, features: np.ndarray) -> List[Dict]:
        """
        Predict on a batch of samples.

        Args:
            features: Batch of input features

        Returns:
            List of prediction dictionaries
        """
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # Get model predictions
            class_logits, domain_logits = self.model(features_tensor)

            # Apply softmax for probabilities
            class_probs = F.softmax(class_logits, dim=1)
            domain_probs = F.softmax(domain_logits, dim=1)

            # Get predictions
            class_preds = torch.argmax(class_probs, dim=1)
            domain_preds = torch.argmax(domain_probs, dim=1)

            # Get confidence scores
            class_confidences = torch.max(class_probs, dim=1)[0]
            domain_confidences = torch.max(domain_probs, dim=1)[0]

            # Format results
            results = []
            for i in range(len(features)):
                results.append(
                    {
                        "class_prediction": class_preds[i].item(),
                        "class_label": "Malicious"
                        if class_preds[i].item() == 1
                        else "Benign",
                        "class_confidence": round(class_confidences[i].item(), 4),
                        "domain_prediction": domain_preds[i].item(),
                        "domain_confidence": round(domain_confidences[i].item(), 4),
                        "class_probabilities": class_probs[i].cpu().numpy().tolist(),
                        "domain_probabilities": domain_probs[i].cpu().numpy().tolist(),
                    }
                )

            return results

    def analyze_predictions(
        self, predictions: List[Dict], true_labels: np.ndarray, metadata: dict
    ) -> Dict:
        """
        Analyze prediction results and compare with ground truth.

        Args:
            predictions: List of prediction dictionaries
            true_labels: Ground truth labels
            metadata: Dataset metadata

        Returns:
            Analysis results
        """
        pred_labels = np.array([p["class_prediction"] for p in predictions])
        confidences = np.array([p["class_confidence"] for p in predictions])
        class_probs = np.array([p["class_probabilities"] for p in predictions])

        # Calculate metrics
        accuracy = np.mean(pred_labels == true_labels)

        # Confusion matrix elements
        tp = np.sum((pred_labels == 1) & (true_labels == 1))
        tn = np.sum((pred_labels == 0) & (true_labels == 0))
        fp = np.sum((pred_labels == 1) & (true_labels == 0))
        fn = np.sum((pred_labels == 0) & (true_labels == 1))

        # Calculate precision, recall, f1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Calculate AUC-ROC
        try:
            from sklearn.metrics import roc_auc_score

            auc_roc = roc_auc_score(true_labels, class_probs[:, 1])
        except:
            auc_roc = 0.5  # Default for edge cases

        # Confidence analysis
        correct_mask = pred_labels == true_labels
        avg_confidence_correct = (
            np.mean(confidences[correct_mask]) if np.any(correct_mask) else 0
        )
        avg_confidence_incorrect = (
            np.mean(confidences[~correct_mask]) if np.any(~correct_mask) else 0
        )

        analysis = {
            "dataset_info": {
                "name": metadata["dataset_name"],
                "samples": metadata["num_samples"],
                "features": metadata["num_features"],
            },
            "performance_metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1_score, 4),
                "auc_roc": round(auc_roc, 4),
            },
            "confusion_matrix": {
                "true_positive": int(tp),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
            },
            "prediction_distribution": {
                "predicted_benign": int(np.sum(pred_labels == 0)),
                "predicted_malicious": int(np.sum(pred_labels == 1)),
                "actual_benign": int(np.sum(true_labels == 0)),
                "actual_malicious": int(np.sum(true_labels == 1)),
            },
            "confidence_analysis": {
                "overall_avg": round(np.mean(confidences), 4),
                "correct_predictions_avg": round(avg_confidence_correct, 4),
                "incorrect_predictions_avg": round(avg_confidence_incorrect, 4),
                "min_confidence": round(np.min(confidences), 4),
                "max_confidence": round(np.max(confidences), 4),
            },
            # Add raw data for JSON export compatibility
            "raw_data": {
                "predictions": pred_labels,
                "probabilities": class_probs,
                "true_labels": true_labels,
            },
        }

        return analysis

    def export_experiment_results(
        self, analysis: Dict, model_config: Dict, output_path: str = None
    ) -> Dict:
        """
        Export results in the same format as experiment_results.json for domain generalization evaluation.

        Args:
            analysis: Analysis results from prediction
            model_config: Model configuration parameters
            output_path: Optional path to save JSON file

        Returns:
            Experiment results dictionary
        """
        import json
        from datetime import datetime

        # Extract target domain from model path
        target_domain = analysis["dataset_info"]["name"]

        # Create experiment results structure
        experiment_results = {
            "config": {
                "model_name": "MoMLNIDS",
                "target_domain": target_domain,
                "batch_size": 128,  # Default from training
                "learning_rate": "N/A",  # Not applicable for inference
                "epochs": "N/A",  # Not applicable for inference
                "datasets": [target_domain],  # Target domain for domain generalization
                "random_seed": 42,
                "model_path": str(self.model_path),
                "evaluation_timestamp": datetime.now().isoformat(),
                "num_samples_evaluated": analysis["dataset_info"]["samples"],
                "model_config": model_config,
            },
            "evaluation_results": {
                target_domain: {
                    "metrics": {
                        f"{target_domain}_accuracy": analysis["performance_metrics"][
                            "accuracy"
                        ],
                        f"{target_domain}_precision": analysis["performance_metrics"][
                            "precision"
                        ],
                        f"{target_domain}_recall": analysis["performance_metrics"][
                            "recall"
                        ],
                        f"{target_domain}_f1_score": analysis["performance_metrics"][
                            "f1_score"
                        ],
                        f"{target_domain}_auc_roc": analysis["performance_metrics"][
                            "auc_roc"
                        ],
                    },
                    "predictions": str(analysis["raw_data"]["predictions"]),
                    "probabilities": str(analysis["raw_data"]["probabilities"]),
                    "true_labels": str(analysis["raw_data"]["true_labels"]),
                    "confusion_matrix": analysis["confusion_matrix"],
                    "confidence_analysis": analysis["confidence_analysis"],
                }
            },
            "domain_generalization_info": {
                "evaluation_type": "target_domain_evaluation",
                "target_domain": target_domain,
                "note": "This is domain generalization evaluation where model trained on source domains is tested on target domain",
            },
        }

        # Save to file if path provided
        if output_path:
            with open(output_path, "w") as f:
                json.dump(experiment_results, f, indent=2)
            self.console.print(f"✅ Experiment results exported to: {output_path}")

        return experiment_results
        """Display comprehensive analysis results."""

        # Dataset Information
        dataset_table = Table(title="📊 Dataset Information")
        dataset_table.add_column("Metric", style="cyan")
        dataset_table.add_column("Value", style="green")

        dataset_table.add_row("Dataset Name", analysis["dataset_info"]["name"])
        dataset_table.add_row("Total Samples", str(analysis["dataset_info"]["samples"]))
        dataset_table.add_row(
            "Feature Dimensions", str(analysis["dataset_info"]["features"])
        )

        # Performance Metrics
        performance_table = Table(title="🎯 Performance Metrics")
        performance_table.add_column("Metric", style="cyan")
        performance_table.add_column("Value", style="green")

        for metric, value in analysis["performance_metrics"].items():
            if metric == "auc_roc":
                performance_table.add_row("AUC-ROC", f"{value:.4f}")
            else:
                performance_table.add_row(
                    metric.replace("_", " ").title(), f"{value:.4f}"
                )

        # Confusion Matrix
        confusion_table = Table(title="📈 Confusion Matrix")
        confusion_table.add_column("Predicted \\ Actual", style="cyan")
        confusion_table.add_column("Benign", style="green")
        confusion_table.add_column("Malicious", style="red")

        cm = analysis["confusion_matrix"]
        confusion_table.add_row(
            "Benign", str(cm["true_negative"]), str(cm["false_negative"])
        )
        confusion_table.add_row(
            "Malicious", str(cm["false_positive"]), str(cm["true_positive"])
        )

        # Prediction Distribution
        distribution_table = Table(title="📋 Prediction vs Ground Truth")
        distribution_table.add_column("Category", style="cyan")
        distribution_table.add_column("Predicted", style="yellow")
        distribution_table.add_column("Actual", style="green")

        dist = analysis["prediction_distribution"]
        distribution_table.add_row(
            "Benign", str(dist["predicted_benign"]), str(dist["actual_benign"])
        )
        distribution_table.add_row(
            "Malicious", str(dist["predicted_malicious"]), str(dist["actual_malicious"])
        )

        # Confidence Analysis
        confidence_table = Table(title="🔍 Confidence Analysis")
        confidence_table.add_column("Metric", style="cyan")
        confidence_table.add_column("Value", style="green")

        for metric, value in analysis["confidence_analysis"].items():
            confidence_table.add_row(metric.replace("_", " ").title(), f"{value:.4f}")

        # Display tables in columns
        self.console.print(Columns([dataset_table, performance_table]))
        self.console.print(Columns([confusion_table, distribution_table]))
        self.console.print(confidence_table)

        # Top predictions sample
        if top_predictions:
            self.console.print(f"\n🔍 Sample Predictions:")

            sample_table = Table(title="Sample Prediction Results")
            sample_table.add_column("Sample", style="cyan")
            sample_table.add_column("Predicted", style="green")
            sample_table.add_column("Confidence", style="yellow")
            sample_table.add_column("Domain", style="magenta")

            for i, pred in enumerate(top_predictions[:10]):
                sample_table.add_row(
                    str(i + 1),
                    pred["class_label"],
                    f"{pred['class_confidence']:.4f}",
                    str(pred["domain_prediction"]),
                )

            self.console.print(sample_table)

    def run_demo(
        self,
        dataset_name: str = None,
        num_samples: int = 25000,
        show_samples: bool = True,
        export_json: str = None,
        percentage: float = None,
    ):
        """
        Run automatic demonstration with real data.

        Args:
            dataset_name: Specific dataset to use (if None, will prompt)
            num_samples: Number of samples to process (ignored if percentage is set)
            show_samples: Whether to show sample predictions
            export_json: Path to export JSON results
            percentage: Percentage of dataset to use (0.0-1.0), overrides num_samples
        """
        self.console.print(
            Panel.fit("🚀 MoMLNIDS Automatic Demo with Real Data", style="bold blue")
        )

        # Select dataset
        if dataset_name is None:
            self.console.print("\n📊 Available Datasets:")
            for i, name in enumerate(self.available_datasets.keys(), 1):
                count = len(self.available_datasets[name])
                self.console.print(f"  {i}. {name} ({count} chunks)")

            # For demo, just use the first available dataset
            dataset_name = list(self.available_datasets.keys())[0]
            self.console.print(f"\n🎯 Using dataset: {dataset_name}")

        # Load data
        try:
            features, labels, metadata = self.load_sample_data(
                dataset_name, num_samples, percentage
            )

            # Make predictions
            self.console.print(f"\n🔍 Making predictions on {len(features)} samples...")
            predictions = self.predict_batch(features)

            # Analyze results
            self.console.print(f"\n📊 Analyzing results...")
            analysis = self.analyze_predictions(predictions, labels, metadata)

            # Export to JSON if requested
            if export_json:
                model_config = {
                    "input_nodes": self.model.FeatureExtractorLayer.input_nodes,
                    "hidden_nodes": getattr(
                        self.model.FeatureExtractorLayer,
                        "hidden_nodes",
                        [64, 32, 16, 10],
                    ),
                    "num_domains": getattr(
                        self.model.DomainClassifier, "output_nodes", 4
                    ),
                    "num_class": getattr(self.model.LabelClassifier, "output_nodes", 2),
                }
                experiment_results = self.export_experiment_results(
                    analysis, model_config, export_json
                )

            # Display results
            self.display_results(analysis, predictions if show_samples else None)

            # Attack type analysis if available
            if (
                "attack_distribution" in metadata
                and len(metadata["attack_distribution"]) > 1
            ):
                self.console.print(f"\n🔍 Attack Type Distribution:")
                attack_table = Table(title="Attack Types in Dataset")
                attack_table.add_column("Attack Type", style="cyan")
                attack_table.add_column("Count", style="green")
                attack_table.add_column("Percentage", style="yellow")

                total_attacks = sum(metadata["attack_distribution"].values())
                for attack, count in sorted(
                    metadata["attack_distribution"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]:
                    percentage = (count / total_attacks) * 100
                    attack_table.add_row(str(attack), str(count), f"{percentage:.2f}%")

                self.console.print(attack_table)

            self.console.print(f"\n✨ Demo completed successfully!")

            return analysis

        except Exception as e:
            self.console.print(f"❌ Error during demo: {e}")
            raise

    def display_results(self, analysis: Dict, top_predictions: List[Dict] = None):
        """Display comprehensive analysis results."""

        # Dataset Information
        dataset_table = Table(title="📊 Dataset Information")
        dataset_table.add_column("Metric", style="cyan")
        dataset_table.add_column("Value", style="green")

        dataset_table.add_row("Dataset Name", analysis["dataset_info"]["name"])
        dataset_table.add_row("Total Samples", str(analysis["dataset_info"]["samples"]))
        dataset_table.add_row(
            "Feature Dimensions", str(analysis["dataset_info"]["features"])
        )

        # Performance Metrics
        performance_table = Table(title="🎯 Performance Metrics")
        performance_table.add_column("Metric", style="cyan")
        performance_table.add_column("Value", style="green")

        for metric, value in analysis["performance_metrics"].items():
            if metric == "auc_roc":
                performance_table.add_row("AUC-ROC", f"{value:.4f}")
            else:
                performance_table.add_row(
                    metric.replace("_", " ").title(), f"{value:.4f}"
                )

        # Confusion Matrix
        confusion_table = Table(title="📈 Confusion Matrix")
        confusion_table.add_column("Predicted \\ Actual", style="cyan")
        confusion_table.add_column("Benign", style="green")
        confusion_table.add_column("Malicious", style="red")

        cm = analysis["confusion_matrix"]
        confusion_table.add_row(
            "Benign", str(cm["true_negative"]), str(cm["false_negative"])
        )
        confusion_table.add_row(
            "Malicious", str(cm["false_positive"]), str(cm["true_positive"])
        )

        # Prediction Distribution
        distribution_table = Table(title="📋 Prediction vs Ground Truth")
        distribution_table.add_column("Category", style="cyan")
        distribution_table.add_column("Predicted", style="yellow")
        distribution_table.add_column("Actual", style="green")

        dist = analysis["prediction_distribution"]
        distribution_table.add_row(
            "Benign", str(dist["predicted_benign"]), str(dist["actual_benign"])
        )
        distribution_table.add_row(
            "Malicious", str(dist["predicted_malicious"]), str(dist["actual_malicious"])
        )

        # Confidence Analysis
        confidence_table = Table(title="🔍 Confidence Analysis")
        confidence_table.add_column("Metric", style="cyan")
        confidence_table.add_column("Value", style="green")

        for metric, value in analysis["confidence_analysis"].items():
            confidence_table.add_row(metric.replace("_", " ").title(), f"{value:.4f}")

        # Display tables in columns
        self.console.print(Columns([dataset_table, performance_table]))
        self.console.print(Columns([confusion_table, distribution_table]))
        self.console.print(confidence_table)

        # Top predictions sample
        if top_predictions:
            self.console.print(f"\n🔍 Sample Predictions:")

            sample_table = Table(title="Sample Prediction Results")
            sample_table.add_column("Sample", style="cyan")
            sample_table.add_column("Predicted", style="green")
            sample_table.add_column("Confidence", style="yellow")
            sample_table.add_column("Domain", style="magenta")

            for i, pred in enumerate(top_predictions[:10]):
                sample_table.add_row(
                    str(i + 1),
                    pred["class_label"],
                    f"{pred['class_confidence']:.4f}",
                    str(pred["domain_prediction"]),
                )

            self.console.print(sample_table)

    def export_experiment_results(
        self, analysis: Dict, model_config: Dict, output_path: str = None
    ) -> Dict:
        """
        Export results in the same format as experiment_results.json for domain generalization evaluation.
        """
        import json
        from datetime import datetime

        target_domain = analysis["dataset_info"]["name"]

        experiment_results = {
            "config": {
                "model_name": "MoMLNIDS",
                "target_domain": target_domain,
                "batch_size": 128,
                "learning_rate": "N/A",
                "epochs": "N/A",
                "datasets": [target_domain],
                "random_seed": 42,
                "model_path": str(self.model_path),
                "evaluation_timestamp": datetime.now().isoformat(),
                "num_samples_evaluated": analysis["dataset_info"]["samples"],
                "model_config": model_config,
            },
            "evaluation_results": {
                target_domain: {
                    "metrics": {
                        f"{target_domain}_accuracy": analysis["performance_metrics"][
                            "accuracy"
                        ],
                        f"{target_domain}_precision": analysis["performance_metrics"][
                            "precision"
                        ],
                        f"{target_domain}_recall": analysis["performance_metrics"][
                            "recall"
                        ],
                        f"{target_domain}_f1_score": analysis["performance_metrics"][
                            "f1_score"
                        ],
                        f"{target_domain}_auc_roc": analysis["performance_metrics"][
                            "auc_roc"
                        ],
                    },
                    "predictions": str(analysis["raw_data"]["predictions"]),
                    "probabilities": str(analysis["raw_data"]["probabilities"]),
                    "true_labels": str(analysis["raw_data"]["true_labels"]),
                    "confusion_matrix": analysis["confusion_matrix"],
                    "confidence_analysis": analysis["confidence_analysis"],
                }
            },
            "domain_generalization_info": {
                "evaluation_type": "target_domain_evaluation",
                "target_domain": target_domain,
                "note": "This is domain generalization evaluation where model trained on source domains is tested on target domain",
            },
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(experiment_results, f, indent=2)
            self.console.print(f"✅ Experiment results exported to: {output_path}")

        return experiment_results


@click.command()
@click.option(
    "--model-path", "-m", required=True, help="Path to trained model (.pt file)"
)
@click.option(
    "--data-path",
    "-d",
    default="src/data/parquet",
    help="Path to parquet data directory",
)
@click.option("--dataset", "-ds", help="Specific dataset to use")
@click.option(
    "--num-samples",
    "-n",
    default=25000,
    help="Number of samples to process (ignored if --percentage is used)",
)
@click.option(
    "--percentage",
    "-p",
    type=float,
    help="Percentage of dataset to use (0.0-1.0), e.g., 0.1 for 10%",
)
@click.option("--device", default="auto", help="Device to use (cpu/cuda/auto)")
@click.option("--show-samples", is_flag=True, help="Show individual sample predictions")
@click.option("--all-datasets", is_flag=True, help="Run demo on all available datasets")
@click.option("--export-json", help="Export results to JSON file")
def main(
    model_path,
    data_path,
    dataset,
    num_samples,
    percentage,
    device,
    show_samples,
    all_datasets,
    export_json,
):
    """
    MoMLNIDS Automatic Prediction Demo with Real Data

    Demonstrates network intrusion detection using trained MoMLNIDS models
    with real data from parquet files.

    Sample Usage:

    # Small test (default 25k samples)
    python auto_prediction_demo.py -m model.pt --dataset NF-CSE-CIC-IDS2018-v2

    # Use percentage of dataset
    python auto_prediction_demo.py -m model.pt --dataset NF-CSE-CIC-IDS2018-v2 --percentage 0.1  # 10%
    python auto_prediction_demo.py -m model.pt --dataset NF-CSE-CIC-IDS2018-v2 --percentage 0.5  # 50%
    python auto_prediction_demo.py -m model.pt --dataset NF-CSE-CIC-IDS2018-v2 --percentage 1.0  # 100%

    # Large sample counts
    python auto_prediction_demo.py -m model.pt --dataset NF-CSE-CIC-IDS2018-v2 --num-samples 500000   # 500k
    python auto_prediction_demo.py -m model.pt --dataset NF-CSE-CIC-IDS2018-v2 --num-samples 1000000  # 1M
    """
    console = Console()

    try:
        demo = AutoMoMLNIDSDemo(model_path, data_path, device)

        if all_datasets:
            results = {}
            for dataset_name in demo.available_datasets.keys():
                console.print(f"\n{'=' * 60}")
                console.print(f"Running demo on dataset: {dataset_name}")
                console.print(f"{'=' * 60}")

                analysis = demo.run_demo(
                    dataset_name, num_samples, show_samples, None, percentage
                )
                results[dataset_name] = analysis

            console.print(f"\n{'=' * 60}")
            console.print("📊 Summary Across All Datasets")
            console.print(f"{'=' * 60}")

            summary_table = Table(title="Performance Summary")
            summary_table.add_column("Dataset", style="cyan")
            summary_table.add_column("Accuracy", style="green")
            summary_table.add_column("F1-Score", style="yellow")
            summary_table.add_column("Samples", style="magenta")

            for dataset_name, analysis in results.items():
                summary_table.add_row(
                    dataset_name,
                    f"{analysis['performance_metrics']['accuracy']:.4f}",
                    f"{analysis['performance_metrics']['f1_score']:.4f}",
                    str(analysis["dataset_info"]["samples"]),
                )

            console.print(summary_table)

        else:
            demo.run_demo(dataset, num_samples, show_samples, export_json, percentage)

    except Exception as e:
        console.print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
