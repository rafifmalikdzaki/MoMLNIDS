#!/usr/bin/env python3
"""
MoMLNIDS Per-Sample/Per-Batch Prediction Demo

This script demonstrates how to load trained MoMLNIDS models and perform
predictions on individual samples or batches of network intrusion data.
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
import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.append("src")

from skripsi_code.model.MoMLNIDS import momlnids
from skripsi_code.utils.dataloader import random_split_dataloader


class MoMLNIDSPredictor:
    """
    MoMLNIDS model predictor for network intrusion detection.

    Supports both per-sample and per-batch predictions with confidence scores.
    """

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize predictor with trained model.

        Args:
            model_path: Path to the trained model (.pt file)
            device: Device to run inference on ("cpu", "cuda", or "auto")
        """
        self.console = Console()
        self.device = self._setup_device(device)
        self.model = None
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._load_model()

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        torch_device = torch.device(device)
        self.console.print(f"🔧 Using device: {torch_device}")
        return torch_device

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

            # Debug: print some keys to understand structure
            self.console.print("🔍 Analyzing model structure...")
            sample_keys = list(state_dict.keys())[:10]
            for key in sample_keys:
                self.console.print(f"  - {key}: {state_dict[key].shape}")

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
            "input_nodes": 43,  # Default
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
        for key in sorted(state_dict.keys()):
            if "FeatureExtractorLayer.fc_modules" in key and ".weight" in key:
                layer_num = int(key.split(".")[2])
                if layer_num < 10:  # Avoid parsing issues
                    weight_shape = state_dict[key].shape
                    if len(hidden_nodes) <= layer_num:
                        hidden_nodes.extend([0] * (layer_num + 1 - len(hidden_nodes)))
                    hidden_nodes[layer_num] = weight_shape[0]

        if hidden_nodes:
            config["hidden_nodes"] = [h for h in hidden_nodes if h > 0]

        # Infer number of classes from LabelClassifier final layer
        for key in state_dict.keys():
            if "LabelClassifier" in key and "weight" in key:
                # Look for the final layer (output layer)
                if any(
                    final_key in key for final_key in ["final_layer", "output_layer"]
                ):
                    config["num_class"] = state_dict[key].shape[0]
                elif key.endswith(".weight") and "LabelClassifier" in key:
                    # If no explicit final layer, take the last one we find
                    config["num_class"] = state_dict[key].shape[0]

        # Infer number of domains from DomainClassifier final layer
        for key in state_dict.keys():
            if "DomainClassifier" in key and "weight" in key:
                if any(
                    final_key in key for final_key in ["final_layer", "output_layer"]
                ):
                    config["num_domains"] = state_dict[key].shape[0]
                elif key.endswith(".weight") and "DomainClassifier" in key:
                    config["num_domains"] = state_dict[key].shape[0]

        # Infer classifier nodes from DomainClassifier architecture
        classifier_nodes = []
        for key in sorted(state_dict.keys()):
            if (
                "DomainClassifier" in key
                and "weight" in key
                and "output_layer" not in key
            ):
                layer_match = key.split(".")
                if len(layer_match) >= 3:
                    try:
                        layer_num = int(layer_match[2])
                        weight_shape = state_dict[key].shape
                        if len(classifier_nodes) <= layer_num:
                            classifier_nodes.extend(
                                [0] * (layer_num + 1 - len(classifier_nodes))
                            )
                        classifier_nodes[layer_num] = weight_shape[0]
                    except (ValueError, IndexError):
                        continue

        if classifier_nodes:
            config["classifier_nodes"] = [c for c in classifier_nodes if c > 0]

        return config

    def _print_model_info(self, config: dict):
        """Print model configuration information."""
        info_table = Table(title="Model Configuration")
        info_table.add_column("Parameter", style="cyan")
        info_table.add_column("Value", style="green")

        for key, value in config.items():
            info_table.add_row(str(key), str(value))

        self.console.print(info_table)

    def predict_sample(
        self, features: Union[np.ndarray, torch.Tensor, List[float]]
    ) -> Dict[str, Union[int, float, str]]:
        """
        Predict on a single sample.

        Args:
            features: Input features (should match model's input dimension)

        Returns:
            Dictionary containing prediction results
        """  # Convert input to tensor
        if isinstance(features, (list, np.ndarray)):
            features = torch.tensor(features, dtype=torch.float32)

        if features.dim() == 1:
            features = features.unsqueeze(0)  # Add batch dimension

        features = features.to(self.device)

        with torch.no_grad():
            # Get model predictions
            class_logits, domain_logits = self.model(features)

            # Apply softmax for probabilities
            class_probs = F.softmax(class_logits, dim=1)
            domain_probs = F.softmax(domain_logits, dim=1)

            # Get predictions
            class_pred = torch.argmax(class_probs, dim=1).item()
            domain_pred = torch.argmax(domain_probs, dim=1).item()

            # Get confidence scores
            class_confidence = torch.max(class_probs, dim=1)[0].item()
            domain_confidence = torch.max(domain_probs, dim=1)[0].item()

            return {
                "class_prediction": class_pred,
                "class_label": "Malicious" if class_pred == 1 else "Benign",
                "class_confidence": round(class_confidence, 4),
                "domain_prediction": domain_pred,
                "domain_confidence": round(domain_confidence, 4),
                "class_probabilities": class_probs.cpu().numpy().tolist()[0],
                "domain_probabilities": domain_probs.cpu().numpy().tolist()[0],
            }

    def predict_batch(
        self, features_batch: Union[np.ndarray, torch.Tensor]
    ) -> List[Dict[str, Union[int, float, str]]]:
        """
        Predict on a batch of samples.

        Args:
            features_batch: Batch of input features (N x input_dim)

        Returns:
            List of prediction dictionaries for each sample
        """
        if isinstance(features_batch, np.ndarray):
            features_batch = torch.tensor(features_batch, dtype=torch.float32)

        features_batch = features_batch.to(self.device)

        with torch.no_grad():
            # Get model predictions
            class_logits, domain_logits = self.model(features_batch)

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
            for i in range(len(features_batch)):
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
        self, predictions: List[Dict]
    ) -> Dict[str, Union[int, float]]:
        """
        Analyze batch predictions to get summary statistics.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            Analysis summary
        """
        if not predictions:
            return {}

        class_preds = [p["class_prediction"] for p in predictions]
        class_confidences = [p["class_confidence"] for p in predictions]
        domain_confidences = [p["domain_confidence"] for p in predictions]

        analysis = {
            "total_samples": len(predictions),
            "benign_count": sum(1 for p in class_preds if p == 0),
            "malicious_count": sum(1 for p in class_preds if p == 1),
            "benign_percentage": round(
                (sum(1 for p in class_preds if p == 0) / len(class_preds)) * 100, 2
            ),
            "malicious_percentage": round(
                (sum(1 for p in class_preds if p == 1) / len(class_preds)) * 100, 2
            ),
            "avg_class_confidence": round(np.mean(class_confidences), 4),
            "min_class_confidence": round(min(class_confidences), 4),
            "max_class_confidence": round(max(class_confidences), 4),
            "avg_domain_confidence": round(np.mean(domain_confidences), 4),
        }

        return analysis


def generate_sample_data(
    num_samples: int = 10, input_dim: int = 39, seed: int = 42
) -> np.ndarray:
    """Generate synthetic network flow features for demonstration."""
    np.random.seed(seed)

    # Generate features with the correct input dimension
    features = np.random.randn(num_samples, input_dim)

    # Normalize features to reasonable ranges
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    return features.astype(np.float32)


@click.command()
@click.option(
    "--model-path", "-m", required=True, help="Path to trained model (.pt file)"
)
@click.option("--device", "-d", default="auto", help="Device to use (cpu/cuda/auto)")
@click.option(
    "--mode",
    "-o",
    default="demo",
    type=click.Choice(["demo", "single", "batch"]),
    help="Prediction mode",
)
@click.option(
    "--num-samples", "-n", default=10000, help="Number of samples for demo/batch mode"
)
@click.option("--input-file", "-i", help="CSV file with input features")
@click.option("--output-file", "-o", help="Save predictions to file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def main(model_path, device, mode, num_samples, input_file, output_file, verbose):
    """
    MoMLNIDS Prediction Demo

    Demonstrates per-sample and per-batch prediction capabilities
    of trained MoMLNIDS models for network intrusion detection.
    """
    console = Console()

    console.print(Panel.fit("🔍 MoMLNIDS Prediction Demo", style="bold blue"))

    try:
        # Initialize predictor
        predictor = MoMLNIDSPredictor(model_path, device)

        if mode == "demo":
            # Demo mode with synthetic data
            console.print(f"\n🎯 Running demo with {num_samples} synthetic samples...")

            # Get input dimension from loaded model
            input_dim = predictor.model.FeatureExtractorLayer.input_nodes
            console.print(f"📏 Model input dimension: {input_dim}")

            # Generate sample data
            sample_data = generate_sample_data(num_samples, input_dim)

            # Single sample prediction
            console.print("\n📊 Single Sample Prediction:")
            single_result = predictor.predict_sample(sample_data[0])

            single_table = Table(title="Single Sample Results")
            single_table.add_column("Metric", style="cyan")
            single_table.add_column("Value", style="green")

            single_table.add_row("Class Prediction", single_result["class_label"])
            single_table.add_row(
                "Class Confidence", f"{single_result['class_confidence']:.4f}"
            )
            single_table.add_row(
                "Domain Prediction", str(single_result["domain_prediction"])
            )
            single_table.add_row(
                "Domain Confidence", f"{single_result['domain_confidence']:.4f}"
            )

            console.print(single_table)

            if verbose:
                console.print(
                    f"\nClass Probabilities: {single_result['class_probabilities']}"
                )
                console.print(
                    f"Domain Probabilities: {single_result['domain_probabilities']}"
                )

            # Batch prediction
            console.print(f"\n📈 Batch Prediction ({num_samples} samples):")
            batch_results = predictor.predict_batch(sample_data)

            # Display batch results
            batch_table = Table(title="Batch Prediction Results")
            batch_table.add_column("Sample", style="cyan")
            batch_table.add_column("Class", style="green")
            batch_table.add_column("Confidence", style="yellow")
            batch_table.add_column("Domain", style="magenta")

            for i, result in enumerate(batch_results):
                batch_table.add_row(
                    str(i + 1),
                    result["class_label"],
                    f"{result['class_confidence']:.4f}",
                    str(result["domain_prediction"]),
                )

            console.print(batch_table)

            # Analysis
            analysis = predictor.analyze_predictions(batch_results)

            analysis_table = Table(title="Batch Analysis")
            analysis_table.add_column("Metric", style="cyan")
            analysis_table.add_column("Value", style="green")

            analysis_table.add_row("Total Samples", str(analysis["total_samples"]))
            analysis_table.add_row(
                "Benign Samples",
                f"{analysis['benign_count']} ({analysis['benign_percentage']}%)",
            )
            analysis_table.add_row(
                "Malicious Samples",
                f"{analysis['malicious_count']} ({analysis['malicious_percentage']}%)",
            )
            analysis_table.add_row(
                "Avg Class Confidence", f"{analysis['avg_class_confidence']:.4f}"
            )
            analysis_table.add_row(
                "Min/Max Confidence",
                f"{analysis['min_class_confidence']:.4f} / {analysis['max_class_confidence']:.4f}",
            )

            console.print(analysis_table)

        elif mode == "single":
            # Single prediction mode with user input
            console.print("\n🎯 Single Sample Prediction Mode")
            input_dim = predictor.model.FeatureExtractorLayer.input_nodes
            console.print(f"Enter {input_dim} comma-separated feature values:")

            try:
                user_input = input("Features: ")
                features = [float(x.strip()) for x in user_input.split(",")]

                if len(features) != input_dim:
                    raise ValueError(
                        f"Expected {input_dim} features, got {len(features)}"
                    )

                result = predictor.predict_sample(features)

                result_table = Table(title="Prediction Result")
                result_table.add_column("Metric", style="cyan")
                result_table.add_column("Value", style="green")

                for key, value in result.items():
                    if key not in ["class_probabilities", "domain_probabilities"]:
                        result_table.add_row(key.replace("_", " ").title(), str(value))

                console.print(result_table)

            except (ValueError, EOFError) as e:
                console.print(f"❌ Input error: {e}")
        elif mode == "batch" and input_file:
            # Batch mode with file input
            console.print(f"\n📈 Batch Prediction from file: {input_file}")

            try:
                # Load data from CSV
                df = pd.read_csv(input_file)
                features = df.values.astype(np.float32)

                console.print(
                    f"Loaded {len(features)} samples with {features.shape[1]} features"
                )

                # Make predictions
                with Progress() as progress:
                    task = progress.add_task("Predicting...", total=len(features))

                    batch_size = 32
                    all_results = []

                    for i in range(0, len(features), batch_size):
                        batch = features[i : i + batch_size]
                        batch_results = predictor.predict_batch(batch)
                        all_results.extend(batch_results)
                        progress.advance(task, len(batch))

                # Analysis
                analysis = predictor.analyze_predictions(all_results)

                analysis_table = Table(title="File Prediction Analysis")
                analysis_table.add_column("Metric", style="cyan")
                analysis_table.add_column("Value", style="green")

                for key, value in analysis.items():
                    analysis_table.add_row(key.replace("_", " ").title(), str(value))

                console.print(analysis_table)

                # Save results if requested
                if output_file:
                    results_df = pd.DataFrame(all_results)
                    results_df.to_csv(output_file, index=False)
                    console.print(f"✅ Results saved to: {output_file}")

            except Exception as e:
                console.print(f"❌ Error processing file: {e}")

        else:
            console.print("❌ Invalid mode or missing input file for batch mode")

        console.print("\n✨ Prediction demo completed!")

    except Exception as e:
        console.print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
