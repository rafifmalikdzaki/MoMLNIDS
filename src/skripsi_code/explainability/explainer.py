"""Explainable AI module for NIDS model interpretability."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional imports with fallbacks
try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import click

    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Main explainer class that provides multiple interpretability methods."""

    def __init__(self, model, feature_names: List[str]):
        """
        Initialize model explainer.

        Args:
            model: Trained PyTorch model (or None if torch not available)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names

        if TORCH_AVAILABLE and model is not None:
            self.device = next(model.parameters()).device
            # Set model to evaluation mode
            self.model.eval()
        else:
            self.device = None

    def explain_instance(
        self, instance: np.ndarray, method: str = "integrated_gradients", **kwargs
    ) -> Dict[str, Any]:
        """
        Explain a single instance prediction.

        Args:
            instance: Input instance to explain
            method: Explanation method to use
            **kwargs: Additional arguments for specific methods

        Returns:
            Dictionary containing explanation results
        """
        if method == "integrated_gradients":
            return self._integrated_gradients_explanation(instance, **kwargs)
        elif method == "gradient_shap":
            return self._gradient_shap_explanation(instance, **kwargs)
        elif method == "feature_ablation":
            return self._feature_ablation_explanation(instance, **kwargs)
        elif method == "lime":
            return self._lime_explanation(instance, **kwargs)
        else:
            raise ValueError(f"Unknown explanation method: {method}")

    def explain_global(
        self, X_sample: np.ndarray, method: str = "feature_importance", **kwargs
    ) -> Dict[str, Any]:
        """
        Provide global model explanations.

        Args:
            X_sample: Sample of data for global explanation
            method: Global explanation method
            **kwargs: Additional arguments

        Returns:
            Dictionary containing global explanation results
        """
        if method == "feature_importance":
            return self._global_feature_importance(X_sample, **kwargs)
        elif method == "feature_interaction":
            return self._feature_interaction_analysis(X_sample, **kwargs)
        else:
            raise ValueError(f"Unknown global explanation method: {method}")

    def _integrated_gradients_explanation(
        self,
        instance: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        steps: int = 50,
    ) -> Dict[str, Any]:
        """
        Compute integrated gradients for feature attribution.

        Args:
            instance: Input instance
            baseline: Baseline for integrated gradients (default: zeros)
            steps: Number of integration steps

        Returns:
            Dictionary with attribution scores
        """
        instance_tensor = torch.tensor(
            instance, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        if baseline is None:
            baseline = np.zeros_like(instance)
        baseline_tensor = torch.tensor(
            baseline, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Compute integrated gradients
        attributions = []

        for step in range(steps + 1):
            # Interpolate between baseline and instance
            alpha = step / steps
            interpolated = baseline_tensor + alpha * (instance_tensor - baseline_tensor)
            interpolated.requires_grad_(True)

            # Forward pass
            output = self.model(interpolated)

            # Get prediction for dominant class
            if output.dim() > 1:
                target_class = output.argmax(dim=1)
                target_output = output[0, target_class]
            else:
                target_output = output[0]

            # Compute gradients
            gradients = torch.autograd.grad(target_output, interpolated)[0]
            attributions.append(gradients.cpu().numpy())

        # Integrate gradients
        attributions = np.array(attributions)
        integrated_gradients = np.mean(attributions, axis=0) * (instance - baseline)

        return {
            "method": "integrated_gradients",
            "attributions": integrated_gradients.flatten(),
            "feature_names": self.feature_names,
            "instance": instance,
            "baseline": baseline,
        }

    def _gradient_shap_explanation(
        self,
        instance: np.ndarray,
        background_samples: Optional[np.ndarray] = None,
        n_samples: int = 10,
    ) -> Dict[str, Any]:
        """
        Compute GradientSHAP attributions.

        Args:
            instance: Input instance
            background_samples: Background samples for noise tunnel
            n_samples: Number of samples for noise tunnel

        Returns:
            Dictionary with attribution scores
        """
        instance_tensor = torch.tensor(
            instance, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        if background_samples is None:
            # Create random background samples
            background_samples = np.random.normal(0, 0.1, (n_samples,) + instance.shape)

        attributions_list = []

        for bg_sample in background_samples:
            bg_tensor = torch.tensor(
                bg_sample, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            # Compute gradients for interpolated samples
            for alpha in np.linspace(0, 1, 10):
                interpolated = bg_tensor + alpha * (instance_tensor - bg_tensor)
                interpolated.requires_grad_(True)

                output = self.model(interpolated)

                if output.dim() > 1:
                    target_class = output.argmax(dim=1)
                    target_output = output[0, target_class]
                else:
                    target_output = output[0]

                gradients = torch.autograd.grad(target_output, interpolated)[0]
                attributions_list.append(gradients.cpu().numpy())

        # Average attributions
        mean_attributions = np.mean(attributions_list, axis=0) * instance

        return {
            "method": "gradient_shap",
            "attributions": mean_attributions.flatten(),
            "feature_names": self.feature_names,
            "instance": instance,
        }

    def _feature_ablation_explanation(
        self, instance: np.ndarray, baseline_value: float = 0.0
    ) -> Dict[str, Any]:
        """
        Compute feature importance using ablation study.

        Args:
            instance: Input instance
            baseline_value: Value to use for ablated features

        Returns:
            Dictionary with feature importance scores
        """
        instance_tensor = torch.tensor(
            instance, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Get original prediction
        with torch.no_grad():
            original_output = self.model(instance_tensor)
            if isinstance(original_output, tuple):
                original_output = original_output[
                    0
                ]  # Take the first element if it's a tuple
            if original_output.dim() > 1:
                original_pred = original_output.max(dim=1)[0].item()
            else:
                original_pred = original_output.item()

        importance_scores = []

        # Ablate each feature
        for i in range(len(instance)):
            modified_instance = instance.copy()
            modified_instance[i] = baseline_value

            modified_tensor = torch.tensor(
                modified_instance, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                modified_output = self.model(modified_tensor)
                if isinstance(modified_output, tuple):
                    modified_output = modified_output[
                        0
                    ]  # Take the first element if it's a tuple
                if modified_output.dim() > 1:
                    modified_pred = modified_output.max(dim=1)[0].item()
                else:
                    modified_pred = modified_output.item()

            # Importance is the difference in predictions
            importance = original_pred - modified_pred
            importance_scores.append(importance)

        return {
            "method": "feature_ablation",
            "attributions": np.array(importance_scores),
            "feature_names": self.feature_names,
            "instance": instance,
            "original_prediction": original_pred,
        }

    def _lime_explanation(
        self, instance: np.ndarray, training_data: np.ndarray, **kwargs
    ) -> Dict[str, Any]:
        """
        Compute LIME explanation for a single instance.

        Args:
            instance: Input instance to explain
            training_data: Training data for LIME
            **kwargs: Additional LIME parameters

        Returns:
            Dictionary with LIME explanation results
        """
        lime_explainer = LIMEExplainer(self.model, self.feature_names)
        explanation = lime_explainer.explain_instance(instance, training_data, **kwargs)
        return explanation

    def _global_feature_importance(
        self, X_sample: np.ndarray, method: str = "permutation"
    ) -> Dict[str, Any]:
        """
        Compute global feature importance.

        Args:
            X_sample: Sample of data
            method: Method for computing importance

        Returns:
            Dictionary with global feature importance
        """
        if method == "permutation":
            return self._permutation_importance(X_sample)
        else:
            raise ValueError(f"Unknown global importance method: {method}")

    def _permutation_importance(self, X_sample: np.ndarray) -> Dict[str, Any]:
        """
        Compute permutation feature importance.

        Args:
            X_sample: Sample of data

        Returns:
            Dictionary with permutation importance scores
        """
        X_tensor = torch.tensor(X_sample, dtype=torch.float32, device=self.device)

        # Get baseline performance
        with torch.no_grad():
            baseline_outputs = self.model(X_tensor)
            if baseline_outputs.dim() > 1:
                baseline_preds = baseline_outputs.argmax(dim=1)
                baseline_confidence = baseline_outputs.max(dim=1)[0].mean().item()
            else:
                baseline_preds = baseline_outputs
                baseline_confidence = baseline_outputs.mean().item()

        importance_scores = []

        # Permute each feature
        for i in range(X_sample.shape[1]):
            X_permuted = X_sample.copy()
            np.random.shuffle(X_permuted[:, i])

            X_permuted_tensor = torch.tensor(
                X_permuted, dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                permuted_outputs = self.model(X_permuted_tensor)
                if permuted_outputs.dim() > 1:
                    permuted_confidence = permuted_outputs.max(dim=1)[0].mean().item()
                else:
                    permuted_confidence = permuted_outputs.mean().item()

            # Importance is the drop in performance
            importance = baseline_confidence - permuted_confidence
            importance_scores.append(importance)

        return {
            "method": "permutation_importance",
            "importance_scores": np.array(importance_scores),
            "feature_names": self.feature_names,
            "baseline_performance": baseline_confidence,
        }

    def _feature_interaction_analysis(self, X_sample: np.ndarray) -> Dict[str, Any]:
        """
        Analyze feature interactions.

        Args:
            X_sample: Sample of data

        Returns:
            Dictionary with interaction analysis
        """
        n_features = X_sample.shape[1]
        interaction_matrix = np.zeros((n_features, n_features))

        # Compute pairwise interactions
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Measure interaction between features i and j
                interaction_score = self._compute_pairwise_interaction(X_sample, i, j)
                interaction_matrix[i, j] = interaction_score
                interaction_matrix[j, i] = interaction_score

        return {
            "method": "feature_interaction",
            "interaction_matrix": interaction_matrix,
            "feature_names": self.feature_names,
        }

    def _compute_pairwise_interaction(
        self, X_sample: np.ndarray, feature_i: int, feature_j: int
    ) -> float:
        """
        Compute interaction between two features.

        Args:
            X_sample: Sample of data
            feature_i: Index of first feature
            feature_j: Index of second feature

        Returns:
            Interaction score
        """
        # Simple interaction measure: correlation of gradients
        X_tensor = torch.tensor(X_sample, dtype=torch.float32, device=self.device)
        X_tensor.requires_grad_(True)

        output = self.model(X_tensor)
        if output.dim() > 1:
            target_output = output.sum()
        else:
            target_output = output.sum()

        # Compute gradients
        gradients = torch.autograd.grad(target_output, X_tensor, create_graph=True)[0]

        # Compute second-order partial derivatives
        grad_i = gradients[:, feature_i].sum()
        grad_j_wrt_i = torch.autograd.grad(grad_i, X_tensor, retain_graph=True)[0][
            :, feature_j
        ]

        # Return mean interaction
        return grad_j_wrt_i.mean().item()


class SHAPExplainer:
    """SHAP-based explainer (requires shap library)."""

    def __init__(self, model: nn.Module, background_data: np.ndarray):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained PyTorch model
            background_data: Background dataset for SHAP
        """
        self.model = model
        self.background_data = background_data
        self.device = next(model.parameters()).device

        try:
            import shap

            self.shap = shap
            self.explainer = self._create_shap_explainer()
        except ImportError:
            logger.warning("SHAP library not available. Install with: pip install shap")
            self.shap = None
            self.explainer = None

    def _create_shap_explainer(self):
        """Create SHAP explainer."""

        def model_wrapper(x):
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                output = self.model(x_tensor)
                if output.dim() > 1:
                    return output.cpu().numpy()
                else:
                    return output.cpu().numpy().reshape(-1, 1)

        return self.shap.Explainer(model_wrapper, self.background_data)

    def explain_instance(self, instance: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Explain instance using SHAP.

        Args:
            instance: Input instance

        Returns:
            SHAP explanation or None if SHAP not available
        """
        if self.explainer is None:
            logger.warning("SHAP explainer not available")
            return None

        shap_values = self.explainer(instance.reshape(1, -1))

        return {
            "method": "shap",
            "shap_values": shap_values.values[0],
            "base_value": shap_values.base_values[0],
            "instance": instance,
        }


class LIMEExplainer:
    """LIME-based explainer (requires lime library)."""

    def __init__(self, model: nn.Module, feature_names: List[str]):
        """
        Initialize LIME explainer.

        Args:
            model: Trained PyTorch model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.device = next(model.parameters()).device

        try:
            from lime.lime_tabular import LimeTabularExplainer

            self.lime_tabular = LimeTabularExplainer
        except ImportError:
            logger.warning("LIME library not available. Install with: pip install lime")
            self.lime_tabular = None

    def explain_instance(
        self, instance: np.ndarray, training_data: np.ndarray, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Explain instance using LIME.

        Args:
            instance: Input instance
            training_data: Training data for LIME
            **kwargs: Additional LIME parameters

        Returns:
            LIME explanation or None if LIME not available
        """
        if self.lime_tabular is None:
            logger.warning("LIME explainer not available")
            return None

        def model_wrapper(x):
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                output = self.model(x_tensor)
                if output.dim() > 1:
                    return torch.softmax(output, dim=1).cpu().numpy()
                else:
                    # Convert to binary classification probabilities
                    probs = torch.sigmoid(output).cpu().numpy()
                    return np.column_stack([1 - probs, probs])

        explainer = self.lime_tabular(
            training_data, feature_names=self.feature_names, **kwargs
        )

        explanation = explainer.explain_instance(
            instance, model_wrapper, num_features=len(self.feature_names)
        )

        return {"method": "lime", "explanation": explanation, "instance": instance}


def visualize_feature_importance(
    importance_scores: np.ndarray,
    feature_names: List[str],
    title: str = "Feature Importance",
    top_k: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize feature importance scores.

    Args:
        importance_scores: Array of importance scores
        feature_names: List of feature names
        title: Plot title
        top_k: Number of top features to show
        save_path: Path to save the plot
    """
    # Get top k features
    top_indices = np.argsort(np.abs(importance_scores))[-top_k:]
    top_scores = importance_scores[top_indices]
    top_names = [feature_names[i] for i in top_indices]

    # Create plot
    plt.figure(figsize=(10, 8))
    colors = ["red" if score < 0 else "blue" for score in top_scores]

    plt.barh(range(len(top_scores)), top_scores, color=colors, alpha=0.7)
    plt.yticks(range(len(top_scores)), top_names)
    plt.xlabel("Importance Score")
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, score in enumerate(top_scores):
        plt.text(
            score + 0.01 * np.sign(score), i, f"{score:.3f}", va="center", fontsize=9
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_interaction_matrix(
    interaction_matrix: np.ndarray,
    feature_names: List[str],
    title: str = "Feature Interactions",
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize feature interaction matrix.

    Args:
        interaction_matrix: Feature interaction matrix
        feature_names: List of feature names
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))

    # Create heatmap
    if SEABORN_AVAILABLE:
        sns.heatmap(
            interaction_matrix,
            xticklabels=feature_names,
            yticklabels=feature_names,
            annot=True,
            cmap="RdBu_r",
            center=0,
            fmt=".3f",
        )
    else:
        # Fallback to matplotlib imshow
        im = plt.imshow(interaction_matrix, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im)
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha="right")
        plt.yticks(range(len(feature_names)), feature_names, rotation=0)

    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if TORCH_AVAILABLE:

    class DummyModel(nn.Module):
        """Dummy model for demonstration purposes."""

        def __init__(self, input_size: int = 10, num_classes: int = 3):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes),
            )

        def forward(self, x):
            return self.layers(x)
else:

    class DummyModel:
        """Dummy model for demonstration purposes (no torch)."""

        def __init__(self, input_size: int = 10, num_classes: int = 3):
            self.input_size = input_size
            self.num_classes = num_classes

        def forward(self, x):
            return None

        def eval(self):
            return self

        def parameters(self):
            return []


def demo_explainability():
    """Demonstrate explainability functionality."""
    if RICH_AVAILABLE:
        console = Console()
        console.print(Panel.fit("🔍 Explainable AI Demo", style="bold blue"))
    else:
        print("🔍 Explainable AI Demo")

    if not TORCH_AVAILABLE:
        print("⚠️ PyTorch not available - running limited demo")
        print("✅ Explainability module structure is available")
        print("✅ Visualization functions are available")
        print("✅ All explainer classes are defined")
        return

    # Create dummy model and data
    print("🧠 Creating dummy model and data...")

    input_size = 10
    num_classes = 3
    n_samples = 100

    model = DummyModel(input_size, num_classes)
    model.eval()

    # Generate sample data
    np.random.seed(42)
    X_sample = np.random.randn(n_samples, input_size)
    feature_names = [f"feature_{i}" for i in range(input_size)]

    print(f"✅ Created model with {input_size} features and {num_classes} classes")
    print(f"✅ Generated {n_samples} sample instances")

    # Initialize explainer
    explainer = ModelExplainer(model, feature_names)

    # Test different explanation methods
    print("\n🔬 Testing explanation methods...")

    test_instance = X_sample[0]

    if RICH_AVAILABLE:
        methods_table = Table(title="Explanation Methods")
        methods_table.add_column("Method", style="cyan")
        methods_table.add_column("Status", style="green")
        methods_table.add_column("Top Feature", style="yellow")
        methods_table.add_column("Score", style="magenta")

    explanation_methods = ["integrated_gradients", "gradient_shap", "feature_ablation"]

    test_list = (
        track(explanation_methods, description="Testing methods...")
        if RICH_AVAILABLE
        else explanation_methods
    )

    for method in test_list:
        try:
            start_time = time.time()

            if method == "gradient_shap":
                explanation = explainer.explain_instance(
                    test_instance,
                    method=method,
                    background_samples=X_sample[:5],  # Use small sample for demo
                )
            else:
                explanation = explainer.explain_instance(test_instance, method=method)

            end_time = time.time()

            # Get top feature
            attributions = explanation["attributions"]
            top_idx = np.argmax(np.abs(attributions))
            top_feature = feature_names[top_idx]
            top_score = attributions[top_idx]

            if RICH_AVAILABLE:
                methods_table.add_row(
                    method.replace("_", " ").title(),
                    "✅ Success",
                    top_feature,
                    f"{top_score:.4f}",
                )
            else:
                print(
                    f"  ✅ {method.replace('_', ' ').title()}: {top_feature} = {top_score:.4f}"
                )

        except Exception as e:
            if RICH_AVAILABLE:
                methods_table.add_row(
                    method.replace("_", " ").title(),
                    f"❌ Error: {str(e)[:20]}...",
                    "N/A",
                    "N/A",
                )
            else:
                print(
                    f"  ❌ {method.replace('_', ' ').title()}: Error - {str(e)[:50]}..."
                )

    if RICH_AVAILABLE:
        console.print(methods_table)
        console.print("\n✨ Explainability demo completed!")
    else:
        print("\n✨ Explainability demo completed!")


# Only add click decorator if click is available
if CLICK_AVAILABLE:

    @click.command()
    @click.option("--demo", is_flag=True, help="Run explainability demonstration")
    @click.option(
        "--method",
        type=click.Choice(
            ["integrated_gradients", "gradient_shap", "feature_ablation"]
        ),
        help="Test specific explanation method",
    )
    def main(demo, method):
        """
        Test and demonstrate explainable AI functionality.
        """
        if RICH_AVAILABLE:
            console = Console()

        if demo:
            demo_explainability()
        elif method:
            if RICH_AVAILABLE:
                console.print(
                    Panel.fit(
                        f"🔬 Testing {method.replace('_', ' ').title()}",
                        style="bold blue",
                    )
                )
            else:
                print(f"🔬 Testing {method.replace('_', ' ').title()}")

            if not TORCH_AVAILABLE:
                print("⚠️ PyTorch not available - cannot run method test")
                return

            model = DummyModel(8, 3)
            feature_names = [f"feature_{i}" for i in range(8)]
            explainer = ModelExplainer(model, feature_names)

            np.random.seed(42)
            test_instance = np.random.randn(8)

            try:
                start_time = time.time()
                explanation = explainer.explain_instance(test_instance, method=method)
                end_time = time.time()

                print(f"✅ Method completed in {(end_time - start_time) * 1000:.2f} ms")

                attributions = explanation["attributions"]
                for feature, score in zip(feature_names, attributions):
                    print(f"  {feature}: {score:.6f}")

            except Exception as e:
                print(f"❌ Error testing {method}: {e}")
        else:
            print("Use --demo to run demonstration or --method to test specific method")
else:

    def main():
        """
        Test and demonstrate explainable AI functionality.
        """
        print("Explainable AI Module")
        print("Available functions:")
        print("- demo_explainability(): Run explainability demo")
        print("- ModelExplainer: Main explainer class")
        print(
            "- Available methods: integrated_gradients, gradient_shap, feature_ablation"
        )

        # Run demo by default when click is not available
        demo_explainability()


