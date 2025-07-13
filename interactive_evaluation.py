#!/usr/bin/env python3
"""
MoMLNIDS Interactive Model Selection Interface

This script provides an interactive interface to browse and select trained models
for evaluation without needing to specify full model paths.
"""

import os
import glob
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
import subprocess
import sys


class ModelSelector:
    """Interactive model selector for MoMLNIDS evaluation."""

    def __init__(self):
        self.console = Console()
        self.model_directories = [
            "ProperTraining",
            "ProperTraining50Epoch",
            "Training_results",
        ]
        self.available_models = []
        self.discover_models()

    def discover_models(self):
        """Discover all available trained models."""
        self.console.print("🔍 Discovering available models...")

        for base_dir in self.model_directories:
            if not os.path.exists(base_dir):
                continue

            # Find all .pt files in subdirectories
            pattern = os.path.join(base_dir, "**", "*.pt")
            model_files = glob.glob(pattern, recursive=True)

            for model_file in model_files:
                # Extract meaningful information
                rel_path = os.path.relpath(model_file)
                path_parts = Path(rel_path).parts

                if len(path_parts) >= 3:
                    training_type = path_parts[
                        0
                    ]  # ProperTraining, ProperTraining50Epoch, or Training_results

                    # Handle different directory structures
                    if training_type == "Training_results":
                        # For Training_results: Training_results/NF-UNSW-NB15-v2_NSingleLayer|1-N/model_best.pt
                        full_experiment = path_parts[
                            1
                        ]  # NF-UNSW-NB15-v2_NSingleLayer|1-N
                        model_name = path_parts[-1]  # Model file name

                        # Extract dataset name (before the first underscore + method)
                        if "_N" in full_experiment:
                            dataset = full_experiment.split("_N")[0]  # NF-UNSW-NB15-v2
                            experiment = full_experiment.split("_N")[
                                1
                            ]  # SingleLayer|1-N
                        else:
                            dataset = full_experiment
                            experiment = "Unknown"
                    else:
                        # For ProperTraining: ProperTraining/NF-CSE-CIC-IDS2018-v2/NF-CSE-CIC-IDS2018-v2_N|PseudoLabelling|Cluster_2/model_best.pt
                        dataset = path_parts[1]  # Target dataset
                        experiment = path_parts[2]  # Experiment configuration
                        model_name = path_parts[-1]  # Model file name

                    # Extract epochs info from training type
                    epochs = "50" if "50Epoch" in training_type else "20"

                    # For Training_results, try to extract epochs from experiment name or default to 20
                    if training_type == "Training_results":
                        epochs = "20"  # Default for Training_results models

                    # Parse experiment configuration for method and clusters
                    method_info = self._parse_experiment_config(experiment)

                    self.available_models.append(
                        {
                            "path": rel_path,
                            "training_type": training_type,
                            "target_dataset": dataset,
                            "experiment": experiment,
                            "model_name": model_name,
                            "epochs": epochs,
                            "size_mb": round(
                                os.path.getsize(model_file) / (1024 * 1024), 1
                            ),
                            "method": method_info["method"],
                            "clusters": method_info["clusters"],
                            "is_pseudo_label": method_info["is_pseudo_label"],
                            "method_display": method_info["method_display"],
                        }
                    )

        # Sort by target dataset, method, and epochs
        self.available_models.sort(
            key=lambda x: (
                x["target_dataset"],
                x["method"],
                int(x["epochs"]),
                x["model_name"],
            )
        )

        self.console.print(f"✅ Found {len(self.available_models)} trained models")

    def _filter_best_and_last_models(self, models):
        """Filter models to show only best and last for each configuration."""
        from collections import defaultdict
        import re

        # Group models by (dataset, method, epochs, clusters)
        config_groups = defaultdict(list)

        for model in models:
            key = (
                model["target_dataset"],
                model["method"],
                model["epochs"],
                model["clusters"],
            )
            config_groups[key].append(model)

        filtered_models = []

        for config_key, config_models in config_groups.items():
            if not config_models:
                continue

            # Separate best models from numbered models
            best_models = [
                m for m in config_models if "best" in m["model_name"].lower()
            ]
            numbered_models = [
                m for m in config_models if "best" not in m["model_name"].lower()
            ]

            # Add best model if exists
            if best_models:
                # Sort by file size (larger might be better) and take the first one
                best_model = sorted(
                    best_models, key=lambda x: x["size_mb"], reverse=True
                )[0]
                filtered_models.append(best_model)

            # Add last numbered model if exists
            if numbered_models:
                # Extract numbers from model names and find the highest
                def extract_number(model_name):
                    numbers = re.findall(r"model_(\d+)\.pt", model_name)
                    return int(numbers[0]) if numbers else -1

                last_model = max(
                    numbered_models, key=lambda x: extract_number(x["model_name"])
                )
                filtered_models.append(last_model)

        # Sort filtered models by dataset, method, epochs, clusters
        filtered_models.sort(
            key=lambda x: (
                x["target_dataset"],
                x["method"],
                int(x["epochs"]),
                x["clusters"] or 0,
            )
        )

        return filtered_models

    def _parse_experiment_config(self, experiment_config: str) -> dict:
        """Parse experiment configuration to extract method and cluster information."""
        import re

        config_lower = experiment_config.lower()

        # Check if it's a pseudo-labeling method
        is_pseudo_label = "pseudolabel" in config_lower or "pseudo" in config_lower

        # Extract method name
        if "pseudolabel" in config_lower or "pseudolabelling" in config_lower:
            method = "PseudoLabelling"
        elif "singlelayer" in config_lower:
            method = "SingleLayer"
        elif "baseline" in config_lower:
            method = "Baseline"
        elif "dann" in config_lower:
            method = "DANN"
        elif "coral" in config_lower:
            method = "CORAL"
        else:
            # Try to extract method from the beginning of the config
            parts = experiment_config.split("|")
            if len(parts) > 0 and parts[0].strip():
                method = parts[0].strip()
            else:
                # Fallback to splitting by underscore
                parts = experiment_config.split("_")
                if parts:
                    method = parts[0]
                else:
                    method = "Unknown"

        # Extract cluster count for pseudo-labeling methods
        clusters = None
        if is_pseudo_label:
            # Look for cluster patterns like "Cluster_4", "4", or similar
            cluster_patterns = [
                r"cluster[_\-]?(\d+)",  # Cluster_4, cluster-4, cluster4
                r"(\d+)[_\-]?cluster",  # 4_cluster, 4-cluster, 4cluster
                r"k[_\-]?(\d+)",  # k_4, k-4, k4
                r"(\d+)[_\-]?k",  # 4_k, 4-k, 4k
            ]

            for pattern in cluster_patterns:
                match = re.search(pattern, config_lower)
                if match:
                    try:
                        clusters = int(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue

            # If no clusters found but it's pseudo-labeling, look for any number
            if clusters is None:
                numbers = re.findall(r"\d+", experiment_config)
                # Filter out common non-cluster numbers
                potential_clusters = [int(n) for n in numbers if 2 <= int(n) <= 20]
                if potential_clusters:
                    clusters = potential_clusters[0]  # Take the first reasonable number

        # Create display name
        if is_pseudo_label and clusters:
            method_display = f"{method} (K={clusters})"
        elif is_pseudo_label:
            method_display = f"{method} (Clustering)"
        else:
            method_display = method

        return {
            "method": method,
            "clusters": clusters,
            "is_pseudo_label": is_pseudo_label,
            "method_display": method_display,
        }

    def display_models(
        self, filter_dataset=None, filter_epochs=None, filter_method=None
    ):
        """Display available models in a formatted table grouped by dataset."""

        # Filter models if requested
        models_to_show = self.available_models
        if filter_dataset:
            models_to_show = [
                m
                for m in models_to_show
                if filter_dataset.lower() in m["target_dataset"].lower()
            ]
        if filter_epochs:
            models_to_show = [
                m for m in models_to_show if m["epochs"] == str(filter_epochs)
            ]
        if filter_method:
            models_to_show = [
                m
                for m in models_to_show
                if filter_method.lower() in m["method"].lower()
            ]

        if not models_to_show:
            self.console.print("❌ No models found matching the criteria")
            return []

        # Filter to show only best and last models for each configuration
        models_to_show = self._filter_best_and_last_models(models_to_show)

        # Group models by dataset
        datasets_grouped = {}
        for model in models_to_show:
            dataset = model["target_dataset"]
            if dataset not in datasets_grouped:
                datasets_grouped[dataset] = []
            datasets_grouped[dataset].append(model)

        # Display each dataset group
        model_counter = 1
        all_models = []

        for dataset_name in sorted(datasets_grouped.keys()):
            dataset_models = datasets_grouped[dataset_name]

            # Create table for this dataset
            table = Table(
                title=f"🎯 {dataset_name} Models (Best & Last only - {len(dataset_models)} shown)"
            )
            table.add_column("ID", style="cyan", width=3)
            table.add_column("Method", style="blue", width=18)
            table.add_column("Epochs", style="yellow", width=6)
            table.add_column("Clusters", style="magenta", width=8)
            table.add_column("Model Type", style="green", width=12)
            table.add_column("File", style="white", width=12)
            table.add_column("Size", style="white", width=8)

            for model in dataset_models:
                # Format clusters display
                clusters_display = ""
                if model["is_pseudo_label"]:
                    if model["clusters"]:
                        clusters_display = f"K={model['clusters']}"
                    else:
                        clusters_display = "Yes"
                else:
                    clusters_display = "N/A"

                # Determine model type
                if "best" in model["model_name"].lower():
                    model_type = "🏆 Best"
                    file_display = "model_best.pt"
                else:
                    # Extract model number
                    import re

                    numbers = re.findall(r"model_(\d+)\.pt", model["model_name"])
                    if numbers:
                        model_type = f"📊 Final"
                        file_display = f"model_{numbers[0]}.pt"
                    else:
                        model_type = "📄 Other"
                        file_display = model["model_name"]

                # Truncate long method names
                method_name = model["method_display"]
                if len(method_name) > 17:
                    method_name = method_name[:14] + "..."

                table.add_row(
                    str(model_counter),
                    method_name,
                    model["epochs"],
                    clusters_display,
                    model_type,
                    file_display,
                    f"{model['size_mb']} MB",
                )

                all_models.append(model)
                model_counter += 1

            self.console.print(table)
            self.console.print()  # Add spacing between tables

        return all_models

    def get_unique_datasets(self):
        """Get list of unique target datasets."""
        datasets = list(set(model["target_dataset"] for model in self.available_models))
        return sorted(datasets)

    def get_unique_epochs(self):
        """Get list of unique epoch counts."""
        epochs = list(set(model["epochs"] for model in self.available_models))
        return sorted(epochs)

    def get_unique_methods(self):
        """Get list of unique methods."""
        methods = list(set(model["method"] for model in self.available_models))
        return sorted(methods)

    def interactive_selection(self):
        """Interactive model selection process."""

        self.console.print(
            Panel.fit("🎯 MoMLNIDS Interactive Model Selector", style="bold blue")
        )

        while True:
            # Show available filters
            datasets = self.get_unique_datasets()
            epochs = self.get_unique_epochs()
            methods = self.get_unique_methods()

            self.console.print(f"\n📊 Available filters:")
            self.console.print(f"   Datasets: {', '.join(datasets)}")
            self.console.print(f"   Epochs: {', '.join(epochs)}")
            self.console.print(f"   Methods: {', '.join(methods)}")

            # Get filter preferences
            filter_dataset = None
            filter_epochs = None
            filter_method = None

            if len(datasets) > 1:
                filter_choice = Prompt.ask(
                    "\n🔍 Filter by dataset? (press Enter for all)",
                    choices=datasets + [""],
                    default="",
                    show_choices=False,
                )
                if filter_choice:
                    filter_dataset = filter_choice

            if len(epochs) > 1:
                filter_choice = Prompt.ask(
                    "🔍 Filter by epochs? (press Enter for all)",
                    choices=epochs + [""],
                    default="",
                    show_choices=False,
                )
                if filter_choice:
                    filter_epochs = filter_choice

            if len(methods) > 1:
                filter_choice = Prompt.ask(
                    "🔍 Filter by method? (press Enter for all)",
                    choices=methods + [""],
                    default="",
                    show_choices=False,
                )
                if filter_choice:
                    filter_method = filter_choice

            # Display filtered models
            self.console.print()
            filtered_models = self.display_models(
                filter_dataset, filter_epochs, filter_method
            )

            if not filtered_models:
                if Confirm.ask("Try different filters?"):
                    continue
                else:
                    return None

            # Get user selection
            while True:
                try:
                    choice = Prompt.ask(
                        f"\n🎯 Select model (1-{len(filtered_models)}) or 'f' to change filters",
                        default="1",
                    )

                    if choice.lower() == "f":
                        break  # Go back to filter selection

                    model_index = int(choice) - 1
                    if 0 <= model_index < len(filtered_models):
                        selected_model = filtered_models[model_index]

                        # Display selection details
                        self.display_model_details(selected_model)

                        if Confirm.ask("Use this model?", default=True):
                            return selected_model
                        else:
                            continue
                    else:
                        self.console.print(
                            f"❌ Please enter a number between 1 and {len(filtered_models)}"
                        )

                except ValueError:
                    self.console.print("❌ Please enter a valid number or 'f'")
                except KeyboardInterrupt:
                    self.console.print("\n👋 Goodbye!")
                    return None

    def display_model_details(self, model):
        """Display detailed information about selected model."""

        details_table = Table(title=f"📋 Model Details: {model['model_name']}")
        details_table.add_column("Property", style="cyan")
        details_table.add_column("Value", style="green")

        details_table.add_row("Full Path", model["path"])
        details_table.add_row("Target Dataset", model["target_dataset"])
        details_table.add_row("Method", model["method_display"])
        details_table.add_row("Training Epochs", model["epochs"])
        details_table.add_row(
            "Pseudo-Labeling", "Yes" if model["is_pseudo_label"] else "No"
        )
        if model["clusters"]:
            details_table.add_row("Cluster Count", str(model["clusters"]))
        details_table.add_row("Experiment Config", model["experiment"])
        details_table.add_row("File Size", f"{model['size_mb']} MB")
        details_table.add_row("Training Type", model["training_type"])

        self.console.print(details_table)


@click.command()
@click.option(
    "--evaluation-type",
    "-t",
    type=click.Choice(["single", "comprehensive", "all-datasets"]),
    default="comprehensive",
    help="Type of evaluation to run",
)
@click.option(
    "--num-samples",
    "-n",
    help="Number of samples per dataset (e.g., 10000, 50000, 100000)",
)
@click.option(
    "--percentage",
    "-p",
    type=click.Choice(["20", "40", "60", "80", "100"]),
    help="Percentage of dataset to use (20%, 40%, 60%, 80%, 100%)",
)
@click.option("--export-json", help="Export results to JSON file")
@click.option("--show-samples", is_flag=True, help="Show individual sample predictions")
@click.option(
    "--auto-select",
    help="Auto-select model by dataset name (e.g., 'NF-CSE-CIC-IDS2018-v2')",
)
def main(
    evaluation_type, num_samples, percentage, export_json, show_samples, auto_select
):
    """
    Interactive MoMLNIDS Model Evaluation Interface

    Browse and select trained models interactively, then run evaluations
    without needing to remember full model paths.

    Evaluation types:
    - single: Test on one specific dataset
    - comprehensive: Full domain generalization analysis across ALL datasets
    - all-datasets: Basic summary across all datasets
    """
    console = Console()

    try:
        # Initialize model selector
        selector = ModelSelector()

        if not selector.available_models:
            console.print("❌ No trained models found in ProperTraining directories")
            console.print(
                "   Make sure you have models in ProperTraining/ or ProperTraining50Epoch/"
            )
            return

        # Select model
        if auto_select:
            # Auto-select by dataset name
            matching_models = [
                m
                for m in selector.available_models
                if auto_select.lower() in m["target_dataset"].lower()
            ]
            if matching_models:
                # Prefer model_best.pt, then by training type
                selected_model = None
                for model in matching_models:
                    if "best" in model["model_name"].lower():
                        selected_model = model
                        break
                if not selected_model:
                    selected_model = matching_models[0]

                console.print(f"🎯 Auto-selected model: {selected_model['path']}")
                selector.display_model_details(selected_model)
            else:
                console.print(f"❌ No models found for dataset: {auto_select}")
                return
        else:
            # Interactive selection
            selected_model = selector.interactive_selection()
            if not selected_model:
                console.print("👋 No model selected. Exiting.")
                return

        model_path = selected_model["path"]
        target_dataset = selected_model["target_dataset"]

        # Handle sample count configuration
        if not num_samples and not percentage:
            # Interactive sample count selection
            sample_choice = Prompt.ask(
                "\n🔢 Choose sample count method",
                choices=["fixed", "percentage"],
                default="fixed",
            )

            if sample_choice == "fixed":
                num_samples = Prompt.ask(
                    "Number of samples per dataset",
                    choices=["10000", "25000", "50000", "100000", "150000"],
                    default="50000",
                )
                num_samples = int(num_samples)
                console.print(f"🔢 Using {num_samples:,} samples (fixed count mode)")
            else:
                percentage = Prompt.ask(
                    "Percentage of dataset to use",
                    choices=["20", "40", "60", "80", "100"],
                    default="100",
                )
                percentage_float = float(percentage) / 100.0
                console.print(f"🔢 Using {percentage}% of dataset (percentage mode)")
        elif num_samples:
            num_samples = int(num_samples)
            console.print(f"🔢 Using {num_samples:,} samples (fixed count mode)")
        else:
            # Convert percentage to samples if provided
            percentage_float = float(percentage) / 100.0
            console.print(f"🔢 Using {percentage}% of dataset (percentage mode)")

        # Prepare command based on evaluation type
        if evaluation_type == "comprehensive":
            console.print(
                f"\n🌐 Running comprehensive domain generalization evaluation..."
            )

            # Use comprehensive_evaluation.py
            cmd = [
                sys.executable,
                "comprehensive_evaluation.py",
                "--model-path",
                model_path,
            ]

            # Add sampling parameters
            if "percentage_float" in locals():
                cmd.extend(["--percentage", str(percentage_float)])
            else:
                cmd.extend(["--num-samples", str(num_samples)])

            if export_json:
                cmd.extend(["--export-json", export_json])
            else:
                # Auto-generate filename
                safe_name = target_dataset.replace("-", "_")
                auto_json = f"results/comprehensive_{safe_name}_{selected_model['epochs']}ep.json"
                cmd.extend(["--export-json", auto_json])

        elif evaluation_type == "single":
            console.print(
                f"\n🎯 Running single dataset evaluation on {target_dataset}..."
            )

            # Use auto_prediction_demo.py for single dataset
            cmd = [
                sys.executable,
                "auto_prediction_demo.py",
                "--model-path",
                model_path,
                "--dataset",
                target_dataset,
            ]

            # Add sampling parameters
            if "percentage_float" in locals():
                cmd.extend(["--percentage", str(percentage_float)])
            else:
                cmd.extend(["--num-samples", str(num_samples)])

            if show_samples:
                cmd.append("--show-samples")

            if export_json:
                cmd.extend(["--export-json", export_json])

        elif evaluation_type == "all-datasets":
            console.print(f"\n📊 Running evaluation on all datasets...")

            # Use auto_prediction_demo.py with all-datasets flag
            cmd = [
                sys.executable,
                "auto_prediction_demo.py",
                "--model-path",
                model_path,
                "--all-datasets",
            ]

            # Add sampling parameters
            if "percentage_float" in locals():
                cmd.extend(["--percentage", str(percentage_float)])
            else:
                cmd.extend(["--num-samples", str(num_samples)])

            if show_samples:
                cmd.append("--show-samples")

        # Display command that will be executed
        console.print(f"\n🚀 Executing: {' '.join(cmd)}")
        console.print()

        # Run the evaluation
        result = subprocess.run(cmd, check=False)

        if result.returncode == 0:
            console.print(f"\n✅ Evaluation completed successfully!")
        else:
            console.print(
                f"\n❌ Evaluation failed with return code: {result.returncode}"
            )

    except KeyboardInterrupt:
        console.print("\n👋 Evaluation cancelled by user")
    except Exception as e:
        console.print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
