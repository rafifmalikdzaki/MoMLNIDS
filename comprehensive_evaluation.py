#!/usr/bin/env python3
"""
MoMLNIDS Comprehensive Domain Generalization Evaluation

This script runs the automatic prediction demo on ALL datasets to evaluate
domain generalization performance across different network domains.
"""

import subprocess
import json
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import numpy as np


@click.command()
@click.option(
    "--model-path", "-m", required=True, help="Path to trained model (.pt file)"
)
@click.option(
    "--num-samples", "-n", default=100000, help="Number of samples per dataset"
)
@click.option("--export-json", help="Export comprehensive results to JSON file")
def main(model_path, num_samples, export_json):
    """
    Run comprehensive domain generalization evaluation across ALL datasets.

    This evaluates the MoMLNIDS model's ability to generalize across different
    network intrusion detection domains.
    """
    console = Console()

    console.print(
        Panel.fit(
            "🌐 MoMLNIDS Comprehensive Domain Generalization Evaluation",
            style="bold blue",
        )
    )

    # Extract target domain from model path
    model_target_domain = None
    # Define domain relationships based on MoMLNIDS framework
    # BoT-IoT is considered source domain, others are target domains
    source_domain = "NF-BoT-IoT-v2"

    available_datasets = [
        "NF-CSE-CIC-IDS2018-v2",
        "NF-ToN-IoT-v2",
        "NF-UNSW-NB15-v2",
        "NF-BoT-IoT-v2",  # Source domain
    ]

    for dataset_name in available_datasets:
        if dataset_name in model_path:
            model_target_domain = dataset_name
            break

    if model_target_domain:
        console.print(f"🎯 Model trained for target domain: {model_target_domain}")
        console.print(f"📊 Evaluating domain generalization across all datasets")
    else:
        console.print(f"🔍 Evaluating model across all available datasets")

    all_results = {}

    # Run evaluation on each dataset
    for i, dataset_name in enumerate(available_datasets, 1):
        console.print(f"\n{'=' * 80}")
        console.print(f"📊 Dataset {i}/{len(available_datasets)}: {dataset_name}")

        # Identify domain type based on MoMLNIDS framework
        if dataset_name == model_target_domain:
            eval_type = "🎯 TARGET DOMAIN (Model trained on this)"
        elif dataset_name == source_domain:
            eval_type = "📦 SOURCE DOMAIN (BoT-IoT source data)"
        else:
            eval_type = "🔄 TARGET DOMAIN (Cross-domain generalization)"

        console.print(f"   {eval_type}")
        console.print(f"{'=' * 80}")

        try:
            # Create output file for this dataset
            output_file = f"results/eval_{dataset_name}.json"

            # Run the auto prediction demo for this dataset
            cmd = [
                "python",
                "auto_prediction_demo.py",
                "--model-path",
                model_path,
                "--dataset",
                dataset_name,
                "--num-samples",
                str(num_samples),
                "--export-json",
                output_file,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Load the results
                if Path(output_file).exists():
                    with open(output_file, "r") as f:
                        dataset_result = json.load(f)

                    # Extract key metrics
                    dataset_metrics = dataset_result["evaluation_results"][
                        dataset_name
                    ]["metrics"]

                    all_results[dataset_name] = {
                        "evaluation_type": "trained_target"
                        if dataset_name == model_target_domain
                        else "source"
                        if dataset_name == source_domain
                        else "target",
                        "accuracy": dataset_metrics[f"{dataset_name}_accuracy"],
                        "precision": dataset_metrics[f"{dataset_name}_precision"],
                        "recall": dataset_metrics[f"{dataset_name}_recall"],
                        "f1_score": dataset_metrics[f"{dataset_name}_f1_score"],
                        "auc_roc": dataset_metrics[f"{dataset_name}_auc_roc"],
                        "samples": dataset_result["config"]["num_samples_evaluated"],
                    }

                    console.print(f"✅ {dataset_name} evaluation completed!")
                    console.print(
                        f"   Accuracy: {all_results[dataset_name]['accuracy']:.4f}"
                    )
                    console.print(
                        f"   F1-Score: {all_results[dataset_name]['f1_score']:.4f}"
                    )
                else:
                    console.print(f"❌ Output file not created for {dataset_name}")
                    all_results[dataset_name] = {"error": "No output file created"}
            else:
                console.print(f"❌ Error running evaluation for {dataset_name}")
                console.print(f"   Error: {result.stderr}")
                all_results[dataset_name] = {"error": result.stderr}

        except Exception as e:
            console.print(f"❌ Exception during {dataset_name} evaluation: {e}")
            all_results[dataset_name] = {"error": str(e)}

    # Display comprehensive summary
    display_comprehensive_summary(console, all_results, model_target_domain)

    # Export comprehensive results if requested
    if export_json:
        export_comprehensive_results(
            all_results, model_target_domain, model_path, export_json
        )
        console.print(f"📊 Comprehensive results exported to: {export_json}")

    console.print(f"\n✨ Comprehensive evaluation completed!")


def display_comprehensive_summary(console, all_results, model_target_domain):
    """Display comprehensive summary across all datasets."""

    console.print(f"\n{'=' * 100}")
    console.print("📊 COMPREHENSIVE DOMAIN GENERALIZATION EVALUATION SUMMARY")
    console.print(f"{'=' * 100}")

    # Performance Summary Table
    summary_table = Table(title="🎯 Performance Across All Datasets")
    summary_table.add_column("Dataset", style="cyan")
    summary_table.add_column("Type", style="blue")
    summary_table.add_column("Accuracy", style="green")
    summary_table.add_column("F1-Score", style="yellow")
    summary_table.add_column("AUC-ROC", style="magenta")
    summary_table.add_column("Samples", style="white")

    target_performance = None
    cross_domain_performances = []

    for dataset_name, result in all_results.items():
        if "error" in result:
            summary_table.add_row(dataset_name, "❌ Error", "-", "-", "-", "-")
            continue

        eval_type = (
            "🎯 Target"
            if dataset_name == model_target_domain
            else "📦 Source"
            if dataset_name == "NF-BoT-IoT-v2"
            else "🔄 Target"
        )

        summary_table.add_row(
            dataset_name,
            eval_type,
            f"{result['accuracy']:.4f}",
            f"{result['f1_score']:.4f}",
            f"{result['auc_roc']:.4f}",
            str(result["samples"]),
        )

        # Track performance for analysis (only trained target vs other targets, excluding source)
        if dataset_name == model_target_domain:
            target_performance = result["accuracy"]
        elif (
            dataset_name != "NF-BoT-IoT-v2"
        ):  # Exclude source domain from cross-domain analysis
            cross_domain_performances.append(result["accuracy"])

    console.print(summary_table)

    # Domain Generalization Analysis
    if target_performance is not None and cross_domain_performances:
        avg_cross_domain = np.mean(cross_domain_performances)
        performance_drop = target_performance - avg_cross_domain

        analysis_table = Table(title="🔬 Domain Generalization Analysis")
        analysis_table.add_column("Metric", style="cyan")
        analysis_table.add_column("Value", style="green")
        analysis_table.add_column("Interpretation", style="yellow")

        analysis_table.add_row(
            "Target Domain Performance",
            f"{target_performance:.4f}",
            "Baseline performance",
        )
        analysis_table.add_row(
            "Avg Other Target Performance",
            f"{avg_cross_domain:.4f}",
            "Generalization to other targets",
        )
        analysis_table.add_row(
            "Performance Drop",
            f"{performance_drop:.4f}",
            "Target vs other targets difference",
        )

        # Generalization quality assessment
        if performance_drop < 0.1:
            quality = "🌟 Excellent"
        elif performance_drop < 0.2:
            quality = "✅ Good"
        elif performance_drop < 0.3:
            quality = "⚠️ Moderate"
        else:
            quality = "❌ Poor"

        analysis_table.add_row(
            "Generalization Quality", quality, "Cross-target adaptation effectiveness"
        )

        console.print(analysis_table)

    # Best and worst performing datasets
    valid_results = {k: v for k, v in all_results.items() if "error" not in v}
    if len(valid_results) > 1:
        best_dataset = max(valid_results.items(), key=lambda x: x[1]["accuracy"])
        worst_dataset = min(valid_results.items(), key=lambda x: x[1]["accuracy"])

        comparison_table = Table(title="📈 Best vs Worst Performance")
        comparison_table.add_column("Metric", style="cyan")
        comparison_table.add_column("Best (" + best_dataset[0] + ")", style="green")
        comparison_table.add_column("Worst (" + worst_dataset[0] + ")", style="red")

        comparison_table.add_row(
            "Accuracy",
            f"{best_dataset[1]['accuracy']:.4f}",
            f"{worst_dataset[1]['accuracy']:.4f}",
        )
        comparison_table.add_row(
            "F1-Score",
            f"{best_dataset[1]['f1_score']:.4f}",
            f"{worst_dataset[1]['f1_score']:.4f}",
        )
        comparison_table.add_row(
            "AUC-ROC",
            f"{best_dataset[1]['auc_roc']:.4f}",
            f"{worst_dataset[1]['auc_roc']:.4f}",
        )

        console.print(comparison_table)


def export_comprehensive_results(
    all_results, model_target_domain, model_path, output_path
):
    """Export comprehensive results across all datasets."""
    from datetime import datetime

    # Calculate summary statistics
    valid_results = {k: v for k, v in all_results.items() if "error" not in v}

    if valid_results:
        accuracies = [v["accuracy"] for v in valid_results.values()]
        f1_scores = [v["f1_score"] for v in valid_results.values()]
        auc_scores = [v["auc_roc"] for v in valid_results.values()]

        summary_stats = {
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "mean_f1_score": float(np.mean(f1_scores)),
            "std_f1_score": float(np.std(f1_scores)),
            "mean_auc_roc": float(np.mean(auc_scores)),
            "std_auc_roc": float(np.std(auc_scores)),
            "best_performing_dataset": max(
                valid_results.items(), key=lambda x: x[1]["accuracy"]
            )[0],
            "worst_performing_dataset": min(
                valid_results.items(), key=lambda x: x[1]["accuracy"]
            )[0],
        }
    else:
        summary_stats = {}

    comprehensive_results = {
        "evaluation_info": {
            "evaluation_type": "comprehensive_domain_generalization",
            "model_path": model_path,
            "model_target_domain": model_target_domain,
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_datasets_evaluated": len(all_results),
            "successful_evaluations": len(valid_results),
            "failed_evaluations": len(all_results) - len(valid_results),
        },
        "summary_statistics": summary_stats,
        "individual_dataset_results": all_results,
        "domain_generalization_analysis": {
            "note": "This evaluation tests the model's ability to generalize across different network domains",
            "target_domain": model_target_domain,
            "cross_domain_datasets": [
                k for k in all_results.keys() if k != model_target_domain
            ],
        },
    }

    with open(output_path, "w") as f:
        json.dump(comprehensive_results, f, indent=2)


if __name__ == "__main__":
    main()
