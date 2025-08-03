#!/usr/bin/env python3
"""
Demo script for the Enhanced Granular MoMLNIDS Results Visualizer
Shows the new features including dataset-specific breakdowns and cluster analysis.
"""

import sys
from pathlib import Path
import time

# Add the src directory to path if needed
sys.path.append(str(Path(__file__).parent / "src"))

from enhanced_results_visualizer import EnhancedMoMLNIDSAnalyzer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()


def demo_enhanced_features():
    """Demonstrate the enhanced features of the visualizer."""

    console.print(
        Rule("[bold blue]Enhanced MoMLNIDS Results Visualizer Demo[/bold blue]")
    )
    console.print()

    # Initialize analyzer
    analyzer = EnhancedMoMLNIDSAnalyzer(".")

    # Discover experiments
    console.print("[yellow]🔍 Discovering experiments...[/yellow]")
    analyzer.discover_experiments()

    if not analyzer.all_experiments:
        console.print("[red]❌ No experiments found in current directory![/red]")
        return

    console.print(
        f"[green]✅ Found {len(analyzer.all_experiments)} experiments[/green]"
    )
    console.print()

    # Show experiment tree with enhanced information
    console.print(Panel("🌳 Experiment Tree (Enhanced)", style="bold blue"))
    tree = analyzer.create_experiment_tree()
    console.print(tree)
    console.print()

    # Wait for user
    input("Press Enter to see dataset-specific performance breakdown...")
    console.clear()

    # Show dataset-specific performance breakdown
    console.print(
        Rule("[bold green]📊 Dataset-Specific Performance Breakdown[/bold green]")
    )
    console.print()

    panels = analyzer.create_dataset_performance_breakdown()
    for panel in panels:
        console.print(panel)
        console.print()

    # Wait for user
    input("Press Enter to see enhanced comparison table...")
    console.clear()

    # Show enhanced comparison table for a group
    console.print(Rule("[bold cyan]📋 Enhanced Comparison Table[/bold cyan]"))
    console.print()

    # Get experiments from first available group
    if analyzer.experiment_groups:
        first_group = list(analyzer.experiment_groups.keys())[0]
        experiments = [
            f"{first_group}::{exp}"
            for exp in analyzer.experiment_groups[first_group].keys()
        ]

        console.print(
            f"[bold green]Showing detailed comparison for: {first_group}[/bold green]"
        )
        console.print()

        table = analyzer.create_comparison_table(experiments)
        console.print(table)
        console.print()

    # Wait for user
    input("Press Enter to see best performers with cluster information...")
    console.clear()

    # Show best performers with enhanced details
    console.print(Rule("[bold gold1]🏆 Enhanced Best Performers Analysis[/bold gold1]"))
    console.print()

    # Collect performers data for demonstration
    performers = []
    for exp_name, exp_data in analyzer.all_experiments.items():
        if exp_data["type"] == "logs" and exp_data.get("target_performance"):
            perfs = exp_data["target_performance"]
            if perfs:
                best_perf = max(perfs, key=lambda x: x["accuracy"])
                config = exp_data.get("config", {})
                performers.append(
                    {
                        "name": exp_name,
                        "accuracy": best_perf["accuracy"],
                        "dataset": config.get("dataset_short", "Unknown"),
                        "method": config.get(
                            "full_method_description", config.get("method", "Unknown")
                        ),
                        "clusters": config.get("clusters"),
                    }
                )

    # Show top 5 with detailed information
    performers.sort(key=lambda x: x["accuracy"], reverse=True)

    console.print("[bold]🥇 Top 5 Performers with Cluster Information:[/bold]")
    for i, perf in enumerate(performers[:5], 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        clusters_info = f" (K={perf['clusters']})" if perf["clusters"] else ""

        console.print(
            f"{medal} [cyan]{perf['dataset']}[/cyan] - [green]{perf['method']}{clusters_info}[/green]"
        )
        console.print(f"    Accuracy: [bold green]{perf['accuracy']:.4f}[/bold green]")
        console.print()

    # Wait for user
    input("Press Enter to see method comparison with clustering analysis...")
    console.clear()

    # Show clustering analysis
    console.print(
        Rule("[bold purple]🔍 PseudoLabelling Clustering Analysis[/bold purple]")
    )
    console.print()

    # Collect clustering data
    from collections import defaultdict

    clustering_data = defaultdict(lambda: defaultdict(list))

    for exp_name, exp_data in analyzer.all_experiments.items():
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

    if clustering_data:
        console.print("[bold]Cluster Count Impact on Performance:[/bold]")
        for dataset, cluster_groups in clustering_data.items():
            console.print(f"\n[yellow]📊 {dataset}:[/yellow]")
            for clusters, accuracies in sorted(cluster_groups.items()):
                if accuracies:
                    avg_acc = sum(accuracies) / len(accuracies)
                    best_acc = max(accuracies)
                    console.print(
                        f"  K={clusters}: Best={best_acc:.4f}, Avg={avg_acc:.4f} ({len(accuracies)} experiments)"
                    )
    else:
        console.print(
            "[yellow]No PseudoLabelling experiments with cluster information found.[/yellow]"
        )

    console.print()

    # Show feature summary
    console.print(Rule("[bold blue]✨ Enhanced Features Summary[/bold blue]"))

    features = [
        "🎯 Detailed configuration parsing with cluster extraction",
        "📊 Dataset-specific performance breakdowns",
        "🔍 Method descriptions showing cluster counts (e.g., 'PseudoLabelling (K=5)')",
        "🏆 Enhanced best performers with precision, recall, and final loss",
        "📋 Granular comparison tables with cluster, precision, recall, and loss columns",
        "🔬 Advanced method comparison across datasets",
        "📈 Clustering analysis for PseudoLabelling methods",
        "📋 Detailed experiment analysis with stability metrics and loss tracking",
        "🏅 Dataset champions and method performance statistics",
        "❌ Removed architecture column (as requested)",
        "➕ Added precision, recall, and final loss columns",
    ]

    feature_panel = Panel(
        "\n".join(features),
        title="[bold green]New Enhanced Features[/bold green]",
        border_style="green",
    )
    console.print(feature_panel)
    console.print()

    console.print(
        "[bold green]🎉 Demo completed! Run 'python enhanced_results_visualizer.py' for interactive mode.[/bold green]"
    )


if __name__ == "__main__":
    demo_enhanced_features()
