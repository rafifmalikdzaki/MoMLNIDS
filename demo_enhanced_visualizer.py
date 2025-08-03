#!/usr/bin/env python3
"""
Demo script for the Enhanced MoMLNIDS Results Visualizer
This demonstrates the capabilities with your actual experiment data.
"""

import sys
from pathlib import Path
from enhanced_results_visualizer import EnhancedMoMLNIDSAnalyzer
from rich.console import Console
from rich.rule import Rule
from rich.panel import Panel

console = Console()


def demo_enhanced_visualizer():
    """Demonstrate the enhanced visualizer with actual data."""
    console.print(
        Rule("[bold blue]🚀 Enhanced MoMLNIDS Results Visualizer Demo[/bold blue]")
    )
    console.print()

    # Initialize analyzer
    analyzer = EnhancedMoMLNIDSAnalyzer(".")

    console.print(
        "[yellow]📂 Discovering experiments across all directories...[/yellow]"
    )
    experiments = analyzer.discover_experiments()

    if not experiments:
        console.print("[red]❌ No experiments found![/red]")
        return

    # Show discovery summary
    total_experiments = sum(len(exps) for exps in experiments.values())
    console.print(
        f"[green]✅ Found {total_experiments} experiments across {len(experiments)} directories[/green]"
    )
    console.print()

    # Show experiment tree
    console.print("[cyan]🌳 Experiment Directory Structure:[/cyan]")
    tree = analyzer.create_experiment_tree()
    console.print(tree)
    console.print()

    # Show comparison for each group
    for group_name, group_experiments in experiments.items():
        if group_experiments:  # Only show if there are experiments
            console.print(
                f"[bold magenta]📊 Comparison for {group_name}:[/bold magenta]"
            )
            experiment_names = [
                f"{group_name}::{exp}" for exp in group_experiments.keys()
            ]
            table = analyzer.create_comparison_table(
                experiment_names[:10]
            )  # Limit to first 10
            console.print(table)
            console.print()

    # Show best performers
    console.print("[bold gold1]🏆 Analyzing Best Performers...[/bold gold1]")
    analyzer._show_best_performers()
    console.print()

    # Show training progress for a sample experiment
    if analyzer.all_experiments:
        # Find an experiment with training data
        sample_exp = None
        for exp_name, exp_data in analyzer.all_experiments.items():
            if exp_data["type"] == "logs" and exp_data.get("target_performance"):
                sample_exp = exp_name
                break

        if sample_exp:
            console.print("[bold cyan]📈 Sample Training Progress:[/bold cyan]")
            progress_chart = analyzer.create_training_progress_chart(sample_exp)
            console.print(progress_chart)
            console.print()

    # Show features summary
    features_panel = Panel(
        """[bold blue]🎯 Enhanced Visualizer Features:[/bold blue]

✨ [bold green]Multi-Directory Support[/bold green]: Automatically discovers experiments in:
   • Training_results (log-based experiments)
   • ProperTraining (pseudo-labeling experiments) 
   • ProperTraining50Epoch (extended training experiments)
   • results (JSON-format results)

📊 [bold yellow]Interactive Exploration[/bold yellow]:
   • Tree view of all experiments grouped by directory and dataset
   • Comparison tables with performance metrics
   • Training progress visualization with ASCII charts
   • Best performers ranking across all experiments

🔍 [bold cyan]Intelligent Parsing[/bold cyan]:
   • Extracts configuration from directory paths
   • Parses multiple log file formats (source_trained.log, target_performance.log, etc.)
   • Handles different experiment types (PseudoLabelling, SingleLayer, etc.)
   • Automatic metric extraction and performance tracking

📈 [bold magenta]Rich Visualizations[/bold magenta]:
   • Color-coded performance metrics
   • Training progress charts with epoch-by-epoch accuracy
   • Grouped comparisons by dataset and method
   • Export capabilities for reports

🖥️ [bold red]Interactive Mode[/bold red]:
   • Live experiment discovery and refresh
   • Detailed experiment analysis
   • Performance comparison tools
   • Export functionality for documentation

Run with: [bold white]python enhanced_results_visualizer.py --interactive[/bold white]""",
        title="[bold blue]Enhanced Features Summary[/bold blue]",
        border_style="blue",
    )
    console.print(features_panel)
    console.print()

    console.print(
        "[bold green]🎉 Demo completed! Use --interactive flag for full exploration.[/bold green]"
    )


if __name__ == "__main__":
    demo_enhanced_visualizer()
