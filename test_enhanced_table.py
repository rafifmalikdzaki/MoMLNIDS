#!/usr/bin/env python3
"""
Quick test to demonstrate the enhanced comparison table functionality.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from enhanced_results_visualizer import EnhancedMoMLNIDSAnalyzer
from rich.console import Console

console = Console()


def test_comparison_table():
    """Test the enhanced comparison table with the new columns."""

    console.print("[bold blue]Testing Enhanced Comparison Table[/bold blue]")
    console.print()

    # Initialize analyzer
    analyzer = EnhancedMoMLNIDSAnalyzer(".")

    # Discover experiments
    console.print("[yellow]🔍 Discovering experiments...[/yellow]")
    analyzer.discover_experiments()

    if not analyzer.experiment_groups:
        console.print("[red]❌ No experiment groups found![/red]")
        return

    # Test with Training_results group
    if "Training_results" in analyzer.experiment_groups:
        console.print("[green]✅ Testing with Training_results group[/green]")
        console.print()

        # Get first 5 experiments for demo
        experiments = list(analyzer.experiment_groups["Training_results"].keys())[:5]
        full_experiment_names = [f"Training_results::{exp}" for exp in experiments]

        console.print(
            f"[bold green]Showing enhanced comparison for first 5 Training_results experiments:[/bold green]"
        )
        console.print()

        # Create and show the enhanced table
        table = analyzer.create_comparison_table(full_experiment_names)
        console.print(table)
        console.print()

        console.print("[bold blue]Enhanced Features Demonstrated:[/bold blue]")
        console.print("✅ Removed Architecture column")
        console.print("✅ Added Precision column")
        console.print("✅ Added Recall column")
        console.print("✅ Added Final Loss column (color-coded)")
        console.print("✅ Cluster information highlighted in purple")
        console.print("✅ Performance color-coding (green/yellow/red)")
        console.print("✅ Compact display with proper column widths")

    else:
        console.print("[red]❌ Training_results group not found![/red]")


if __name__ == "__main__":
    test_comparison_table()
