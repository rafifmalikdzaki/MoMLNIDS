#!/usr/bin/env python3
"""
Test the enhanced training progress visualization with F1-Score charts.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from enhanced_results_visualizer import EnhancedMoMLNIDSAnalyzer
from rich.console import Console

console = Console()


def test_training_progress():
    """Test the enhanced training progress with F1-Score visualization."""

    console.print(
        "[bold blue]Testing Enhanced Training Progress with F1-Score[/bold blue]"
    )
    console.print()

    # Initialize analyzer
    analyzer = EnhancedMoMLNIDSAnalyzer(".")

    # Discover experiments
    console.print("[yellow]🔍 Discovering experiments...[/yellow]")
    analyzer.discover_experiments()

    if not analyzer.all_experiments:
        console.print("[red]❌ No experiments found![/red]")
        return

    # Find a PseudoLabelling experiment to demonstrate
    target_experiment = None
    for exp_name, exp_data in analyzer.all_experiments.items():
        if (
            exp_data["type"] == "logs"
            and exp_data.get("target_performance")
            and "PseudoLabelling" in exp_name
            and "Cluster" in exp_name
        ):
            target_experiment = exp_name
            break

    if not target_experiment:
        console.print("[red]❌ No PseudoLabelling experiments found![/red]")
        return

    console.print(
        f"[green]✅ Testing with: {target_experiment.split('::')[-1]}[/green]"
    )
    console.print()

    # Create and show the enhanced training progress chart
    progress_chart = analyzer.create_training_progress_chart(target_experiment)
    console.print(progress_chart)
    console.print()

    console.print("[bold blue]Enhanced Features Demonstrated:[/bold blue]")
    console.print("✅ Shows both Accuracy and F1-Score for last 10 epochs")
    console.print(
        "✅ Color-coded progress bars (Accuracy: green/yellow/red, F1-Score: blue/cyan/bright_blue)"
    )
    console.print("✅ Best performance indicators with epoch numbers")
    console.print("✅ Final performance values")
    console.print("✅ Best loss tracking")
    console.print("✅ Visual progress bars for both metrics with distinct colors")


if __name__ == "__main__":
    try:
        test_training_progress()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Test interrupted gracefully![/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Error during test: {e}[/red]")
