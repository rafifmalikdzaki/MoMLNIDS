#!/usr/bin/env python3
"""
Demo script for the MoMLNIDS Results Visualizer
"""

from results_visualizer import MoMLNIDSResultsAnalyzer
from rich.console import Console

console = Console()


def demo_visualizer():
    """Demonstrate the results visualizer functionality."""
    console.print("[bold blue]🚀 MoMLNIDS Results Visualizer Demo[/bold blue]")
    console.print()

    # Initialize analyzer
    analyzer = MoMLNIDSResultsAnalyzer("results")

    # Load results
    console.print("[yellow]Loading experiment results...[/yellow]")
    results = analyzer.load_results()

    if not results:
        console.print("[red]❌ No result files found![/red]")
        return

    console.print(
        f"[green]✅ Successfully loaded {len(results)} result file(s)[/green]"
    )
    console.print()

    # Display comprehensive analysis
    console.print("[cyan]📊 Displaying comprehensive results analysis...[/cyan]")
    console.print()
    analyzer.display_results()

    # Export summary
    console.print("[cyan]📄 Exporting summary to text file...[/cyan]")
    analyzer.export_summary("experiment_summary.txt")
    console.print()

    # Show available features
    console.print("[bold magenta]🎯 Available Features:[/bold magenta]")
    console.print("✨ Rich visualizations with color-coded metrics")
    console.print("📈 Performance comparison across datasets")
    console.print("🔍 Detailed prediction analysis with confusion matrix stats")
    console.print("🏗️  Model architecture visualization")
    console.print("📊 Summary statistics and performance assessment")
    console.print("💾 Export capabilities to text format")
    console.print("🖥️  Interactive mode (use --interactive flag)")
    console.print()

    console.print(
        "[bold green]🎉 Demo completed! Check 'experiment_summary.txt' for exported results.[/bold green]"
    )


if __name__ == "__main__":
    demo_visualizer()
