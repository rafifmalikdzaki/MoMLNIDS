#!/usr/bin/env python3
"""
Demo script for MoMLNIDS TUI interfaces.
"""

import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import rich
        import textual

        return True, "All dependencies available"
    except ImportError as e:
        return False, f"Missing dependency: {e}"


def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "rich",
                "textual",
                "torch",
                "numpy",
                "scikit-learn",
            ]
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    """Main demo function."""
    print("🤖 MoMLNIDS TUI Demo")
    print("=" * 50)

    # Check dependencies
    deps_ok, msg = check_dependencies()
    if not deps_ok:
        print(f"⚠️  {msg}")
        if input("Install dependencies? (y/n): ").lower() == "y":
            if not install_dependencies():
                print("❌ Failed to install dependencies")
                return
        else:
            print("👋 Exiting...")
            return

    print("\nAvailable TUI interfaces:")
    print(
        "1. 🎨 Full TUI (Textual-based) - Advanced interface with tabs and real-time updates"
    )
    print("2. 🖥️  Simple TUI (Rich-based) - Lightweight interface with manual refresh")
    print("3. 🚀 Quick Demo - Show both interfaces")

    choice = input("\nSelect interface (1-3): ").strip()

    if choice == "1":
        print("\n🎨 Launching Full TUI...")
        try:
            from momlnids_tui import MoMLNIDSTUI

            app = MoMLNIDSTUI()
            app.run()
        except ImportError:
            print("❌ Full TUI not available. Try Simple TUI instead.")

    elif choice == "2":
        print("\n🖥️  Launching Simple TUI...")
        try:
            from simple_tui import MoMLNIDSSimpleTUI

            app = MoMLNIDSSimpleTUI()
            app.run()
        except ImportError:
            print("❌ Simple TUI not available.")

    elif choice == "3":
        print("\n🚀 Running Quick Demo...")
        demo_simple_tui()

    else:
        print("❌ Invalid choice")


def demo_simple_tui():
    """Quick demo of the simple TUI."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.layout import Layout
        from rich import box
        import time

        console = Console()

        # Demo header
        console.clear()
        header = Panel.fit(
            "🤖 MoMLNIDS Demo\nMulti-Domain Network Intrusion Detection System",
            border_style="cyan",
        )
        console.print(header)

        # Demo configuration
        time.sleep(1)
        console.print("\n📋 Configuration:")
        config_table = Table(show_header=False, box=box.SIMPLE)
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Model:", "MoMLNIDS")
        config_table.add_row(
            "Datasets:", "NF-UNSW-NB15-v2, NF-CSE-CIC-IDS2018-v2, NF-ToN-IoT-v2"
        )
        config_table.add_row("Epochs:", "20")
        config_table.add_row("Batch Size:", "1")
        config_table.add_row("Learning Rate:", "0.0015")
        config_table.add_row("Features:", "✓ Clustering, ✗ Explainability, ✗ W&B")

        console.print(config_table)

        # Demo training simulation
        time.sleep(2)
        console.print("\n🚀 Simulating Training...")

        from rich.progress import Progress

        with Progress() as progress:
            task1 = progress.add_task("Loading data...", total=100)
            task2 = progress.add_task("Training model...", total=100)

            for i in range(100):
                time.sleep(0.01)
                progress.update(task1, advance=1)
                if i > 30:
                    progress.update(task2, advance=1)

        # Demo results
        time.sleep(1)
        console.print("\n📊 Results:")
        results_table = Table(show_header=True, box=box.ROUNDED)
        results_table.add_column("Dataset", style="cyan")
        results_table.add_column("Accuracy", style="green")
        results_table.add_column("F1-Score", style="yellow")
        results_table.add_column("Precision", style="blue")

        results_table.add_row("NF-UNSW-NB15-v2", "95.34%", "0.921", "0.931")
        results_table.add_row("NF-CSE-CIC-IDS2018-v2", "90.76%", "0.885", "0.899")
        results_table.add_row("NF-ToN-IoT-v2", "96.56%", "0.943", "0.955")

        console.print(results_table)

        # Demo summary
        time.sleep(2)
        summary = Panel(
            "✅ Demo completed successfully!\n\n"
            "The TUI provides:\n"
            "• Interactive training configuration\n"
            "• Real-time monitoring and progress tracking\n"
            "• Multi-dataset evaluation and comparison\n"
            "• Model explainability features\n"
            "• Experiment history and logging\n\n"
            "To run the full interface: python simple_tui.py",
            title="Demo Summary",
            border_style="green",
        )
        console.print(summary)

        input("\nPress Enter to continue...")

    except ImportError:
        print("❌ Rich library not available for demo")


if __name__ == "__main__":
    main()
