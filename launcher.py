#!/usr/bin/env python3
"""
MoMLNIDS Evaluation Launcher

Quick launcher with common evaluation scenarios.
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import subprocess
import sys


@click.command()
def main():
    """
    MoMLNIDS Evaluation Launcher - Quick access to common evaluation scenarios
    """
    console = Console()

    console.print(Panel.fit("🚀 MoMLNIDS Evaluation Launcher", style="bold blue"))

    # Show available options
    options_table = Table(title="📊 Available Evaluation Options")
    options_table.add_column("Option", style="cyan", width=8)
    options_table.add_column("Description", style="green", width=50)
    options_table.add_column("Command", style="yellow", width=40)

    options_table.add_row(
        "1",
        "🌐 Comprehensive domain generalization (ALL datasets)",
        "interactive_evaluation.py -t comprehensive",
    )
    options_table.add_row(
        "2", "🎯 Single dataset evaluation", "interactive_evaluation.py -t single"
    )
    options_table.add_row(
        "3",
        "📊 All datasets basic summary",
        "interactive_evaluation.py -t all-datasets",
    )
    options_table.add_row(
        "4",
        "⚡ Quick comprehensive (auto-select CSE-CIC)",
        "interactive_evaluation.py --auto-select NF-CSE-CIC-IDS2018-v2",
    )
    options_table.add_row(
        "5",
        "⚡ Quick comprehensive (auto-select ToN-IoT)",
        "interactive_evaluation.py --auto-select NF-ToN-IoT-v2",
    )
    options_table.add_row(
        "6",
        "⚡ Quick comprehensive (auto-select UNSW-NB15)",
        "interactive_evaluation.py --auto-select NF-UNSW-NB15-v2",
    )
    options_table.add_row(
        "7",
        "🔧 Basic prediction demo (synthetic data)",
        "prediction_demo.py --mode demo",
    )

    console.print(options_table)

    # Get user choice
    while True:
        try:
            choice = console.input("\n🎯 Select option (1-7) or 'q' to quit: ")

            if choice.lower() == "q":
                console.print("👋 Goodbye!")
                return

            choice_num = int(choice)

            if choice_num == 1:
                cmd = [
                    sys.executable,
                    "interactive_evaluation.py",
                    "-t",
                    "comprehensive",
                ]
            elif choice_num == 2:
                cmd = [sys.executable, "interactive_evaluation.py", "-t", "single"]
            elif choice_num == 3:
                cmd = [
                    sys.executable,
                    "interactive_evaluation.py",
                    "-t",
                    "all-datasets",
                ]
            elif choice_num == 4:
                cmd = [
                    sys.executable,
                    "interactive_evaluation.py",
                    "--auto-select",
                    "NF-CSE-CIC-IDS2018-v2",
                    "-t",
                    "comprehensive",
                ]
            elif choice_num == 5:
                cmd = [
                    sys.executable,
                    "interactive_evaluation.py",
                    "--auto-select",
                    "NF-ToN-IoT-v2",
                    "-t",
                    "comprehensive",
                ]
            elif choice_num == 6:
                cmd = [
                    sys.executable,
                    "interactive_evaluation.py",
                    "--auto-select",
                    "NF-UNSW-NB15-v2",
                    "-t",
                    "comprehensive",
                ]
            elif choice_num == 7:
                cmd = [
                    sys.executable,
                    "prediction_demo.py",
                    "--mode",
                    "demo",
                    "--num-samples",
                    "5",
                ]
            else:
                console.print("❌ Please enter a number between 1-7")
                continue

            console.print(f"\n🚀 Running: {' '.join(cmd)}")
            console.print()

            # Run the command
            subprocess.run(cmd)

            # Ask if user wants to run another evaluation
            if console.input("\n🔄 Run another evaluation? (y/n): ").lower() != "y":
                break

        except ValueError:
            console.print("❌ Please enter a valid number or 'q'")
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!")
            return
        except Exception as e:
            console.print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
