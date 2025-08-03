#!/usr/bin/env python3
"""
Lightweight smoke test for demo purposes - skips heavy data loading.
"""

import sys
from pathlib import Path
from rich.console import Console

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

console = Console()


def main():
    console.print("🔥 Running lightweight smoke tests...", style="bold blue")

    try:
        # Test 1: Basic imports
        console.print("✓ Testing basic imports...", style="green")
        import torch
        import numpy as np
        from src.skripsi_code.model.MoMLNIDS import momlnids

        console.print("  ✓ PyTorch and model imports successful", style="dim green")

        # Test 2: Model instantiation (no CUDA to be faster)
        console.print("✓ Testing model instantiation...", style="green")
        model = momlnids(
            input_nodes=39,
            hidden_nodes=[64, 32, 16, 10],
            classifier_nodes=[64, 32, 16],
            num_domains=3,
            num_class=2,
            single_layer=True,
        )
        console.print("  ✓ Model instantiated successfully", style="dim green")

        # Test 3: Quick forward pass
        console.print("✓ Testing forward pass...", style="green")
        model = model.double()  # Ensure model is double precision
        test_input = torch.randn(5, 39).double()
        class_output, domain_output = model(test_input)
        console.print(
            f"  ✓ Forward pass successful - Class: {class_output.shape}, Domain: {domain_output.shape}",
            style="dim green",
        )

        # Test 4: Import other modules
        console.print("✓ Testing other module imports...", style="green")
        from src.skripsi_code.clustering.cluster_methods import Kmeans
        from src.skripsi_code.explainability.explainer import ModelExplainer

        console.print("  ✓ All module imports successful", style="dim green")

        console.print("\n🎉 All lightweight tests passed!", style="bold green")
        return 0

    except Exception as e:
        console.print(f"\n❌ Test failed: {e}", style="bold red")
        return 1


if __name__ == "__main__":
    main()
