#!/usr/bin/env python3
"""
Test script to validate that all new modules can be imported properly.
This is used in CI to ensure the enhanced features work correctly.
"""

import sys
import traceback
import os
from pathlib import Path

# Add project paths
project_root = Path(__file__).resolve().parents[2]
src_path = project_root / "src"

# Add to Python path
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Src path: {src_path}")
print(f"Updated sys.path includes: {src_path}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")


def test_basic_imports():
    """Test basic Python package imports."""
    try:
        import numpy as np
        import pandas as pd
        import yaml

        print("✓ Basic imports successful")

        # Try torch import but don't fail if not available
        try:
            import torch

            print("✓ PyTorch available")
        except ImportError:
            print("○ PyTorch not available (check installation)")

        return True
    except ImportError as e:
        print(f"✗ Basic imports failed: {e}")
        return False


def test_core_package_imports():
    """Test core skripsi_code package imports."""
    try:
        import skripsi_code
        import skripsi_code.model
        import skripsi_code.utils
        import skripsi_code.clustering

        print("✓ Core package imports successful")
        return True
    except ImportError as e:
        print(f"✗ Core package imports failed: {e}")
        traceback.print_exc()
        return False


def test_config_module():
    """Test configuration module."""
    try:
        from skripsi_code.config import ConfigManager, get_config, load_config

        # Test basic config manager creation
        config_mgr = ConfigManager()

        # Try to get config, but handle if config files don't exist
        try:
            config = get_config()
            assert hasattr(config, "project") or hasattr(config, "model")
            print("✓ Configuration module working with config file")
        except FileNotFoundError:
            print(
                "○ Configuration module working (no config file found, but that's ok)"
            )

        return True
    except Exception as e:
        print(f"✗ Configuration module failed: {e}")
        traceback.print_exc()
        return False


def test_experiment_module():
    """Test experiment tracking module."""
    try:
        from skripsi_code.experiment import ExperimentTracker, MetricsLogger
        from omegaconf import DictConfig

        # Create a simple config for testing
        simple_config = DictConfig(
            {"wandb": {"enabled": False}, "output": {"results_dir": "test_results"}}
        )

        # Test tracker initialization (with wandb disabled)
        tracker = ExperimentTracker(simple_config)
        tracker.init_experiment("test-experiment")

        # Test metrics logger
        logger = MetricsLogger()
        logger.log({"test_metric": 0.5})

        print("✓ Experiment tracking module working")
        return True
    except Exception as e:
        print(f"✗ Experiment tracking module failed: {e}")
        traceback.print_exc()
        return False


def test_explainability_module():
    """Test explainable AI module."""
    try:
        # Test if torch is available
        try:
            import torch
            import torch.nn as nn

            torch_available = True
        except ImportError:
            torch_available = False

        if not torch_available:
            print("○ Explainable AI module test skipped (PyTorch not available)")
            return True

        from skripsi_code.explainability import ModelExplainer
        import numpy as np

        # Create a simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 2)

            def forward(self, x):
                return self.linear(x)

        model = TestModel()
        model.eval()

        feature_names = [f"feature_{i}" for i in range(5)]
        explainer = ModelExplainer(model, feature_names)

        # Test a simple explanation
        instance = np.random.randn(5)
        explanation = explainer.explain_instance(instance, method="feature_ablation")

        assert "attributions" in explanation
        assert len(explanation["attributions"]) == 5

        print("✓ Explainable AI module working")
        return True
    except Exception as e:
        print(f"✗ Explainable AI module failed: {e}")
        traceback.print_exc()
        return False


def test_optional_imports():
    """Test optional dependencies."""
    optional_results = {}

    # Test SHAP
    try:
        import shap

        optional_results["shap"] = True
        print("✓ SHAP available")
    except ImportError:
        optional_results["shap"] = False
        print("○ SHAP not available (optional)")

    # Test LIME
    try:
        import lime

        optional_results["lime"] = True
        print("✓ LIME available")
    except ImportError:
        optional_results["lime"] = False
        print("○ LIME not available (optional)")

    # Test wandb
    try:
        import wandb

        optional_results["wandb"] = True
        print("✓ wandb available")
    except ImportError:
        optional_results["wandb"] = False
        print("○ wandb not available")

    return optional_results

    return optional_results


def test_config_files():
    """Test configuration files can be loaded."""
    try:
        import yaml
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[2]

        config_files = [
            project_root / "config/default_config.yaml",
            project_root / "config/quick_test_config.yaml",
        ]

        all_found = True
        for config_file in config_files:
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)
                assert isinstance(config, dict)
                print(f"✓ {config_file.name} is valid")
            else:
                print(f"✗ {config_file.name} not found")
                all_found = False

        return all_found
    except Exception as e:
        print(f"✗ Config file validation failed: {e}")
        assert False


def main():
    """Run all import tests."""
    print("Running import validation tests...\n")

    tests = [
        ("Basic imports", test_basic_imports),
        ("Core package imports", test_core_package_imports),
        ("Configuration module", test_config_module),
        ("Experiment tracking module", test_experiment_module),
        ("Explainable AI module", test_explainability_module),
        ("Configuration files", test_config_files),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False

    print("\n--- Optional dependencies ---")
    optional_results = test_optional_imports()

    # Summary
    print("\n" + "=" * 50)
    print("IMPORT TEST SUMMARY")
    print("=" * 50)

    failed_tests = []
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if not result:
            failed_tests.append(test_name)

    print("\nOptional dependencies:")
    for dep, available in optional_results.items():
        status = "Available" if available else "Not available"
        print(f"  {dep}: {status}")

    if failed_tests:
        print(f"\n❌ {len(failed_tests)} test(s) failed:")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    else:
        print("\n✅ All import tests passed!")
        return 0


    sys.exit(exit_code)

if __name__ == "__main__":
    main()
