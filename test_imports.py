#!/usr/bin/env python3
"""
Test script to validate that all new modules can be imported properly.
This is used in CI to ensure the enhanced features work correctly.
"""

import sys
import traceback


def test_basic_imports():
    """Test basic Python package imports."""
    try:
        import torch
        import numpy as np
        import pandas as pd
        import yaml
        print("✓ Basic imports successful")
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
        from skripsi_code.config import Config, ConfigManager, get_config, load_config
        
        # Test basic config loading
        config = get_config()
        assert hasattr(config, 'project')
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        
        print("✓ Configuration module working")
        return True
    except Exception as e:
        print(f"✗ Configuration module failed: {e}")
        traceback.print_exc()
        return False


def test_experiment_module():
    """Test experiment tracking module."""
    try:
        from skripsi_code.experiment import ExperimentTracker, MetricsLogger
        from skripsi_code.config import get_config
        
        # Test tracker initialization (with wandb disabled)
        config = get_config()
        config.wandb['enabled'] = False
        
        tracker = ExperimentTracker(config)
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
        from skripsi_code.explainability import ModelExplainer
        import torch
        import torch.nn as nn
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
        
        assert 'attributions' in explanation
        assert len(explanation['attributions']) == 5
        
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
        optional_results['shap'] = True
        print("✓ SHAP available")
    except ImportError:
        optional_results['shap'] = False
        print("○ SHAP not available (optional)")
    
    # Test LIME
    try:
        import lime
        optional_results['lime'] = True
        print("✓ LIME available")
    except ImportError:
        optional_results['lime'] = False
        print("○ LIME not available (optional)")
    
    # Test wandb
    try:
        import wandb
        optional_results['wandb'] = True
        print("✓ wandb available")
    except ImportError:
        optional_results['wandb'] = False
        print("○ wandb not available")
    
    return optional_results


def test_config_files():
    """Test configuration files can be loaded."""
    try:
        import yaml
        from pathlib import Path
        
        config_files = [
            "config/default_config.yaml",
            "config/quick_test_config.yaml"
        ]
        
        for config_file in config_files:
            path = Path(config_file)
            if path.exists():
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                assert isinstance(config, dict)
                print(f"✓ {config_file} is valid")
            else:
                print(f"○ {config_file} not found")
        
        return True
    except Exception as e:
        print(f"✗ Config file validation failed: {e}")
        return False


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
    print("\n" + "="*50)
    print("IMPORT TEST SUMMARY")
    print("="*50)
    
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


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
