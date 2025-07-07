#!/usr/bin/env python3
"""
Simple test script to verify the enhanced demo functionality
"""

import sys
from pathlib import Path


def test_imports():
    """Test if required packages are available"""
    try:
        import click

        print("✅ Click imported successfully")
    except ImportError:
        print("❌ Click not available")
        return False

    try:
        import rich
        from rich.console import Console

        console = Console()
        console.print("✅ Rich imported successfully", style="green")
    except ImportError:
        print("❌ Rich not available")
        return False

    try:
        from omegaconf import DictConfig, OmegaConf

        print("✅ OmegaConf imported successfully")
    except ImportError:
        print("❌ OmegaConf not available")
        return False

    return True


def test_config_creation():
    """Test configuration creation"""
    try:
        from omegaconf import OmegaConf

        config = {
            "training": {"batch_size": 16, "epochs": 1, "max_batches": 5},
            "project": {"name": "test"},
        }

        omega_config = OmegaConf.create(config)

        # Test saving
        Path("config").mkdir(exist_ok=True)
        with open("config/test_config.yaml", "w") as f:
            OmegaConf.save(omega_config, f)

        # Test loading
        loaded_config = OmegaConf.load("config/test_config.yaml")

        assert loaded_config.training.batch_size == 16
        assert loaded_config.training.epochs == 1
        assert loaded_config.training.max_batches == 5

        print("✅ Configuration creation and loading works")

        # Cleanup
        Path("config/test_config.yaml").unlink(missing_ok=True)

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_rich_features():
    """Test Rich UI features"""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.progress import Progress
        import time

        console = Console()

        # Test panel
        console.print(Panel("Testing Rich Features", style="blue"))

        # Test table
        table = Table(title="Test Results")
        table.add_column("Feature", style="cyan")
        table.add_column("Status", style="green")

        table.add_row("Panel", "✅ Working")
        table.add_row("Table", "✅ Working")

        console.print(table)

        # Test progress bar
        with Progress(console=console) as progress:
            task = progress.add_task("Testing progress...", total=3)
            for i in range(3):
                time.sleep(0.1)
                progress.advance(task)

        console.print("✅ Rich features working correctly", style="bold green")
        return True

    except Exception as e:
        print(f"❌ Rich features test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Testing Enhanced Demo Dependencies\n")

    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config_creation),
        ("Rich Features Test", test_rich_features),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append(result)

    print(f"\n{'=' * 50}")
    print("📊 Test Summary")
    print(f"{'=' * 50}")

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        print("✅ Enhanced demo script is ready to use!")
    else:
        print(f"⚠️  Some tests failed ({passed}/{total})")
        print("❌ Please install missing dependencies:")
        print("   pip install rich click omegaconf")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
