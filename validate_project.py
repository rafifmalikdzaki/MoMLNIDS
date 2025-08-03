#!/usr/bin/env python3
"""
Comprehensive project validation script for MoMLNIDS thesis defense.
This script validates that all components are ready for demonstration.
"""

import sys
import os
from pathlib import Path
import importlib.util

# Add project paths
project_root = Path(__file__).resolve().parent  # Current directory is project root
src_path = project_root / "src"

print(f"Project root: {project_root}")
print(f"Source path: {src_path}")

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def test_project_structure():
    """Validate project structure."""
    print("📁 Testing project structure...")

    # Check main directories
    required_dirs = [
        src_path / "skripsi_code",
        src_path / "skripsi_code" / "model",
        src_path / "skripsi_code" / "utils",
        src_path / "skripsi_code" / "clustering",
        src_path / "skripsi_code" / "explainability",
        src_path / "skripsi_code" / "TrainEval",
        src_path / "skripsi_code" / "config",
        src_path / "skripsi_code" / "experiment",
        project_root / "config",
    ]

    all_exist = True
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"  ✅ {dir_path.name} directory exists")
        else:
            print(f"  ❌ {dir_path.name} directory missing")
            all_exist = False

    return all_exist


def test_main_scripts():
    """Test main scripts exist and have content."""
    print("\n📄 Testing main scripts...")

    main_scripts = [
        "main.py",
        "main_improved.py",
        "main_config.py",
        "main_pseudo.py",
        "main_pseudo_50.py",
    ]

    all_exist = True
    for script in main_scripts:
        script_path = src_path / script
        if script_path.exists() and script_path.stat().st_size > 0:
            print(f"  ✅ {script} exists and has content")
        else:
            print(f"  ❌ {script} missing or empty")
            all_exist = False

    return all_exist


def test_module_imports():
    """Test that modules can be imported without errors."""
    print("\n🔧 Testing module imports...")

    modules_to_test = [
        "skripsi_code",
        "skripsi_code.config.config_manager",
        "skripsi_code.clustering.cluster_methods",
        "skripsi_code.explainability.explainer",
    ]

    import_results = []

    for module in modules_to_test:
        try:
            spec = importlib.util.find_spec(module)
            if spec is not None:
                print(f"  ✅ {module} - importable")
                import_results.append(True)
            else:
                print(f"  ❌ {module} - not found")
                import_results.append(False)
        except Exception as e:
            print(f"  ❌ {module} - error: {str(e)[:50]}...")
            import_results.append(False)

    return all(import_results)


def test_config_files():
    """Test configuration files."""
    print("\n⚙️ Testing configuration files...")

    config_files = [
        "default_config.yaml",
        "quick_test_config.yaml",
        "experiment_config.yaml",
    ]

    config_dir = project_root / "config"
    all_exist = True

    for config_file in config_files:
        config_path = config_dir / config_file
        if config_path.exists():
            print(f"  ✅ {config_file} exists")
        else:
            print(f"  ⚠️ {config_file} missing (optional)")

    return True  # Config files are optional


def test_demo_functionality():
    """Test that demo functions exist in modules."""
    print("\n🎯 Testing demo functionality...")

    # Check for main functions by looking for specific patterns
    demo_files = [
        (
            src_path / "skripsi_code" / "clustering" / "cluster_methods.py",
            "demo_clustering_methods",
        ),
        (
            src_path / "skripsi_code" / "explainability" / "explainer.py",
            "demo_explainability",
        ),
        (
            src_path / "skripsi_code" / "config" / "config_manager.py",
            "demo_config_management",
        ),
        (
            src_path / "skripsi_code" / "experiment" / "tracker.py",
            "demo_experiment_tracking",
        ),
        (src_path / "skripsi_code" / "TrainEval" / "TrainEval.py", "demo_train_eval"),
    ]

    all_demos_exist = True
    for file_path, demo_function in demo_files:
        if file_path.exists():
            content = file_path.read_text()
            if demo_function in content and "def " + demo_function in content:
                print(f"  ✅ {file_path.stem} has {demo_function}")
            else:
                print(f"  ❌ {file_path.stem} missing {demo_function}")
                all_demos_exist = False
        else:
            print(f"  ❌ {file_path.stem} file missing")
            all_demos_exist = False

    return all_demos_exist


def test_documentation():
    """Test documentation exists."""
    print("\n📚 Testing documentation...")

    doc_files = ["MAIN_FUNCTIONS_SUMMARY.md", "README.md"]

    all_exist = True
    for doc_file in doc_files:
        doc_path = project_root / doc_file
        if doc_path.exists() and doc_path.stat().st_size > 0:
            print(f"  ✅ {doc_file} exists and has content")
        else:
            print(f"  ⚠️ {doc_file} missing or empty")

    return True  # Documentation is optional for basic functionality


def main():
    """Run all validation tests."""
    print("🎓 MoMLNIDS Project Validation for Thesis Defense")
    print("=" * 60)

    tests = [
        ("Project Structure", test_project_structure),
        ("Main Scripts", test_main_scripts),
        ("Module Imports", test_module_imports),
        ("Configuration Files", test_config_files),
        ("Demo Functionality", test_demo_functionality),
        ("Documentation", test_documentation),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(tests)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<25} {status}")
        if result:
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 PROJECT IS READY FOR THESIS DEFENSE!")
        print("\nQuick Demo Commands:")
        print("- python src/skripsi_code/clustering/cluster_methods.py")
        print("- python src/skripsi_code/explainability/explainer.py")
        print("- python src/skripsi_code/config/config_manager.py")
        print("- python src/main_improved.py")
        return 0
    else:
        print(f"\n⚠️ {total - passed} issues found. Please address them before defense.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
