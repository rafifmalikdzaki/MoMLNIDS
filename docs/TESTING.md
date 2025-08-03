# Testing MoMLNIDS Project

This document outlines the testing strategy and how to run the tests for the MoMLNIDS project.

## Test Philosophy

Our testing aims to ensure:
- **Module Integrity**: Individual components (e.g., data loaders, models, utilities) function as expected.
- **System Stability**: The core training and evaluation loops are stable and do not crash.
- **Data Handling**: Data loading and preprocessing work correctly with the expected data formats.
- **Reproducibility**: Tests can be run consistently across different environments.

## Prerequisites

Before running tests, ensure you have:
- The project dependencies installed (preferably in a virtual environment).
- The `sk_kc` virtual environment activated.
- The `data/parquet` directory populated with at least some dummy `.parquet` files for the data loading tests. The tests expect subdirectories like `NF-BoT-IoT-v2`, `NF-CSE-CIC-IDS2018-v2`, `NF-ToN-IoT-v2`, `NF-UNSW-NB15-v2` within `data/parquet`, each containing at least one `.parquet` file.

## How to Run Tests

All tests are run using `pytest`.

1.  **Activate the Virtual Environment**:
    ```bash
    source sk_kc/bin/activate
    ```

2.  **Set PYTHONPATH**:
    The `skripsi_code` package is located in the `src/skripsi_code` directory. To allow Python to find this package during testing, you need to add the `src` directory to your `PYTHONPATH`.
    ```bash
    export PYTHONPATH=$PYTHONPATH:/home/dzakirm/Research/MoMLNIDS/src
    ```
    (Note: Replace `/home/dzakirm/Research/MoMLNIDS` with your actual project root path if different.)

3.  **Run Pytest**:
    You can run all tests, or specific test files.

    To run all tests:
    ```bash
    pytest tests/
    ```

    To run specific test files (recommended for focused testing):
    ```bash
    pytest tests/test_imports.py tests/test_dataloader.py tests/smoke_test.py tests/training_test.py
    ```

## Test Suite Overview

The test suite is organized into several files, each focusing on a different aspect of the project:

### `tests/test_imports.py`
This file verifies that all necessary Python packages and internal `skripsi_code` modules can be imported successfully. It acts as a basic health check for the project's dependencies and module structure.

**What it tests:**
- Basic Python libraries (`torch`, `numpy`, `pandas`, `yaml`).
- Core `skripsi_code` packages (`skripsi_code.model`, `skripsi_code.utils`, etc.).
- Configuration module functionality (loading, accessing attributes).
- Experiment tracking module (`ExperimentTracker`, `MetricsLogger`).
- Explainable AI module (`ModelExplainer`).
- Loading of configuration YAML files (`default_config.yaml`, `quick_test_config.yaml`).

### `tests/test_dataloader.py`
This file focuses on testing the data loading mechanisms, particularly the `MultiChunkParquet` dataset and the `random_split_dataloader` function. It ensures that data can be loaded, split, and accessed correctly.

**What it tests:**
- Initialization of `MultiChunkParquet` dataset.
- Correctness of `__getitem__` method for `MultiChunkParquet` (e.g., returning tensors of correct shape and type).
- Functionality of `random_split_dataloader` (e.g., returning `DataLoader` objects, ensuring data splits are non-empty).
- Data types and shapes of batches returned by the data loaders.

### `tests/smoke_test.py`
This file performs a quick "smoke test" to ensure that the most critical components of the system are functional. It's designed to catch major breakages early.

**What it tests:**
- Successful import of all major project modules.
- Basic instantiation and forward pass of the `MoMLDNIDS` model.
- Basic data loading component imports (without requiring actual data files).

### `tests/training_test.py`
This file contains minimal tests for the training loop. It verifies that a single training step and multiple training steps can be executed without errors, using dummy data.

**What it tests:**
- Execution of a single training step (forward pass, loss calculation, backward pass, optimizer step).
- Stability of the training loop over multiple steps.
- Basic loss calculation and gradient updates.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'skripsi_code'`**: Ensure you have activated your virtual environment and correctly set the `PYTHONPATH` as described in step 2 of "How to Run Tests".
- **`pytest.skip("Test data directory not found...")`**: This means the test data (parquet files) are not present in the expected `data/parquet` directory. You need to ensure that the `data/parquet` directory contains the necessary subdirectories and `.parquet` files as mentioned in the "Prerequisites" section.
- **`PytestReturnNotNoneWarning`**: This warning indicates that a test function is returning a value other than `None`. While not an error, it's a best practice for pytest functions to use `assert` statements for verification and implicitly return `None`. The current tests have been updated to use `assert` to mitigate this.
