# 🛡️ MoMLDNIDS: Multi-Domain Network Intrusion Detection with Pseudo-Labeling and Clustering

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/rafifmalikdzaki/DomainGeneralizationSkripsi/actions/workflows/ci.yml/badge.svg)](https://github.com/rafifmalikdzaki/DomainGeneralizationSkripsi/actions)
[![Weights & Biases](https://img.shields.io/badge/MLOps-Weights%20%26%20Biases-yellow.svg)](https://wandb.ai/)

## 🔬 Overview & Research Motivation

Network Intrusion Detection Systems (NIDS) face significant challenges when deploying across different network domains due to domain shift and limited labeled data availability. This project implements **MoMLDNIDS** (Multi-Domain Machine Learning Domain Network Intrusion Detection System), a cutting-edge research platform that addresses cross-domain NIDS deployment through:

🎯 **Core Innovations:**
- 🔄 **Cross-Domain Adaptation**: Leveraging domain adversarial training with Gradient Reversal Layer (GRL) to learn domain-invariant features
- 🏷️ **Pseudo-Labeling with Clustering**: Using unsupervised clustering to generate pseudo-domain labels for improved domain adaptation
- 🌐 **Multi-Domain Learning**: Training on multiple source domains to enhance generalization to unseen target domains
- 📊 **MLOps Integration**: Complete experiment tracking, configuration management, and explainable AI capabilities

✨ **Modern ML Research Features:**
- 📈 **Experiment Tracking**: Weights & Biases integration for comprehensive metric logging (including ROC-AUC, PR curves, confusion matrices, and clustering metrics) and visualization
- ⚡ **Speed Optimizations**: Implemented mixed precision training, increased batch size, and optimized data loading with `num_workers` and `pin_memory` for faster training. for comprehensive experiment management
- ⚙️ **Configuration Management**: YAML-based configuration system with validation
- 🔍 **Explainable AI**: Multiple interpretability methods (SHAP, LIME, Integrated Gradients)
- 🔄 **Reproducible Research**: Automated environment management and deterministic training
- 🚀 **CI/CD Pipeline**: Automated testing and validation workflows

The core innovation lies in the combination of adversarial domain adaptation with clustering-based pseudo-labeling, enabling effective knowledge transfer between network domains while maintaining high intrusion detection accuracy.

## Architecture

The MoMLDNIDS architecture consists of three main components connected in a adversarial training framework:

```
Input Features (39D) 
       ↓
Feature Extractor (DGFeatExt)
  ├─ BatchNorm1d + ELU + Dropout layers
  ├─ Hidden layers: [64, 32, 16, 10]
  └─ Invariant Features (10D)
       ↓
       ├─────────────────────┬─────────────────────
       ↓                     ↓
Label Classifier          Domain Discriminator
├─ Classification         ├─ Domain prediction  
├─ Output: [Benign,       ├─ Gradient Reversal Layer
│   Attack]               └─ Output: Cluster/Domain IDs
└─ CrossEntropy Loss         └─ CrossEntropy Loss
```

### Key Components:

1. **Feature Extractor**: Deep neural network with batch normalization and ELU activation
2. **Label Classifier**: Binary classification for intrusion detection (Benign/Attack)
3. **Domain Discriminator**: Multi-class classification for domain/cluster identification with GRL
4. **Clustering Module**: Mini-batch K-means for pseudo-domain label generation

## Datasets

The project utilizes three network flow datasets in Parquet format:

| Dataset | Description | Download Link | Records |
|---------|-------------|---------------|---------||
| **NF-UNSW-NB15-v2** | Network flows from UNSW-NB15 dataset | [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | ~2.5M |
| **NF-CSE-CIC-IDS2018-v2** | CIC-IDS2018 network flow dataset | [CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) | ~16M |
| **NF-ToN-IoT-v2** | IoT network traffic dataset | [ToN-IoT](https://research.unsw.edu.au/projects/toniot-datasets) | ~22M |

### Expected Data Directory Structure:
```
./data/parquet/
├── NF-UNSW-NB15-v2/
│   ├── *.parquet files
├── NF-CSE-CIC-IDS2018-v2/
│   ├── *.parquet files
└── NF-ToN-IoT-v2/
    ├── *.parquet files
```

## Installation

### Prerequisites
- Python 3.9 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM (for large datasets)

### Setup with uv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd MoMLNIDS

# Create virtual environment using uv
uv venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows

# Install dependencies
uv pip install -r requirements.txt

# For GPU support, install CUDA version
uv pip install -r requirements-gpu.txt  # if available
```

### Alternative with pip

```bash
pip install -r requirements.txt
```

## 🚀 Usage Examples

### 1. 🎯 Enhanced Training with MLOps

Run the improved training script with full configuration management and experiment tracking:

```bash
# Run with default configuration
python src/main_config.py

# Run with custom config file
python src/main_config.py --config config/custom_config.yaml

# Run quick test with minimal setup
python src/main_config.py --config config/quick_test_config.yaml
```

✨ **Enhanced Features:**
- 📈 **Automatic Experiment Tracking**: Weights & Biases integration
- 🔍 **Explainable AI**: Generate model interpretations automatically
- 💾 **Model Versioning**: Save best models with metadata
- 🔧 **Reproducible Results**: Deterministic training with seed management

### 2. 🔄 Legacy Training Methods

#### Source-Only Baseline Training
```bash
python src/main.py
```

#### Pseudo-Label Clustering Sweep
```bash
python src/main_pseudo.py
```

#### Extended Training (50 Epochs)
```bash
python src/main_pseudo_50.py  # Verify file exists
```

### 3. ⚙️ Configuration Management

The enhanced system uses YAML configuration files for better parameter management:

```yaml
# config/default_config.yaml
project:
  name: "MoMLDNIDS"
  version: "2.0.0"
  description: "Multi-Domain NIDS with Enhanced MLOps"

model:
  feature_extractor:
    hidden_layers: [64, 32, 16, 10]
    dropout_rate: 0.3
    activation: "ELU"

training:
  epochs: 20
  batch_size: 1024
  learning_rate: 0.0015
  grl_weight: 1.25
  clustering_step: 2

wandb:
  enabled: true
  project: "nids-research"
  tags: ["domain-adaptation", "intrusion-detection"]
```

### 4. 🔍 Explainable AI Usage

```python
from src.skripsi_code.explainability import ModelExplainer

# Initialize explainer
explainer = ModelExplainer(model, feature_names)

# Generate explanations
explanation = explainer.explain_instance(
    instance, 
    method="integrated_gradients"
)

# Visualize results
explainer.plot_feature_importance(explanation)
```

### 5. 🧪 Environment Validation

Test your setup with the comprehensive validation script:

```bash
python tests/test_imports.py
```

This validates:
- ✅ Core dependencies
- ✅ Package imports
- ✅ Configuration loading
- ✅ Experiment tracking
- ✅ Explainability modules
- ✅ Optional dependencies (SHAP, LIME)

## 📁 Project Structure

### 🚀 Enhanced MLOps Structure

```
MoMLNIDS/
├── 📄 README.md                      # Top-level README (brief, points to docs/)
├── 📋 requirements.txt               # Python dependencies
├── 📋 requirements.in                # Dependency constraints
├── 📋 requirements-gpu.txt           # GPU dependencies
├── 📋 requirements-gpu.in            # GPU dependency constraints
├── 🔒 uv.lock                       # Locked dependencies (if using uv)
│
├── ⚙️ config/                        # Configuration management
│   ├── default_config.yaml          # Default configuration
│   └── quick_test_config.yaml       # Quick test setup
│
├── 📚 docs/                          # Enhanced documentation
│   ├── README.md                     # Comprehensive README (this file)
│   ├── ENHANCED_FEATURES.md          # Feature descriptions
│   ├── PROJECT_ENHANCEMENT_SUMMARY.md # Project enhancement summary
│   ├── SMOKE_TEST_SUMMARY.md         # Smoke test summary
│   └── UV_WORKFLOW.md                # UV workflow documentation
│
├── 🔄 .github/workflows/             # CI/CD pipelines
│   ├── ci.yml                        # Main CI workflow
│   └── enhanced-features.yml         # Enhanced features testing
│
├── 📦 src/                           # Main source code
│   ├── __init__.py                   # Python package indicator
│   ├── main.py                       # Legacy: Source-only baseline
│   ├── main_config.py                # ✨ Enhanced training with MLOps
│   ├── main_improved.py              # If still needed, otherwise remove
│   ├── main_pseudo.py                # Legacy: Pseudo-labeling experiments
│   ├── main_pseudo_50.py             # Legacy: Extended training
│   ├── manualisasi.py                # If still needed, otherwise remove
│   ├── of-farneback.py               # If still needed, otherwise remove
│   ├── of-lukas.py                   # If still needed, otherwise remove
│   │
│   ├── skripsi_code/                 # Core project logic
│   │   ├── 🧠 model/                 # Neural network models
│   │   │   ├── MoMLNIDS.py          # Main model architecture
│   │   │   ├── FeatureExtractor.py   # Domain-invariant feature extractor
│   │   │   ├── Classifier.py         # Label classifier
│   │   │   └── Discriminator.py      # Domain discriminator with GRL
│   │   │
│   │   ├── 🎯 clustering/            # Clustering algorithms
│   │   │   ├── cluster_methods.py    # K-means, GMM, Spectral clustering
│   │   │   └── cluster_utils.py      # Pseudo-labeling utilities
│   │   │
│   │   ├── 🛠️ utils/                 # Utility functions
│   │   │   ├── dataloader.py         # Data loading and preprocessing
│   │   │   ├── domain_dataset.py     # Domain-aware dataset class
│   │   │   ├── loss.py               # Custom loss functions
│   │   │   └── utils.py              # Training utilities
│   │   │
│   │   ├── 🏃 TrainEval/             # Training and evaluation
│   │   │   └── TrainEval.py          # Training/validation loops
│   │   │
│   │   ├── ⚙️ config/                # Configuration management (if specific to skripsi_code)
│   │   ├── 📊 experiment/            # Experiment tracking (W&B)
│   │   ├── 🔍 explainability/        # Explainable AI module
│   │   ├── figures/                  # Figures generated by notebooks/scripts
│   │   ├── notebooks/                # Jupyter notebooks for exploration/analysis
│   │   ├── Parameters/               # Model parameters/checkpoints
│   │   ├── pipeline/                 # Data processing pipelines
│   │   └── preprocessing/            # Data preprocessing scripts
│   │
│   └── utils/                        # General utility functions (if any outside skripsi_code)
│
├── 🧪 tests/                         # Unit and integration tests
│   ├── smoke_test.py                 # Quick functionality tests
│   ├── test_imports.py               # Comprehensive import validation
│   ├── test.py                       # General test cases
│   └── training_test.py              # Training specific tests
│
├── 📊 results/                       # Experiment outputs
│   ├── experiments/                  # Experiment outputs
│   ├── models/                       # Saved models
│   ├── explanations/                 # XAI outputs
│   └── logs/                         # Training logs
│
├── 💾 data/                          # Raw and processed data
│   └── parquet/                      # Preprocessed datasets
│
├── 📈 logs/                          # General logs
├── 🖼️ plots/                         # General plots
├── 📦 build/                         # Build artifacts
├── 🐍 .venv/                         # Python virtual environment
├── 📊 wandb/                         # Weights & Biases run data
├── 🗑️ __pycache__/                   # Python bytecode cache
├── ⚙️ .aider.tags.cache.v4/          # Aider cache
├── ⚙️ .git/                          # Git repository data
├── ⚙️ .pytest_cache/                 # Pytest cache
├── ⚙️ sk_kc/                         # Potentially redundant virtual environment
├── ⚙️ skripsi_code.egg-info/         # Python package metadata
├── ⚙️ skripsi_kc/                    # Potentially redundant virtual environment
├── ⚙️ Training_results/              # Old training results (consider moving to results/)
└── 📄 pyproject.toml                 # Project metadata and build system
```

### 🆕 New MLOps Features

| Component | Description | Benefits |
|-----------|-------------|----------|
| ⚙️ **Configuration** | YAML-based config management | Reproducible experiments, easy parameter tuning |
| 📊 **Experiment Tracking** | Weights & Biases integration | Automatic logging, comparison, collaboration |
| 🔍 **Explainable AI** | Multiple interpretability methods | Model transparency, debugging, trust |
| 🧪 **Environment Validation** | Comprehensive import testing | CI reliability, quick setup validation |
| 📚 **Enhanced Documentation** | Detailed guides and references | Better onboarding, troubleshooting |
| 🔄 **CI/CD Pipelines** | Automated testing workflows | Quality assurance, continuous integration |

## Training/Evaluation Cycle

### Training Process

1. **Initialization**: Model weights initialized using Xavier normal initialization
2. **Data Loading**: Source domains loaded with domain/cluster labels, target domain for testing
3. **Clustering Step**: Every `CLUSTERING_STEP` epochs, re-cluster source data using Mini-batch K-means
4. **Adversarial Training**: 
   - Forward pass through Feature Extractor
   - Label classification loss (CrossEntropy)
   - Domain discrimination loss with GRL
   - Entropy regularization for uncertainty quantification
5. **Optimization**: Separate optimizers for each component with cosine annealing scheduler

### Evaluation Metrics

- **Accuracy**: Classification accuracy on target domain
- **F1-Score**: Weighted F1-score for imbalanced classes
- **Precision/Recall**: Per-class performance metrics
- **ROC AUC**: Area Under the Receiver Operating Characteristic Curve
- **Average Precision**: Area Under the Precision-Recall Curve
- **Matthews Correlation Coefficient (MCC)**: A balanced measure for binary and multiclass classification
- **Sensitivity (Recall)**: True Positive Rate
- **Specificity**: True Negative Rate
- **Domain Classification**: Domain discrimination accuracy (training only)

All metrics are calculated using `torchmetrics` for consistency and efficiency.

### Logging and Checkpoints

- **Training Logs**: Detailed epoch-wise metrics saved to `{result_folder}/training.log`
- **Clustering Logs**: Cluster assignment changes logged to `clustering.log`
- **Model Checkpoints**: Best models saved every `SAVE_STEP` epochs
- **Results Directory**: `./ProperTraining/{target_domain}/{experiment_name}/`

## Results Directory Structure

```
ProperTraining/
├── NF-UNSW-NB15-v2/
│   ├── NF-UNSW-NB15-v2_N|PseudoLabelling|/
│   │   ├── training.log          # Training metrics
│   │   ├── clustering.log        # Clustering changes
│   │   ├── model_best.pth        # Best model checkpoint
│   │   └── config.json           # Experiment configuration
│   └── ...
├── NF-CSE-CIC-IDS2018-v2/
└── NF-ToN-IoT-v2/
```

### Log Interpretation

**Training Log Format**:
```
Train: Epoch: 15/20 | Alpha: 0.8542 |
LClass: 0.3421 | Acc Class: 0.8901 |
LDomain: 1.2341 | Acc Domain: 0.7823 |
Loss Entropy: 0.1234 |
F1 Score: 0.8856 Precision: 0.8934 Recall: 0.8823

Test: Epoch: 15/20 |
Loss: 0.4123 | Accuracy: 0.8756 |
F1 Score: 0.8701 Precision: 0.8823 Recall: 0.8634
```

## Contributing

We welcome contributions to improve MoMLDNIDS! Please follow these guidelines:

### Development Workflow

1. **Create an Issue**: Describe the bug, feature request, or improvement
2. **Fork & Branch**: Create a feature branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Implement Changes**: Follow existing code style and add tests
4. **Submit Pull Request**: Include clear description and link to issue

### Commit Message Style

Follow conventional commits format:
```
type(scope): description

feat(clustering): add spectral clustering method
fix(dataloader): resolve memory leak in batch processing
docs(readme): update installation instructions
test(model): add unit tests for feature extractor
```

### Code Style

- Follow PEP 8 for Python code formatting
- Use type hints where applicable
- Add docstrings for all public functions
- Ensure all tests pass before submitting PR



## 📚 Documentation

For comprehensive documentation, guides, and tutorials, visit the [`documentation/`](./documentation/) directory:

### 🎯 **Quick Start Guides**
- [`documentation/TUI_README.md`](./documentation/TUI_README.md) - Interactive textual user interface
- [`documentation/ENHANCED_DEMO_USAGE.md`](./documentation/ENHANCED_DEMO_USAGE.md) - Enhanced CLI demo script

### 🔍 **Technical Documentation**
- [`documentation/MAIN_FUNCTIONS_SUMMARY.md`](./documentation/MAIN_FUNCTIONS_SUMMARY.md) - Complete codebase overview
- [`docs/ENHANCED_FEATURES.md`](./docs/ENHANCED_FEATURES.md) - Modern ML features and capabilities

### 📊 **Visualization & Analysis**
- [`documentation/ENHANCED_VISUALIZER_GUIDE.md`](./documentation/ENHANCED_VISUALIZER_GUIDE.md) - Advanced plotting tools
- [`documentation/RESULTS_VISUALIZER_GUIDE.md`](./documentation/RESULTS_VISUALIZER_GUIDE.md) - Standard analysis

### 📋 **Complete Documentation Index**
- [`documentation/DOCUMENTATION_OVERVIEW.md`](./documentation/DOCUMENTATION_OVERVIEW.md) - **Meta-documentation** with navigation guide to all available documentation

## Contact Information

- **Author**: Dzaki Rafif Malik
- **Email**: malikdzaki16@gmail.com
- **Research Group**: Intelligent Systems
- **Institution**: Universitas Brawijaya

For questions, issues, or collaboration opportunities, please contact the author or create an issue in the repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UNSW-NB15, CIC-IDS2018, and ToN-IoT dataset providers
- PyTorch and scikit-learn communities
- Research lab/institution support

---

**Last Updated**: January 2025
