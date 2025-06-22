# MoMLDNIDS: Multi-Domain Network Intrusion Detection with Pseudo-Labeling and Clustering

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview & Research Motivation

Network Intrusion Detection Systems (NIDS) face significant challenges when deploying across different network domains due to domain shift and limited labeled data availability. This project implements **MoMLDNIDS** (Multi-Domain Machine Learning Domain Network Intrusion Detection System), a novel approach that addresses cross-domain NIDS deployment through:

- **Cross-Domain Adaptation**: Leveraging domain adversarial training with Gradient Reversal Layer (GRL) to learn domain-invariant features
- **Pseudo-Labeling with Clustering**: Using unsupervised clustering to generate pseudo-domain labels for improved domain adaptation
- **Multi-Domain Learning**: Training on multiple source domains to enhance generalization to unseen target domains

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

### Expected Directory Structure:
```
./skripsi_code/data/parquet/
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
cd skripsi_code

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

## Usage Examples

### 1. Source-Only Baseline Training

Train the model using only source domain data without domain adaptation:

```bash
python main.py
```

This will:
- Train on 2 source domains, test on 1 target domain (rotating through all domains)
- Use standard adversarial training without clustering
- Save results to `./ProperTraining/{target_domain}/`

### 2. Pseudo-Label Clustering Sweep

Run experiments with different cluster numbers to find optimal configuration:

```bash
python main_pseudo.py
```

Features:
- Tests cluster numbers from 5-8 across all domain combinations
- Generates pseudo-domain labels using Mini-batch K-means clustering
- Re-clusters every 2 epochs
- Results saved with cluster configuration in filename

### 3. Extended Training (50 Epochs)

For more thorough training with extended epochs:

```bash
python main_pseudo_50.py
```

**Note**: Verify this file exists in your setup before running.

### Configuration Parameters

Key hyperparameters can be modified in the scripts:

```python
NUM_EPOCH = 20          # Training epochs
BATCH_SIZE = 1          # Batch size
CLUSTERING_STEP = 2     # Re-clustering frequency
NUM_CLUSTERS = 4        # Number of pseudo-domains
INIT_LEARNING_RATE = 0.0015  # Initial learning rate
GRL_WEIGHT = 1.25       # Gradient reversal strength
```

## Project Structure

```
skripsi_code/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── main.py                      # Source-only baseline training
├── main_pseudo.py               # Pseudo-labeling experiments
├── main_pseudo_50.py            # Extended training variant
├── smoke_test.py                # Quick functionality tests
│
├── skripsi_code/                # Main package
│   ├── model/                   # Neural network models
│   │   ├── MoMLNIDS.py         # Main model architecture
│   │   ├── FeatureExtractor.py  # Domain-invariant feature extractor
│   │   ├── Classifier.py        # Label classifier
│   │   └── Discriminator.py     # Domain discriminator with GRL
│   │
│   ├── clustering/              # Clustering algorithms
│   │   ├── cluster_methods.py   # K-means, GMM, Spectral clustering
│   │   └── cluster_utils.py     # Pseudo-labeling utilities
│   │
│   ├── utils/                   # Utility functions
│   │   ├── dataloader.py        # Data loading and preprocessing
│   │   ├── domain_dataset.py    # Domain-aware dataset class
│   │   ├── loss.py              # Custom loss functions
│   │   └── utils.py             # Training utilities
│   │
│   ├── TrainEval/              # Training and evaluation
│   │   └── TrainEval.py        # Training/validation loops
│   │
│   ├── data/                   # Data directory
│   │   ├── parquet/            # Preprocessed datasets
│   │   ├── raw/                # Original datasets
│   │   └── interim/            # Intermediate processing files
│   │
│   └── pipeline/               # Data processing pipelines
│       └── data_chunking_parquet.py  # Parquet conversion utilities
│
└── tests/                      # Unit tests
    └── test.py                 # Test cases
```

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
- **Domain Classification**: Domain discrimination accuracy (training only)

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

## Citation

If you use this work in your research, please cite:

```bibtex
@article{momlднids2024,
  title={Multi-Domain Network Intrusion Detection with Pseudo-Labeling and Clustering},
  author={[Author Name]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## Contact Information

- **Author**: [Your Name]
- **Email**: [your.email@university.edu]
- **Research Group**: [Research Group/Lab Name]
- **Institution**: [University/Institution Name]

For questions, issues, or collaboration opportunities, please contact the author or create an issue in the repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UNSW-NB15, CIC-IDS2018, and ToN-IoT dataset providers
- PyTorch and scikit-learn communities
- Research lab/institution support

---

**Last Updated**: January 2025
