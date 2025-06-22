# Enhanced NIDS Research Project Features

This document describes the new features and improvements added to the NIDS research project for better organization, experiment tracking, and model interpretability.

## 🆕 New Features Added

### 1. Configuration Management
- **YAML-based configuration system** for easy experiment management
- **Hierarchical configuration** with sensible defaults
- **Configuration validation** and merging capabilities
- **Environment-specific configurations**

### 2. Experiment Tracking with Weights & Biases (wandb)
- **Automatic experiment logging** and tracking
- **Model performance metrics** visualization
- **Hyperparameter tracking** and comparison
- **Model artifact storage** and versioning
- **Confusion matrix** and feature importance plots

### 3. Explainable AI (XAI) Integration
- **Multiple explanation methods**:
  - Integrated Gradients
  - Gradient SHAP
  - Feature Ablation
  - Permutation Importance
- **SHAP and LIME support** (optional dependencies)
- **Feature interaction analysis**
- **Automated visualization** of explanations

### 4. Enhanced Project Structure
```
skripsi_code/
├── config/                    # Configuration management
│   ├── __init__.py
│   └── config_manager.py
├── experiment/               # Experiment tracking
│   ├── __init__.py
│   └── tracker.py
├── explainability/          # Explainable AI
│   ├── __init__.py
│   └── explainer.py
├── model/                   # ML models (existing)
├── utils/                   # Utilities (existing)
├── clustering/              # Clustering methods (existing)
└── TrainEval/              # Training logic (existing)
```

## 🚀 Quick Start

### 1. Install Dependencies

Update your environment with the new dependencies:

```bash
# Install/update requirements
pip install -r requirements.txt

# Optional: Install explainability libraries
pip install shap lime

# Optional: Setup wandb
wandb login
```

### 2. Configuration

The default configuration is located at `config/default_config.yaml`. You can:

- **Use default configuration**:
  ```bash
  python main_improved.py
  ```

- **Use custom configuration**:
  ```bash
  python main_improved.py --config path/to/your/config.yaml
  ```

- **Override specific settings** in your custom config:
  ```yaml
  # custom_config.yaml
  training:
    epochs: 100
    batch_size: 512
  
  wandb:
    enabled: true
    project: "my-nids-experiment"
  ```

### 3. Running Experiments

#### Basic Training
```bash
python main_improved.py \
  --experiment-name "baseline-experiment" \
  --tags "baseline" "initial-run" \
  --notes "Initial baseline experiment with default settings"
```

#### With Custom Configuration
```bash
python main_improved.py \
  --config experiments/high_performance_config.yaml \
  --experiment-name "optimized-model" \
  --tags "optimized" "production"
```

#### Hyperparameter Tuning
```bash
# Create different configs for different hyperparameter sets
python main_improved.py --config configs/lr_001.yaml --experiment-name "lr-0.001"
python main_improved.py --config configs/lr_0001.yaml --experiment-name "lr-0.0001"
```

## 📊 Experiment Tracking

### Weights & Biases Integration

The project now automatically tracks:

- **Training metrics**: Loss, accuracy, precision, recall, F1-score
- **Validation metrics**: Per-epoch validation performance
- **Model artifacts**: Trained model weights and metadata
- **Hyperparameters**: All configuration parameters
- **Visualizations**: Confusion matrices, feature importance plots

### Accessing Results

1. **View in wandb dashboard**: `https://wandb.ai/your-username/nids-research`
2. **Local results**: Saved in `results/` directory
3. **Model checkpoints**: Saved in `models/` directory

## 🔍 Explainable AI

### Available Explanation Methods

1. **Integrated Gradients**: Path-based attribution method
2. **Gradient SHAP**: Gradient-based SHAP values
3. **Feature Ablation**: Occlusion-based importance
4. **Permutation Importance**: Global feature importance
5. **Feature Interactions**: Pairwise feature interaction analysis

### Generating Explanations

Explanations are automatically generated when `explainable_ai.enabled: true` in config:

```python
from skripsi_code.explainability import ModelExplainer

# Create explainer
explainer = ModelExplainer(model, feature_names)

# Explain single instance
explanation = explainer.explain_instance(
    instance, 
    method="integrated_gradients"
)

# Global feature importance
global_explanation = explainer.explain_global(
    X_sample, 
    method="feature_importance"
)
```

### Visualizations

- **Feature importance plots**: `plots/feature_importance.png`
- **Feature interaction heatmaps**: `plots/feature_interactions.png`
- **Instance-level explanations**: Logged to wandb

## ⚙️ Configuration Guide

### Configuration Structure

```yaml
# Project metadata
project:
  name: "nids-research"
  description: "Network Intrusion Detection System"
  version: "1.0.0"

# Experiment tracking
wandb:
  enabled: true
  project: "nids-research"
  entity: "your-username"  # Set your wandb username
  tags: ["nids", "cybersecurity"]

# Model architecture
model:
  name: "MoMLNIDS"
  feature_extractor:
    hidden_sizes: [128, 64, 32]
    dropout: 0.3
  classifier:
    hidden_sizes: [64, 32]
    num_classes: 2

# Training parameters
training:
  batch_size: 256
  epochs: 50
  learning_rate: 0.001
  early_stopping:
    enabled: true
    patience: 10

# Data configuration
data:
  datasets:
    - "NF-BoT-IoT-v2"
    - "NF-CSE-CIC-IDS2018-v2"
    - "NF-ToN-IoT-v2"
    - "NF-UNSW-NB15-v2"
  preprocessing:
    normalize: true
    handle_missing: "drop"

# Explainable AI
explainable_ai:
  enabled: true
  methods:
    - "integrated_gradients"
    - "feature_ablation"
  feature_importance:
    enabled: true
    top_k_features: 20
```

### Environment Variables

You can override configuration using environment variables:

```bash
export WANDB_PROJECT="my-experiment"
export CUDA_VISIBLE_DEVICES="0"
python main_improved.py
```

## 📈 Monitoring and Visualization

### Real-time Monitoring

- **Training progress**: Live plots in wandb dashboard
- **Resource usage**: GPU/CPU utilization tracking
- **Experiment comparison**: Compare multiple runs side-by-side

### Generated Plots

1. **Training curves**: Loss and accuracy over epochs
2. **Confusion matrices**: Per-dataset performance
3. **Feature importance**: Global and local explanations
4. **Model architecture**: Automatic model graph visualization

## 🔧 Advanced Usage

### Custom Experiment Configurations

Create experiment-specific configurations:

```bash
# experiments/ablation_study.yaml
model:
  feature_extractor:
    hidden_sizes: [64, 32]  # Smaller model

training:
  epochs: 30
  
explainable_ai:
  methods: ["feature_ablation"]  # Focus on ablation
```

### Batch Experiments

Run multiple experiments programmatically:

```python
import subprocess
from itertools import product

# Hyperparameter grid
learning_rates = [0.001, 0.0001]
hidden_sizes = [[128, 64], [256, 128, 64]]

for lr, hidden in product(learning_rates, hidden_sizes):
    config_updates = {
        'training': {'learning_rate': lr},
        'model': {'feature_extractor': {'hidden_sizes': hidden}}
    }
    
    # Create custom config and run experiment
    # Implementation details...
```

### Model Comparison

Compare different model architectures:

```bash
# Baseline model
python main_improved.py --config configs/baseline.yaml --experiment-name "baseline"

# Deep model
python main_improved.py --config configs/deep_model.yaml --experiment-name "deep"

# Wide model
python main_improved.py --config configs/wide_model.yaml --experiment-name "wide"
```

## 🐛 Troubleshooting

### Common Issues

1. **wandb not working**:
   ```bash
   wandb login
   # Or disable wandb in config: wandb.enabled: false
   ```

2. **SHAP/LIME import errors**:
   ```bash
   pip install shap lime
   # Or disable in config: explainable_ai.enabled: false
   ```

3. **GPU memory issues**:
   ```yaml
   training:
     batch_size: 128  # Reduce batch size
   device:
     cuda_device: 0  # Specify GPU
   ```

4. **Configuration validation errors**:
   - Check YAML syntax
   - Verify required fields are present
   - Use configuration validation: `config_manager.validate_config()`

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 📚 Next Steps

### Planned Enhancements

1. **Hyperparameter optimization** with Optuna integration
2. **Model ensemble** capabilities
3. **Advanced visualization** with interactive plots
4. **Automated reporting** generation
5. **Cloud deployment** configurations

### Contributing

To add new features:

1. **Configuration**: Add new settings to `default_config.yaml`
2. **Experiment tracking**: Extend `ExperimentTracker` class
3. **Explanations**: Add new methods to `ModelExplainer`
4. **Documentation**: Update this README

## 📖 References

- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

For questions or issues, please check the project's GitHub issues or create a new one.
