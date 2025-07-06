# Enhanced MoMLNIDS Demo Script Usage Guide

## Overview

The enhanced `demo_sidang_enhanced.py` script provides a beautiful, interactive CLI experience powered by **Click**, **Rich**, and **OmegaConf**. It features progress bars, colorful output, model testing capabilities, and flexible configuration management.

## 🚀 Quick Start

```bash
# Install rich if not already installed
pip install rich

# Make script executable
chmod +x demo_sidang_enhanced.py

# Show help
python demo_sidang_enhanced.py --help

# Run comprehensive tests
python demo_sidang_enhanced.py test

# Test specific functions only
python demo_sidang_enhanced.py test -f environment,explainability

# Test a trained model
python demo_sidang_enhanced.py test-model path/to/model.pth

# Run quick training
python demo_sidang_enhanced.py train --test-mode

# Run experiments
python demo_sidang_enhanced.py experiment -s baseline --test-mode
```

## 🎨 Features

### ✨ Beautiful CLI Interface
- **Rich progress bars** with real-time updates
- **Colorful output** with status indicators
- **Interactive prompts** for configuration
- **Formatted tables** for results display
- **Panel layouts** for organized information

### ⚡ Fast Testing Defaults
- **Small batch sizes** (16 by default)
- **Limited batches** (5 per epoch)
- **Single epoch** training
- **Quick timeouts** for faster feedback

### 🧪 Model Testing
- **Load .pth files** and validate models
- **Run inference tests** with dummy data
- **Display model information** (parameters, device, etc.)
- **Error handling** for corrupted models

### ⚙️ Smart Configuration
- **OmegaConf integration** for flexible configs
- **Interactive config creation**
- **Override defaults** easily
- **Template-based** configuration management

## 📋 Commands

### `test` - Run Functionality Tests

```bash
# Run all tests
python demo_sidang_enhanced.py test

# Run specific test functions
python demo_sidang_enhanced.py test --functions environment,explainability,quick_training

# Limit training batches
python demo_sidang_enhanced.py test --max-batches 3

# Use custom config
python demo_sidang_enhanced.py test --config config/my_config.yaml
```

**Available test functions:**
- `environment` - Test imports and dependencies
- `explainability` - Test XAI functionality  
- `quick_training` - Run minimal training loop

### `test-model` - Test Trained Models

```bash
# Test a model file
python demo_sidang_enhanced.py test-model models/best_model.pth

# Test with custom data
python demo_sidang_enhanced.py test-model models/best_model.pth --data-path data/test.csv

# Custom batch size for inference
python demo_sidang_enhanced.py test-model models/best_model.pth --batch-size 64
```

### `train` - Run Training

```bash
# Quick training (test mode)
python demo_sidang_enhanced.py train --test-mode

# Custom training parameters
python demo_sidang_enhanced.py train --epochs 2 --max-batches 10

# Use custom configuration
python demo_sidang_enhanced.py train --config config/my_training_config.yaml
```

### `experiment` - Run Full Experiments

```bash
# Run baseline experiment (test mode)
python demo_sidang_enhanced.py experiment --scenario baseline --test-mode

# Run pseudo-labeling experiment
python demo_sidang_enhanced.py experiment --scenario pseudo_labeling --test-mode

# Run all experiments
python demo_sidang_enhanced.py experiment --scenario all --test-mode

# Custom batch limits
python demo_sidang_enhanced.py experiment --scenario baseline --max-batches 20
```

### `config` - Configuration Management

```bash
# Show current configuration and create custom configs
python demo_sidang_enhanced.py config
```

This command provides an interactive interface to:
- View default configuration
- Create custom configurations
- Set batch sizes, epochs, and other parameters
- Save configurations for reuse

## 🎯 Usage Examples

### Quick Validation Before Defense

```bash
# 1. Test environment setup
python demo_sidang_enhanced.py test -f environment

# 2. Quick functionality check
python demo_sidang_enhanced.py test -f explainability,quick_training --max-batches 3

# 3. Test a trained model
python demo_sidang_enhanced.py test-model models/baseline_model.pth
```

### During Defense Presentation

```bash
# Show training process (very quick)
python demo_sidang_enhanced.py train --test-mode --epochs 1 --max-batches 2

# Demonstrate explainability
python demo_sidang_enhanced.py test -f explainability

# Show experiment results
python demo_sidang_enhanced.py experiment -s baseline --test-mode --max-batches 3
```

### Development and Debugging

```bash
# Create custom config for development
python demo_sidang_enhanced.py config

# Test with custom settings
python demo_sidang_enhanced.py test --config config/dev_config.yaml --max-batches 1

# Quick model validation
python demo_sidang_enhanced.py test-model models/checkpoint_epoch_5.pth
```

## 🔧 Configuration

### Default Configuration

The script uses smart defaults optimized for speed:

```yaml
training:
  batch_size: 16      # Small for fast processing
  epochs: 1           # Single epoch for demo
  max_batches: 5      # Limit batches per epoch
  learning_rate: 0.001
  
wandb:
  enabled: false      # Disabled by default for speed

device:
  num_workers: 0      # No multiprocessing for simplicity
```

### Custom Configuration

Create custom configurations interactively:

```bash
python demo_sidang_enhanced.py config
```

Or manually create YAML files:

```yaml
# config/my_demo_config.yaml
project:
  name: "My_Demo"

training:
  batch_size: 32
  epochs: 2
  max_batches: 10
  learning_rate: 0.001

wandb:
  enabled: true
  project: "my-demo-project"
```

## 🎨 Rich UI Features

### Progress Bars
- Real-time command execution progress
- Elapsed time tracking
- Status updates with command output

### Formatted Output
- ✅ Success indicators
- ❌ Failure indicators  
- ⚠️ Warning indicators
- 🎯 Action indicators

### Interactive Tables
- Test results summary
- Model information display
- Configuration overview

### Panels and Layouts
- Organized information display
- Color-coded sections
- Professional presentation

## 🚨 Error Handling

The enhanced script provides robust error handling:

- **Timeout protection** - Commands won't hang indefinitely
- **Graceful failures** - Clear error messages with suggestions
- **Model validation** - Checks for corrupted or incompatible models
- **Configuration validation** - Validates YAML syntax and required fields

## 🔍 Troubleshooting

### Common Issues

1. **Rich not installed**
   ```bash
   pip install rich
   ```

2. **Model loading errors**
   ```bash
   # Check if model file exists and is valid
   python demo_sidang_enhanced.py test-model path/to/model.pth
   ```

3. **Configuration errors**
   ```bash
   # Use default configuration
   python demo_sidang_enhanced.py test --max-batches 1
   ```

4. **Import errors**
   ```bash
   # Test environment first
   python demo_sidang_enhanced.py test -f environment
   ```

### Performance Tips

1. **Use test mode** for quick validation:
   ```bash
   python demo_sidang_enhanced.py train --test-mode
   ```

2. **Limit batches** for ultra-fast testing:
   ```bash
   python demo_sidang_enhanced.py test --max-batches 1
   ```

3. **Test specific functions** only:
   ```bash
   python demo_sidang_enhanced.py test -f environment
   ```

## 🎓 Integration with Thesis Defense

### Pre-Defense Checklist

```bash
# 1. Environment validation
python demo_sidang_enhanced.py test -f environment

# 2. Model testing
python demo_sidang_enhanced.py test-model models/best_baseline.pth
python demo_sidang_enhanced.py test-model models/best_pseudo.pth

# 3. Quick functionality demo
python demo_sidang_enhanced.py test -f explainability,quick_training --max-batches 2
```

### Live Demonstration

```bash
# Show beautiful training process (30 seconds)
python demo_sidang_enhanced.py train --test-mode --max-batches 2

# Demonstrate explainability features
python demo_sidang_enhanced.py test -f explainability

# Show experiment comparison
python demo_sidang_enhanced.py experiment -s all --test-mode --max-batches 1
```

## 🆚 Comparison with Original Script

| Feature | Original Script | Enhanced Script |
|---------|----------------|-----------------|
| UI | Plain text | Rich, colorful, interactive |
| Progress | Basic prints | Progress bars with time |
| Configuration | YAML only | OmegaConf + interactive |
| Model Testing | None | Full model validation |
| Error Handling | Basic | Comprehensive with timeouts |
| Speed | Slow defaults | Fast defaults |
| Flexibility | Limited | Highly configurable |

## 🔮 Advanced Usage

### Batch Processing Multiple Models

```bash
# Test multiple models
for model in models/*.pth; do
    echo "Testing $model"
    python demo_sidang_enhanced.py test-model "$model"
done
```

### Custom Experiment Workflows

```bash
# Create custom config
python demo_sidang_enhanced.py config

# Run with custom config
python demo_sidang_enhanced.py experiment -s baseline --config config/custom_config.yaml
```

### Integration with CI/CD

```bash
# Quick validation in CI
python demo_sidang_enhanced.py test -f environment --max-batches 1
```

The enhanced demo script transforms the MoMLNIDS demonstration experience with modern CLI tools, making it perfect for thesis defense presentations and development workflows.