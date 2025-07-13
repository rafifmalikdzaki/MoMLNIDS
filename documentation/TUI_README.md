# MoMLNIDS Textual User Interface (TUI)

Interactive dashboard for the Multi-Domain Network Intrusion Detection System training and evaluation.

## Features

🎨 **Two Interface Options**:
- **Full TUI** (`momlnids_tui.py`) - Advanced interface with real-time updates, tabs, and interactive widgets
- **Simple TUI** (`simple_tui.py`) - Lightweight interface with manual refresh and rich terminal output

🚀 **Key Capabilities**:
- Interactive training configuration and monitoring
- Real-time progress tracking with live metrics
- Multi-dataset evaluation and performance comparison
- Model explainability and feature importance visualization
- Experiment history and logging
- System information and resource monitoring

## Quick Start

### 1. Run the Demo
```bash
python3 demo_tui.py
```

### 2. Launch Simple TUI (Recommended)
```bash
python3 simple_tui.py
```

### 3. Launch Full TUI (Advanced)
```bash
python3 momlnids_tui.py
```

## Interface Overview

### Simple TUI Features:
- 📋 **Configuration Panel**: Modify training parameters interactively
- 📊 **Status Dashboard**: Real-time training status and metrics
- 📈 **Performance Metrics**: Multi-dataset evaluation results
- 📝 **Activity History**: Track experiment progress
- ⌨️ **Keyboard Shortcuts**: Quick access to common actions

### Commands:
- `1` or `train` - Start model training
- `2` or `config` - Modify configuration settings
- `3` or `evaluate` - Run model evaluation
- `4` or `explain` - Generate model explanations
- `h` or `help` - Show help documentation
- `q` or `quit` - Exit application

## Configuration Options

The TUI allows you to configure:

### Model Settings:
- Model architecture (MoMLNIDS, Custom)
- Hidden layer dimensions
- Number of training epochs
- Batch size and learning rate

### Dataset Selection:
- NF-UNSW-NB15-v2
- NF-CSE-CIC-IDS2018-v2
- NF-ToN-IoT-v2
- Multi-dataset training combinations

### Features:
- ✅ **Clustering**: Enable pseudo-labeling with clustering
- 🧠 **Explainability**: Generate SHAP values and feature importance
- 📊 **W&B Logging**: Experiment tracking with Weights & Biases

## Installation

### Dependencies:
```bash
pip install rich textual torch numpy scikit-learn
```

### Optional (for full functionality):
```bash
pip install wandb matplotlib seaborn shap
```

## Usage Examples

### Basic Training Workflow:
1. Launch TUI: `python3 simple_tui.py`
2. Configure settings: Press `2`
3. Start training: Press `1`
4. Monitor progress in real-time
5. Evaluate results: Press `3`
6. Generate explanations: Press `4`

### Advanced Features:
- **Live Dashboard**: Use `--live` flag for auto-refreshing interface
- **Demo Mode**: Use `--demo` flag to run with simulated data
- **Help System**: Press `h` for context-sensitive help

## Integration with Existing Code

The TUI interfaces integrate with your existing MoMLNIDS components:

### Training Integration:
```python
from src.skripsi_code.TrainEval.TrainEval import train, eval
from src.skripsi_code.model.MoMLNIDS import momlnids
from src.skripsi_code.utils.dataloader import random_split_dataloader
```

### Configuration Management:
```python
from src.skripsi_code.config import load_config, get_config
```

### Explainability Features:
```python
from src.skripsi_code.explainability.explainer import ModelExplainer
```

## Screenshots

### Simple TUI Main Dashboard:
```
┌─────────────────────────────────────────────────────────────┐
│        MoMLNIDS - Multi-Domain Network Intrusion           │
│             Detection System                                │
│        Interactive Training & Evaluation Dashboard         │
└─────────────────────────────────────────────────────────────┘

┌─Configuration─────────┐  ┌─Training Status────────┐
│ Model: MoMLNIDS       │  │ Status: 🟢 Ready      │
│ Datasets: 3 selected  │  │ Last Run: Not started │
│ Epochs: 20            │  │ Next: Configure & Train│
│ Batch Size: 1         │  │                       │
│ Learning Rate: 0.0015 │  └───────────────────────┘
│ ✓ Clustering          │
│ ✗ Explainability      │  ┌─Performance Metrics────┐
│ ✗ W&B Logging         │  │ No metrics available   │
└───────────────────────┘  │ Run training to see    │
                          │ results.               │
                          └───────────────────────┘

Commands: [1] Train  [2] Config  [3] Evaluate  [4] Explain  [q] Quit
```

## Troubleshooting

### Common Issues:

1. **Import Errors**: Install missing dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA Issues**: TUI will automatically fallback to CPU mode

3. **Data Path Issues**: Update paths in configuration files

4. **Permission Errors**: Ensure scripts are executable:
   ```bash
   chmod +x *.py
   ```

## Customization

### Adding New Features:
1. Extend the configuration panels
2. Add new command handlers
3. Integrate with additional MoMLNIDS modules

### Theming:
- Modify color schemes in CSS sections
- Customize panel layouts and styles
- Add new visual components

## Contributing

To extend the TUI:
1. Follow the existing widget patterns
2. Add new panels to the layout system
3. Implement corresponding action handlers
4. Update help documentation

## License

Same license as the main MoMLNIDS project.