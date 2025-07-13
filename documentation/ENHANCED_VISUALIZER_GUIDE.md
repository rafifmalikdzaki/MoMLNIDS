# Enhanced MoMLNIDS Results Visualizer

A comprehensive, interactive tool for visualizing and analyzing MoMLNIDS experiment results across multiple training directories with intelligent parsing and rich visualizations.

## 🚀 Key Features

### 📂 Multi-Directory Support
- **Training_results**: Traditional log-based experiments with SingleLayer methods
- **ProperTraining**: PseudoLabelling experiments with clustering
- **ProperTraining50Epoch**: Extended training experiments (50 epochs)
- **results**: JSON-format experiment results

### 🎯 Intelligent Experiment Discovery
- Automatically scans all experiment directories
- Extracts configuration from directory paths (dataset, method, clusters, architecture)
- Parses multiple log file formats (`source_trained.log`, `target_performance.log`, `val_performance.log`, `clustering.log`)
- Groups experiments by directory and dataset for easy navigation

### 📊 Rich Interactive Visualizations
- **Tree View**: Hierarchical display of all experiments grouped by type and dataset
- **Comparison Tables**: Side-by-side performance metrics with color coding
- **Training Progress Charts**: ASCII-based epoch-by-epoch accuracy visualization
- **Best Performers Ranking**: Top experiments across all categories
- **Detailed Analysis**: Deep dive into individual experiment performance

### 🎨 Advanced Features
- Color-coded performance metrics (Green: ≥80%, Yellow: 60-79%, Red: <60%)
- Real-time experiment discovery and refresh
- Export capabilities for documentation and reports
- Search and filter functionality
- Performance assessment with clear indicators

## 📋 Requirements

- Python 3.7+
- Rich library (`pip install rich`)

## 🚀 Quick Start

### Basic Usage
```bash
# Run interactive explorer (recommended)
python enhanced_results_visualizer.py --interactive

# Run demo to see all features
python demo_enhanced_visualizer.py

# Specify different base directory
python enhanced_results_visualizer.py /path/to/experiments --interactive
```

### Interactive Mode Commands

1. **📊 Compare experiments by group**: View performance comparison tables for specific experiment directories
2. **🔍 Analyze specific experiment**: Deep dive into individual experiment with detailed metrics and training progress
3. **📈 Show training progress**: Visualize epoch-by-epoch performance charts
4. **🏆 Show best performers**: Ranking of top experiments across all categories
5. **📋 Export comparison report**: Generate comprehensive text reports
6. **🌳 Refresh experiment tree**: Re-scan directories for new experiments

## 📊 What You'll See

### Experiment Tree Structure
```
MoMLNIDS Experiment Results
├── Training_results (15 experiments)
│   ├── NF-UNSW-NB15-v2 (5 experiments)
│   │   ├── NF-UNSW-NB15-v2_NSingleLayer|1-N
│   │   │   └── Best Acc: 0.948, F1: 0.939
│   └── NF-CSE-CIC-IDS2018-v2 (5 experiments)
├── ProperTraining (14 experiments)
│   ├── NF-UNSW-NB15-v2 (6 experiments)
│   └── NF-CSE-CIC-IDS2018-v2 (4 experiments)
└── ProperTraining50Epoch (4 experiments)
```

### Performance Comparison Tables
| Experiment | Dataset | Method | Best Acc | Best F1 | Final Acc | Epochs |
|------------|---------|--------|----------|---------|-----------|--------|
| UNSW-NB15_PseudoLabelling_Cluster_6 | UNSW-NB15 | PseudoLabelling | **0.961** | **0.949** | **0.793** | 20 |
| CSE-CIC-IDS2018_PseudoLabelling_Cluster_4 | CSE-CIC-IDS2018 | PseudoLabelling | 0.616 | 0.680 | 0.616 | 20 |

### Training Progress Visualization
```
Training Progress (Target Performance)

Best Accuracy: 0.9482 (Epoch 14)
Best F1 Score: 0.9391 (Epoch 14)
Final Accuracy: 0.8755
Total Epochs: 20

Last 10 Epochs Accuracy:
Epoch 14: ███████████████████████████████████████████████░░░ 0.948
Epoch 15: █████████████████████████████████████████████░░░░░ 0.905
Epoch 16: ███████████████████████████████████████████░░░░░░░ 0.875
```

## 🏆 Key Insights from Your Data

Based on the analysis of your experiment results:

### Best Performing Models
1. **UNSW-NB15 Dataset**: Consistently high performance (>94% accuracy)
   - Best: PseudoLabelling with Cluster_6 (96.1% accuracy)
   - SingleLayer methods also perform well (95%+ accuracy)

2. **CSE-CIC-IDS2018 Dataset**: Moderate performance (60-65% best)
   - PseudoLabelling with Cluster_4 shows improvement with 50 epochs
   - Domain weighting helps significantly

3. **ToN-IoT Dataset**: Lower performance (~45% accuracy)
   - Consistent across different methods
   - May need different approaches or more training

### Method Comparison
- **PseudoLabelling**: Generally outperforms SingleLayer on most datasets
- **Extended Training (50 epochs)**: Shows improvement for CSE-CIC-IDS2018
- **Domain Weighting**: Beneficial for some datasets

## 📁 Supported File Formats

### Log Files (Training_results, ProperTraining, ProperTraining50Epoch)
- `source_trained.log`: Training progress with loss, accuracy, F1 metrics
- `target_performance.log`: Target domain evaluation metrics
- `val_performance.log`: Validation performance metrics
- `clustering.log`: Clustering information for PseudoLabelling

### JSON Files (results directory)
- Standard JSON format with config, training_results, evaluation_results

## 🔧 Configuration Extraction

The tool automatically extracts experiment configuration from directory paths:

```
NF-CSE-CIC-IDS2018-v2_N|PseudoLabelling|Cluster_4
│    │                    │      │         │
│    │                    │      │         └── Clusters: 4
│    │                    │      └────────────── Method: PseudoLabelling
│    │                    └───────────────────── Architecture: N
│    └────────────────────────────────────────── Dataset: NF-CSE-CIC-IDS2018-v2
└─────────────────────────────────────────────── Base: NF-CSE-CIC-IDS2018-v2
```

## 📊 Export Capabilities

Generate comprehensive reports including:
- Experiment configurations
- Performance metrics summaries
- Best performer rankings
- Training progress summaries
- Method comparisons

## 🎯 Use Cases

1. **Research Analysis**: Compare different methods and configurations
2. **Performance Tracking**: Monitor training progress and identify best models
3. **Documentation**: Generate reports for papers and presentations
4. **Experiment Planning**: Identify which approaches work best for each dataset
5. **Debugging**: Analyze training curves and identify issues

## 🚀 Next Steps

1. Run the interactive explorer: `python enhanced_results_visualizer.py --interactive`
2. Explore different experiment groups to understand performance patterns
3. Analyze top performers to identify successful configurations
4. Export reports for documentation
5. Use insights to plan future experiments

The enhanced visualizer provides a comprehensive view of your MoMLNIDS experiments, making it easy to identify trends, compare methods, and extract actionable insights from your research data.