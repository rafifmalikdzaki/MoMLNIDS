# MoMLNIDS Demo Scripts Guide

This guide explains the purpose and usage of each demonstration script in the MoMLNIDS project. While there are multiple scripts, each serves a specific purpose for different aspects of model evaluation and visualization.

## 📋 Scripts Overview

### 1. **`prediction_demo.py`** - Basic Model Prediction Interface
**Purpose**: Simple demonstration of model inference capabilities with synthetic or manual data.

**Key Features**:
- Basic model loading and prediction
- Supports both single sample and batch prediction
- Synthetic data generation for testing
- Manual feature input mode
- CSV file input support

**When to Use**: 
- Testing model loading functionality
- Quick prediction tests with synthetic data
- Learning how the model works internally

**Usage**:
```bash
# Demo mode with synthetic data
python prediction_demo.py --model-path "ProperTraining/NF-CSE-CIC-IDS2018-v2/model_best.pt" --mode demo --num-samples 10

# Single prediction mode (manual input)
python prediction_demo.py --model-path "ProperTraining/NF-CSE-CIC-IDS2018-v2/model_best.pt" --mode single

# Batch mode with CSV file
python prediction_demo.py --model-path "ProperTraining/NF-CSE-CIC-IDS2018-v2/model_best.pt" --mode batch --input-file data.csv
```

---

### 2. **`auto_prediction_demo.py`** - Real Data Evaluation
**Purpose**: Automatic demonstration using real network intrusion data from parquet files.

**Key Features**:
- Loads real data from `src/data/parquet/` directory
- Automatic dataset discovery
- Comprehensive performance metrics (accuracy, precision, recall, F1, AUC-ROC)
- JSON export compatible with experiment format
- Attack type analysis
- Confidence analysis

**When to Use**:
- Evaluating model performance on real data
- Single dataset analysis
- Generating experiment results for documentation

**Usage**:
```bash
# Evaluate on specific dataset
python auto_prediction_demo.py --model-path "ProperTraining/NF-CSE-CIC-IDS2018-v2/model_best.pt" --dataset "NF-CSE-CIC-IDS2018-v2" --num-samples 1000

# Evaluate on all datasets
python auto_prediction_demo.py --model-path "ProperTraining/NF-CSE-CIC-IDS2018-v2/model_best.pt" --all-datasets --num-samples 500

# Export results to JSON
python auto_prediction_demo.py --model-path "ProperTraining/NF-CSE-CIC-IDS2018-v2/model_best.pt" --dataset "NF-CSE-CIC-IDS2018-v2" --export-json results/my_evaluation.json
```

---

### 3. **`comprehensive_evaluation.py`** - Domain Generalization Analysis
**Purpose**: Systematic evaluation across ALL datasets to analyze domain generalization capabilities.

**Key Features**:
- Tests model on all 4 datasets automatically
- Domain generalization analysis (target vs cross-domain performance)
- Statistical summaries and comparisons
- Best/worst performing dataset identification
- Comprehensive JSON export for research

**When to Use**:
- Research-grade domain generalization evaluation
- Comparing model performance across different network domains
- Generating comprehensive research results

**Usage**:
```bash
# Full domain generalization evaluation
python comprehensive_evaluation.py --model-path "ProperTraining/NF-CSE-CIC-IDS2018-v2/model_best.pt" --num-samples 200 --export-json results/comprehensive_analysis.json
```

---

### 4. **`interactive_evaluation.py`** - Enhanced Model Selection Interface
**Purpose**: Advanced interactive interface with intelligent clustering detection and streamlined model filtering.

**Key Features**:
- **Smart model discovery**: Auto-discovers 247+ models, filters to 34 essential models
- **Intelligent config parsing**: Detects pseudo-labeling methods and cluster counts (K=2,3,4,5,6)
- **Best & Last filtering**: Shows only best models (model_best.pt) and final checkpoints per configuration
- **Dataset-grouped display**: Models organized by target dataset with cluster information
- **Enhanced filtering**: Filter by dataset, epochs, method, and cluster configuration
- **Auto-selection capability**: Select models by dataset name without typing paths
- **Rich model details**: Shows pseudo-labeling status, cluster counts, and configuration info

**When to Use**:
- **Primary interface** for model selection and evaluation
- Exploring clustering configurations across different datasets
- Comparing best vs final model performance
- Research analysis requiring specific cluster configurations

**Usage**:
```bash
# Interactive mode with enhanced filtering (RECOMMENDED)
python interactive_evaluation.py

# Auto-select best model for dataset
python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2"

# Comprehensive evaluation with auto-selection
python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2" --evaluation-type comprehensive --num-samples 200
```

**Enhanced Display Example**:
```
🎯 NF-CSE-CIC-IDS2018-v2 Models (Best & Last only - 12 shown)
┃ ID ┃ Method            ┃ Epochs┃ Clusters┃ Model Type  ┃ File       ┃
┃ 1  ┃ PseudoLabelling.. ┃ 20    ┃ K=2     ┃ 🏆 Best     ┃ model_best ┃
┃ 2  ┃ PseudoLabelling.. ┃ 20    ┃ K=2     ┃ 📊 Final    ┃ model_18   ┃
┃ 3  ┃ PseudoLabelling.. ┃ 20    ┃ K=3     ┃ 🏆 Best     ┃ model_best ┃
```

---

### 5. **`demo_enhanced_granular.py`** - Enhanced Visualizer Demo
**Purpose**: Demonstrates the enhanced results visualizer with detailed feature showcase.

**Key Features**:
- Shows experiment tree visualization
- Dataset-specific performance breakdowns
- Clustering analysis for PseudoLabelling methods
- Enhanced comparison tables
- Feature demonstration with real experiment data

**When to Use**:
- Learning about the enhanced visualizer capabilities
- Seeing what features are available
- Understanding experiment organization

**Usage**:
```bash
# Run interactive demo
python demo_enhanced_granular.py
```

---

### 6. **`demo_enhanced_visualizer.py`** - Results Visualizer Demo
**Purpose**: Demonstrates the enhanced results visualizer with actual experiment data.

**Key Features**:
- Multi-directory experiment discovery
- Training progress visualization
- Best performers analysis
- Interactive exploration capabilities
- Export functionality

**When to Use**:
- Understanding experiment analysis capabilities
- Learning how to use the enhanced visualizer
- Exploring existing experiment results

**Usage**:
```bash
# Run visualizer demo
python demo_enhanced_visualizer.py
```

---

## 🎯 Recommended Usage Workflow

### **Primary Workflow (Enhanced Interface)**
```bash
# 1. Start with enhanced interactive evaluation
python interactive_evaluation.py

# 2. Browse models by dataset and cluster configuration
# 3. Select models using visual interface
# 4. Choose evaluation type (single/comprehensive/all-datasets)
# 5. Results automatically exported with meaningful names
```

### **Quick Model Testing**
```bash
# Auto-select best model for specific dataset
python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2"
```

### **Research Analysis**
```bash
# Comprehensive domain generalization with auto-selection
python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2" --evaluation-type comprehensive --num-samples 200

# Compare different cluster configurations
python interactive_evaluation.py  # Then select different K values interactively
```

### **Cluster Configuration Comparison**
```bash
# Interactive mode allows easy comparison of:
# - K=2 vs K=3 vs K=4 vs K=5 vs K=6 clustering
# - 20-epoch vs 50-epoch training
# - Best vs Final model performance
# - Cross-dataset generalization
```

### For Understanding Model Behavior:
```bash
1. python prediction_demo.py --model-path [PATH] --mode demo
2. python auto_prediction_demo.py --model-path [PATH] --dataset [DATASET]
```

### For Results Analysis:
```bash
1. python demo_enhanced_visualizer.py  # See existing results
2. python enhanced_results_visualizer.py --interactive  # Full exploration
```

---

## 🔧 Enhanced Sample Size Options

### **New Default Sample Sizes** (Meaningful Dataset Portions):
- **`interactive_evaluation.py`**: 50,000 samples (default)
- **`comprehensive_evaluation.py`**: 100,000 samples (default)  
- **`auto_prediction_demo.py`**: 25,000 samples (default)
- **`prediction_demo.py`**: 10,000 samples (default)

### **Percentage-Based Sampling** (NEW):
```bash
# Use percentage of entire dataset
python auto_prediction_demo.py -m model.pt --dataset NF-CSE-CIC-IDS2018-v2 --percentage 0.1   # 10%
python auto_prediction_demo.py -m model.pt --dataset NF-CSE-CIC-IDS2018-v2 --percentage 0.5   # 50% 
python auto_prediction_demo.py -m model.pt --dataset NF-CSE-CIC-IDS2018-v2 --percentage 1.0   # 100%

# Interactive evaluation with predefined percentages
python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2" --percentage 20   # 20%
python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2" --percentage 80   # 80%
python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2" --percentage 100  # 100%
```

### **Large-Scale Evaluation Options**:
```bash
# Half million samples
python auto_prediction_demo.py -m model.pt --dataset NF-CSE-CIC-IDS2018-v2 --num-samples 500000

# One million samples  
python auto_prediction_demo.py -m model.pt --dataset NF-CSE-CIC-IDS2018-v2 --num-samples 1000000

# Full dataset (100%)
python auto_prediction_demo.py -m model.pt --dataset NF-CSE-CIC-IDS2018-v2 --percentage 1.0
```

### **Dataset Size Reference**:
- **NF-BoT-IoT-v2**: ~37.7M samples (377 chunks)
- **NF-CSE-CIC-IDS2018-v2**: ~18.7M samples (187 chunks)  
- **NF-ToN-IoT-v2**: ~16.9M samples (169 chunks)
- **NF-UNSW-NB15-v2**: ~2.3M samples (23 chunks)

## 🔧 Common Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--model-path` | Path to trained model | `"ProperTraining/NF-CSE-CIC-IDS2018-v2/model_best.pt"` |
| `--dataset` | Specific dataset to evaluate | `"NF-CSE-CIC-IDS2018-v2"` |
| `--num-samples` | Number of samples to process | `50000` (new default) |
| `--percentage` | Percentage of dataset to use | `0.2` (20%) or `1.0` (100%) |
| `--export-json` | Export results to JSON | `"results/my_results.json"` |
| `--device` | Computation device | `"cpu"` or `"cuda"` |

---

## 📊 Output Formats

### Console Output:
- Rich formatted tables and panels
- Color-coded performance metrics
- Progress bars for long operations
- Interactive prompts and selections

### JSON Export:
- Compatible with existing experiment format
- Includes model configuration
- Performance metrics for all datasets
- Raw predictions and probabilities
- Confusion matrices and confidence analysis

---

## 💡 Tips for Usage

1. **Start with `interactive_evaluation.py`** - Enhanced interface with clustering detection
2. **Use filtering options** to explore specific configurations (dataset/epochs/clusters)
3. **Compare Best vs Final models** to understand training progression
4. **Analyze cluster impacts** by comparing K=2,3,4,5,6 configurations
5. **Use `comprehensive_evaluation.py`** for research-grade domain generalization analysis
6. **Use `auto_prediction_demo.py`** for single dataset deep dives
7. **Always export results** with auto-generated or custom JSON filenames

### **Enhanced Model Selection Tips**
- **🏆 Best models**: Use for final performance evaluation and publication results
- **📊 Final models**: Use for understanding training convergence and stability
- **Cluster comparison**: Systematically test different K values for optimal clustering
- **Cross-dataset analysis**: Use comprehensive evaluation to study domain generalization
- **Filter combinations**: Use dataset + epochs filters to focus on specific configurations

---

## 🚀 Quick Start Examples

### **Enhanced Interactive Evaluation (RECOMMENDED)**
```bash
# Interactive model selection with clustering information (50k samples default)
python interactive_evaluation.py

# Quick comprehensive evaluation with auto-selection (100k samples default)
python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2" --evaluation-type comprehensive

# Large-scale percentage-based evaluation
python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2" --percentage 80 --evaluation-type comprehensive
```

### **Cluster Configuration Analysis**
```bash
# Compare different clustering configurations interactively (with meaningful sample sizes)
python interactive_evaluation.py --percentage 40
# Filter by dataset: NF-CSE-CIC-IDS2018-v2
# Compare models with K=2, K=3, K=4, K=5 clusters using 40% of dataset

# Auto-select and evaluate specific configurations with large samples
python interactive_evaluation.py --auto-select "NF-UNSW-NB15-v2" --evaluation-type comprehensive --num-samples 200000
# (UNSW dataset has K=2,3,4,5,6 configurations available)
```

### **Full-Scale Evaluation Examples**
```bash
# Full dataset comprehensive evaluation
python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2" --evaluation-type comprehensive --percentage 100

# Large sample comprehensive analysis  
python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2" --evaluation-type comprehensive --num-samples 1000000

# Multi-million sample single dataset analysis
python auto_prediction_demo.py --model-path "ProperTraining/NF-CSE-CIC-IDS2018-v2/model_best.pt" --dataset "NF-CSE-CIC-IDS2018-v2" --num-samples 5000000
```

### **Cross-Dataset Domain Generalization**
```bash
# Systematic evaluation across all target domains
python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2" --evaluation-type comprehensive --export-json results/cse_domain_analysis.json
python interactive_evaluation.py --auto-select "NF-ToN-IoT-v2" --evaluation-type comprehensive --export-json results/ton_domain_analysis.json
python interactive_evaluation.py --auto-select "NF-UNSW-NB15-v2" --evaluation-type comprehensive --export-json results/unsw_domain_analysis.json
```

### Explore Existing Results:
```bash
python demo_enhanced_visualizer.py
```

This structure provides multiple entry points for different use cases while maintaining compatibility and avoiding true redundancy. The **enhanced interactive evaluation interface** serves as the primary entry point, offering intelligent model discovery, clustering configuration analysis, and streamlined evaluation workflows. Each script serves a specific purpose in the overall evaluation ecosystem, with the interactive interface providing the most comprehensive and user-friendly experience for model selection and evaluation.