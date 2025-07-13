# Enhanced Interactive Evaluation Guide

This guide explains the enhanced interactive evaluation system for MoMLNIDS, focusing on the improved model selection interface with intelligent clustering configuration detection and streamlined model filtering.

## 🚀 Overview

The enhanced interactive evaluation system provides a user-friendly interface to browse, select, and evaluate trained MoMLNIDS models without needing to remember complex file paths. The system automatically discovers 247+ trained models and presents them in an organized, filtered view.

## ✨ Key Features

### 📊 **Intelligent Configuration Parsing**
- **Automatic pseudo-labeling detection**: Identifies PseudoLabelling methods
- **Cluster count extraction**: Detects K=2, K=3, K=4, K=5, K=6 configurations
- **Method identification**: Recognizes SingleLayer, DANN, CORAL, and other methods
- **Epoch detection**: Distinguishes between 20-epoch and 50-epoch training

### 🔧 **Smart Model Filtering**
- **Best & Last Only**: Shows only the most relevant models per configuration
  - 🏆 **Best Models**: `model_best.pt` (optimal performance during training)
  - 📊 **Final Models**: Highest numbered checkpoint (e.g., `model_18.pt`, `model_48.pt`)
- **Reduced complexity**: From 247 total models to ~34 essential models
- **Eliminates redundancy**: Filters out intermediate training checkpoints

### 📋 **Enhanced Display**
- **Dataset-grouped tables**: Models organized by target dataset
- **Cluster information**: Clear K=N indicators for pseudo-labeling
- **Model type icons**: Visual distinction between best and final models
- **Compact view**: Essential information in readable format

## 📁 Model Organization

### **Discovered Models Structure**
```
📊 Total Models: 247
📋 Filtered View: 34 (Best & Last only)

🎯 NF-CSE-CIC-IDS2018-v2 Models (12 shown)
├── PseudoLabelling (K=2) - 20 epochs: Best + Final
├── PseudoLabelling (K=3) - 20 epochs: Best + Final  
├── PseudoLabelling (K=4) - 20 epochs: Best + Final
├── PseudoLabelling (K=5) - 20 epochs: Best + Final
├── PseudoLabelling (K=4) - 50 epochs: Best + Final
└── PseudoLabelling (K=5) - 50 epochs: Best + Final

🎯 NF-ToN-IoT-v2 Models (10 shown)
├── PseudoLabelling (K=2,3,4,5) - 20 epochs: Best + Final each
└── PseudoLabelling (K=4) - 50 epochs: Best + Final

🎯 NF-UNSW-NB15-v2 Models (12 shown)
├── PseudoLabelling (K=2,3,4,5,6) - 20 epochs: Best + Final each
└── PseudoLabelling (K=4) - 50 epochs: Best + Final
```

## 🎯 Usage Examples

### **Interactive Mode (Recommended)**
```bash
# Launch interactive interface
python interactive_evaluation.py

# Follow prompts to:
# 1. Filter by dataset (optional)
# 2. Filter by epochs (optional) 
# 3. Filter by method (optional)
# 4. Select model by ID number
# 5. Choose evaluation type
```

### **Auto-Selection Mode**
```bash
# Auto-select best model for specific dataset
python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2"

# Auto-select with comprehensive evaluation
python interactive_evaluation.py \
    --auto-select "NF-CSE-CIC-IDS2018-v2" \
    --evaluation-type comprehensive \
    --num-samples 200
```

### **Filtered Browsing**
```bash
# Show only 50-epoch models
python interactive_evaluation.py --evaluation-type single

# Then filter by epochs: 50
```

## 📊 Model Selection Interface

### **Table Display Format**
```
🎯 NF-CSE-CIC-IDS2018-v2 Models (Best & Last only - 12 shown)
┏━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┓
┃ ID ┃ Method            ┃ Epochs┃ Clusters┃ Model Type  ┃ File       ┃ Size    ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━┩
│ 1  │ PseudoLabelling..│ 20    │ K=2    │ 🏆 Best     │ model_best │ 0.1 MB  │
│ 2  │ PseudoLabelling..│ 20    │ K=2    │ 📊 Final    │ model_18   │ 0.1 MB  │
│ 3  │ PseudoLabelling..│ 20    │ K=3    │ 🏆 Best     │ model_best │ 0.1 MB  │
│ 4  │ PseudoLabelling..│ 20    │ K=3    │ 📊 Final    │ model_18   │ 0.1 MB  │
└────┴───────────────────┴───────┴────────┴─────────────┴────────────┴─────────┘
```

### **Model Detail View**
When selecting a model, detailed information is displayed:
```
📋 Model Details: model_best.pt
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Property          ┃ Value                                                    ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Full Path         │ ProperTraining/NF-CSE-CIC-IDS2018-v2/...                │
│ Target Dataset    │ NF-CSE-CIC-IDS2018-v2                                   │
│ Method            │ PseudoLabelling (K=2)                                   │
│ Training Epochs   │ 20                                                       │
│ Pseudo-Labeling   │ Yes                                                      │
│ Cluster Count     │ 2                                                        │
│ Experiment Config │ NF-CSE-CIC-IDS2018-v2_N|PseudoLabelling|Cluster_2       │
│ File Size         │ 0.1 MB                                                   │
│ Training Type     │ ProperTraining                                           │
└───────────────────┴──────────────────────────────────────────────────────────┘
```

## 🔧 Filtering Options

### **Available Filters**
1. **Dataset Filter**: Filter by target domain
   - `NF-CSE-CIC-IDS2018-v2`
   - `NF-ToN-IoT-v2`
   - `NF-UNSW-NB15-v2`

2. **Epochs Filter**: Filter by training duration
   - `20` (standard training)
   - `50` (extended training)

3. **Method Filter**: Filter by learning approach
   - `PseudoLabelling` (current available method)

### **Filter Combinations**
```bash
# Interactive filter examples:
Filter by dataset? → NF-CSE-CIC-IDS2018-v2
Filter by epochs? → 20
Filter by method? → [Enter for all]

# Result: Shows only 20-epoch models for CSE-CIC-IDS2018 dataset
```

## 🚀 Evaluation Types

### **1. Single Dataset Evaluation**
- Tests model on one specific dataset
- Detailed performance analysis
- Attack type breakdown
- Confidence analysis

```bash
python interactive_evaluation.py --evaluation-type single
```

### **2. Comprehensive Domain Generalization**
- Tests model across ALL 4 datasets
- Domain generalization analysis
- Statistical comparisons
- Research-grade JSON export

```bash
python interactive_evaluation.py --evaluation-type comprehensive
```

### **3. All Datasets Summary**
- Basic performance across all datasets
- Quick overview of model capabilities
- Summary statistics

```bash
python interactive_evaluation.py --evaluation-type all-datasets
```

## 📈 Output and Results

### **Console Output Features**
- 🎨 **Rich formatting**: Color-coded tables and panels
- 📊 **Progress indicators**: Real-time evaluation progress
- 🔍 **Detailed metrics**: Accuracy, F1-score, AUC-ROC, confusion matrices
- 🎯 **Attack analysis**: Distribution of attack types in data

### **JSON Export Capability**
```bash
# Auto-generated filenames
--export-json results/comprehensive_NF_CSE_CIC_IDS2018_v2_20ep.json

# Custom filename
--export-json my_custom_results.json
```

### **Export Format**
```json
{
  "config": {
    "model_name": "MoMLNIDS",
    "target_domain": "NF-CSE-CIC-IDS2018-v2",
    "evaluation_timestamp": "2024-01-15T10:30:00",
    "model_config": {...}
  },
  "evaluation_results": {
    "NF-CSE-CIC-IDS2018-v2": {
      "metrics": {
        "accuracy": 0.8542,
        "f1_score": 0.8421,
        "auc_roc": 0.9156
      },
      "confusion_matrix": {...},
      "confidence_analysis": {...}
    }
  }
}
```

## 💡 Best Practices

### **Recommended Workflow**
1. **Start with auto-select** for quick evaluations:
   ```bash
   python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2"
   ```

2. **Use interactive mode** for model exploration:
   ```bash
   python interactive_evaluation.py
   ```

3. **Choose comprehensive evaluation** for research:
   ```bash
   # Select comprehensive type when prompted
   ```

4. **Export results** for documentation:
   ```bash
   # Always specify --export-json or use auto-generated names
   ```

### **Model Selection Strategy**
- **🏆 Best models**: Use for final performance evaluation
- **📊 Final models**: Use for understanding training progression
- **Different clusters**: Compare K=2,3,4,5,6 for optimal clustering
- **Different epochs**: Compare 20 vs 50 epoch training effects

## 🔍 Advanced Features

### **Auto-Selection Logic**
1. Searches for models matching dataset name
2. Prioritizes `model_best.pt` files
3. Falls back to highest numbered checkpoint
4. Displays selection details before proceeding

### **Configuration Detection Patterns**
```python
# Cluster detection patterns:
- "Cluster_4" → K=4
- "PseudoLabelling|Cluster_2" → K=2  
- Numbers in range 2-20 → Potential cluster count

# Method detection:
- "PseudoLabelling" → PseudoLabelling method
- "SingleLayer" → SingleLayer method
- Path structure analysis for experiment type
```

### **Model Filtering Algorithm**
```python
# For each (dataset, method, epochs, clusters) combination:
1. Find all models matching configuration
2. Select best model (model_best.pt) if available
3. Select final model (highest model_N.pt number)
4. Add both to filtered list
5. Sort by dataset → method → epochs → clusters
```

## 🆘 Troubleshooting

### **Common Issues**

**No models found:**
```bash
❌ No trained models found in ProperTraining directories
```
**Solution**: Ensure models exist in `ProperTraining/` or `ProperTraining50Epoch/` directories

**Empty filtered results:**
```bash
❌ No models found matching the criteria
```
**Solution**: Adjust filters or select "all" for broader search

**Auto-select not working:**
```bash
❌ No models found for dataset: [NAME]
```
**Solution**: Check exact dataset name spelling and available datasets

### **Performance Tips**
- Use smaller `--num-samples` for faster evaluation
- Use `single` evaluation type for quick tests
- Use `comprehensive` for final analysis only
- Export results to avoid re-running evaluations

## 🎉 Quick Start

**Fastest way to get started:**
```bash
# One-liner for complete evaluation
python interactive_evaluation.py \
    --auto-select "NF-CSE-CIC-IDS2018-v2" \
    --evaluation-type comprehensive \
    --num-samples 100

# Interactive exploration
python interactive_evaluation.py
# Then follow the prompts!
```

This enhanced system transforms the complex task of model selection into an intuitive, visual process while maintaining full access to the comprehensive evaluation capabilities of the MoMLNIDS framework.