# MoMLNIDS Interactive Evaluation Guide

## 🎯 **Complete MoMLNIDS Evaluation Suite**

You now have a powerful set of tools for evaluating your MoMLNIDS models without needing to remember complex model paths!

## 📊 **Available Tools**

### 1. **🚀 Interactive Model Selection** - `interactive_evaluation.py`
**The main interface that lets you browse and select models visually**

```bash
# Interactive mode - browse and select models from a list
python interactive_evaluation.py

# Quick auto-selection by dataset name
python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2"
python interactive_evaluation.py --auto-select "NF-ToN-IoT-v2"
python interactive_evaluation.py --auto-select "NF-UNSW-NB15-v2"

# Choose evaluation type
python interactive_evaluation.py --evaluation-type comprehensive  # Best option!
python interactive_evaluation.py --evaluation-type single
python interactive_evaluation.py --evaluation-type all-datasets
```

### 2. **🌐 Comprehensive Domain Generalization** - `comprehensive_evaluation.py`
**Ultimate domain generalization analysis across ALL datasets**

```bash
# Test one model across all datasets
python comprehensive_evaluation.py --model-path "ProperTraining/NF-CSE-CIC-IDS2018-v2/model_best.pt" --export-json results.json
```

### 3. **📊 Single Dataset Evaluation** - `auto_prediction_demo.py`
**Detailed analysis on specific datasets**

```bash
# Single dataset with real data
python auto_prediction_demo.py --model-path "model.pt" --dataset "NF-CSE-CIC-IDS2018-v2" --export-json results.json

# All datasets summary  
python auto_prediction_demo.py --model-path "model.pt" --all-datasets
```

### 4. **🔧 Basic Testing** - `prediction_demo.py`
**Quick testing with synthetic data**

```bash
# Quick synthetic test
python prediction_demo.py --model-path "model.pt" --mode demo --num-samples 5
```

## 🎯 **Recommended Workflow**

### **Option 1: Quick & Easy (Auto-Select)**
```bash
# Just specify the dataset name - automatically finds best model
python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2" --evaluation-type comprehensive
```

### **Option 2: Interactive Selection**
```bash
# Browse and select models interactively
python interactive_evaluation.py --evaluation-type comprehensive
```

### **Option 3: Direct Comprehensive Evaluation**
```bash
# If you know the exact model path
python comprehensive_evaluation.py --model-path "ProperTraining/[dataset]/[experiment]/model_best.pt"
```

## 📋 **What Each Tool Does**

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| **interactive_evaluation.py** | 🎯 Browse & select models | Interactive/Auto-select | Runs chosen evaluation |
| **comprehensive_evaluation.py** | 🌐 Test ALL datasets | Model path | Complete domain analysis |
| **auto_prediction_demo.py** | 📊 Single dataset focus | Model + dataset | Detailed single analysis |
| **prediction_demo.py** | 🔧 Quick testing | Model path | Basic prediction test |

## 🏆 **Key Features**

### ✅ **No More Manual Path Typing!**
- **247 models discovered automatically**
- **Browse by dataset, epochs, experiment**
- **Auto-select best models**
- **Rich terminal interface**

### 🎯 **Smart Model Selection**
- Filters by target dataset
- Shows model size and configuration
- Prefers `model_best.pt` files
- Displays detailed model information

### 🌐 **Comprehensive Analysis**
- **Domain generalization evaluation**
- **Target vs cross-domain performance**
- **Statistical analysis across domains**
- **Generalization quality assessment**

### 📊 **Rich Output Formats**
- **Beautiful terminal tables**
- **JSON export compatible with your format**
- **Individual dataset results**
- **Comprehensive summary statistics**

## 🚀 **Quick Start Examples**

```bash
# 1. Ultimate comprehensive evaluation (recommended!)
python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2" --evaluation-type comprehensive

# 2. Test a specific model on ToN-IoT dataset
python interactive_evaluation.py --auto-select "NF-ToN-IoT-v2" --evaluation-type single

# 3. Quick synthetic data test
python prediction_demo.py --model-path "ProperTraining/[any-model]/model_best.pt" --mode demo

# 4. Browse all models interactively
python interactive_evaluation.py
```

## 🎯 **Your Models Discovered**
- **247 trained models found**
- **4 target datasets**: NF-CSE-CIC-IDS2018-v2, NF-ToN-IoT-v2, NF-UNSW-NB15-v2, NF-BoT-IoT-v2
- **2 training variants**: ProperTraining (20 epochs), ProperTraining50Epoch (50 epochs)
- **Multiple experiment configurations per dataset**

## 📈 **Results You Get**

### **Domain Generalization Analysis**
- Target domain performance baseline
- Cross-domain generalization capability  
- Performance drop analysis
- Generalization quality rating (Excellent/Good/Moderate/Poor)

### **Comprehensive Metrics**
- Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Confusion matrices
- Confidence score analysis
- Attack type distributions

### **Export Formats**
- Individual dataset JSON files
- Comprehensive analysis JSON
- Compatible with your experiment_results.json format

## 🎉 **You're All Set!**

**No more typing long model paths!** Just use:
```bash
python interactive_evaluation.py --auto-select "NF-CSE-CIC-IDS2018-v2"
```

And you'll get a complete domain generalization analysis across all your datasets! 🚀