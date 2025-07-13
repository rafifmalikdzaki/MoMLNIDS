# MoMLNIDS Prediction Demo Documentation

This repository contains comprehensive prediction demonstration tools for the MoMLNIDS (Multi-domain Machine Learning Network Intrusion Detection System) model.

## Overview

The MoMLNIDS system is designed for domain generalization in network intrusion detection, where models trained on source domains can effectively detect attacks in unseen target domains. The prediction demos showcase this capability using real network flow data.

## Available Demo Scripts

### 1. Basic Prediction Demo (`prediction_demo.py`)
A flexible prediction tool supporting both synthetic and real data inputs.

**Features:**
- Per-sample and per-batch prediction
- Synthetic data generation for testing
- CSV input/output support
- Multiple prediction modes (demo, single, batch)
- Rich terminal interface with confidence scores

**Usage Examples:**
```bash
# Demo with synthetic data
python prediction_demo.py --model-path "ProperTraining/model.pt" --mode demo --num-samples 5

# Interactive single prediction
python prediction_demo.py --model-path "ProperTraining/model.pt" --mode single

# Batch prediction from CSV
python prediction_demo.py --model-path "ProperTraining/model.pt" --mode batch --input-file data.csv
```

### 2. Automatic Real Data Demo (`auto_prediction_demo.py`)
Advanced demonstration tool that automatically loads and processes real network intrusion data from parquet files.

**Features:**
- Automatic dataset discovery from `src/data/parquet/`
- Real network flow data processing (39 features)
- Comprehensive performance analysis
- Multi-dataset comparison capability
- Domain generalization evaluation
- Attack type distribution analysis

**Usage Examples:**
```bash
# Single dataset evaluation
python auto_prediction_demo.py --model-path "ProperTraining/NF-CSE-CIC-IDS2018-v2/model_best.pt" --dataset "NF-CSE-CIC-IDS2018-v2" --num-samples 500

# All datasets comparison
python auto_prediction_demo.py --model-path "ProperTraining/NF-CSE-CIC-IDS2018-v2/model_best.pt" --all-datasets --num-samples 200

# Export results to JSON
python auto_prediction_demo.py --model-path "ProperTraining/model.pt" --dataset "NF-ToN-IoT-v2" --export-json results.json
```

## Model Architecture

The MoMLNIDS model consists of three main components:

1. **Feature Extractor Layer (DGFeatExt)**: Extracts domain-invariant features
   - Input: 39-dimensional network flow features
   - Hidden layers: [64, 32, 16, 10] (configurable)
   - Batch normalization and dropout for regularization

2. **Domain Classifier**: Adversarial component for domain adaptation
   - Gradient reversal layer for domain confusion
   - Classifies source domains (typically 3-4 domains)
   - Helps learn domain-invariant representations

3. **Label Classifier**: Binary classification for intrusion detection
   - Output: Benign (0) or Malicious (1)
   - Single layer or multi-layer configuration

## Dataset Information

### Available Datasets
The system supports multiple network intrusion detection datasets:

- **NF-CSE-CIC-IDS2018-v2**: Canadian Institute for Cybersecurity dataset
- **NF-ToN-IoT-v2**: Telemetry dataset of IoT and IIoT devices
- **NF-UNSW-NB15-v2**: University of New South Wales dataset
- **NF-BoT-IoT-v2**: Botnet and IoT dataset

### Data Format
Each dataset contains:
- **45 total columns**: Network flow features + labels
- **39 feature columns** (indices 4-42): Numerical network flow statistics
- **Label column**: Binary classification (0=Benign, 1=Malicious)
- **Attack column**: Specific attack type labels

### Feature Set (39 dimensions)
Network flow features include:
- Packet statistics (IN_BYTES, OUT_BYTES, IN_PKTS, OUT_PKTS)
- Timing information (FLOW_DURATION_MILLISECONDS, DURATION_IN/OUT)
- Protocol information (PROTOCOL, L7_PROTO)
- TCP characteristics (TCP_FLAGS, TCP_WIN_MAX_IN/OUT)
- Packet size distributions (NUM_PKTS_*_BYTES categories)
- Additional protocol-specific features (DNS, ICMP, FTP)

## Domain Generalization Setup

### Training Configuration
- **Source Domains**: Multiple datasets used for training (e.g., NF-BoT-IoT-v2, NF-UNSW-NB15-v2)
- **Target Domain**: Single dataset for evaluation (specified in model directory name)
- **Domain Adaptation**: Adversarial training with gradient reversal
- **Objective**: Learn features that are discriminative for intrusion detection but invariant across domains

### Model Naming Convention
Model paths indicate the target domain:
```
ProperTraining/NF-CSE-CIC-IDS2018-v2/  # Target domain: CSE-CIC-IDS2018
ProperTraining/NF-ToN-IoT-v2/           # Target domain: ToN-IoT
ProperTraining50Epoch/NF-UNSW-NB15-v2/  # Target domain: UNSW-NB15
```

## Performance Metrics

### Classification Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

### Confidence Analysis
- **Overall Average**: Mean confidence across all predictions
- **Correct Predictions Average**: Mean confidence for correct classifications
- **Incorrect Predictions Average**: Mean confidence for incorrect classifications
- **Min/Max Confidence**: Range of prediction confidence scores

### Domain Analysis
- **Domain Prediction**: Model's domain classification capability
- **Domain Confidence**: Confidence in domain predictions
- **Cross-domain Performance**: Evaluation on unseen target domains

## Expected Performance

### Typical Results
Domain generalization performance varies by dataset:

| Target Dataset | Accuracy Range | F1-Score Range | Notes |
|---------------|----------------|----------------|-------|
| NF-CSE-CIC-IDS2018-v2 | 30-40% | 0.10-0.15 | Mixed attack types |
| NF-ToN-IoT-v2 | 30-45% | 0.15-0.50 | IoT-focused attacks |
| NF-UNSW-NB15-v2 | 35-45% | 0.08-0.12 | Diverse attack vectors |
| NF-BoT-IoT-v2 | 2-30% | 0.04-0.08 | Challenging botnet detection |

### Performance Factors
- **Domain Shift**: Larger differences between source and target domains reduce performance
- **Attack Type Overlap**: Similar attack patterns across domains improve generalization
- **Data Quality**: Clean, well-preprocessed data enhances model performance
- **Model Capacity**: Appropriate model size for the complexity of the domain adaptation task

## Attack Type Analysis

### Common Attack Categories
- **DDoS/DoS**: Distributed/Denial of Service attacks
- **Web Attacks**: XSS, SQL injection, web scanning
- **Brute Force**: SSH, FTP password attacks
- **Reconnaissance**: Network scanning and enumeration
- **Botnet**: Automated malicious network activity
- **Infiltration**: Advanced persistent threat techniques

### Dataset-Specific Attacks
Each dataset contains different attack distributions:
- **CSE-CIC-IDS2018**: DDOS, DoS variants, SSH/FTP brute force
- **ToN-IoT**: XSS, injection, scanning, DDoS
- **UNSW-NB15**: Generic malware, backdoors, fuzzers, exploits
- **BoT-IoT**: Primarily DDoS, DoS, and reconnaissance

## Configuration Options

### Model Parameters
```python
model_config = {
    "input_nodes": 39,           # Network flow features
    "hidden_nodes": [64, 32, 16, 10],  # Feature extractor layers
    "classifier_nodes": [64, 32, 16],   # Domain classifier layers
    "num_domains": 4,            # Number of source domains
    "num_class": 2,              # Binary classification
    "single_layer": True         # Label classifier architecture
}
```

### Demo Parameters
```bash
--model-path          # Path to trained model file
--data-path           # Path to parquet data directory
--dataset             # Specific dataset name
--num-samples         # Number of samples to evaluate
--device              # cpu/cuda/auto
--show-samples        # Display individual predictions
--all-datasets        # Evaluate on all available datasets
--export-json         # Export results to JSON format
```

## Output Formats

### Terminal Display
Rich terminal interface with:
- Model configuration tables
- Performance metrics summary
- Confusion matrix visualization
- Sample predictions display
- Attack type distribution charts

### JSON Export
Structured results compatible with experiment tracking:
```json
{
  "config": {
    "model_name": "MoMLNIDS",
    "target_domain": "NF-CSE-CIC-IDS2018-v2",
    "num_samples": 500
  },
  "evaluation_results": {
    "target_domain": {
      "metrics": {
        "accuracy": 0.347,
        "precision": 0.070,
        "recall": 0.318,
        "f1_score": 0.114,
        "auc_roc": 0.52
      },
      "predictions": "[0 1 0 1 ...]",
      "probabilities": "[[0.93 0.07] ...]",
      "true_labels": "[0 1 0 1 ...]"
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Verify model path exists
   - Check device compatibility (CPU/GPU)
   - Ensure model architecture matches saved state

2. **Data Loading Issues**
   - Confirm parquet files exist in `src/data/parquet/`
   - Check file permissions and accessibility
   - Verify data format consistency

3. **Performance Issues**
   - Reduce `--num-samples` for faster execution
   - Use CPU device for smaller datasets
   - Close other GPU-intensive applications

4. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path includes `src/`
   - Verify required packages (torch, polars, rich, click)

### Memory Management
- Large datasets may require batch processing
- Adjust buffer sizes in dataset loaders
- Monitor GPU memory usage during inference

## Future Enhancements

### Planned Features
- Real-time streaming prediction capability
- Advanced visualization dashboards
- Model performance comparison tools
- Automated hyperparameter optimization
- Integration with network monitoring systems

### Extensibility
The demo framework supports:
- Additional dataset formats (CSV, HDF5, etc.)
- Custom model architectures
- New evaluation metrics
- Plugin-based analysis modules

## Citation

If you use this MoMLNIDS demonstration framework in your research, please cite:

```bibtex
@software{momlnids_demo,
  title={MoMLNIDS Prediction Demonstration Framework},
  author={Research Team},
  year={2024},
  url={https://github.com/your-repo/MoMLNIDS}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, issues, or contributions, please contact the development team or open an issue in the repository.