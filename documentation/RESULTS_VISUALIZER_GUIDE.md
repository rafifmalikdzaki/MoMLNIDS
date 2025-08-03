# MoMLNIDS Results Visualizer

A comprehensive visualization and analysis tool for MoMLNIDS experiment results using Rich library for beautiful terminal output.

## Features

🎨 **Rich Visualizations**
- Color-coded performance metrics with intuitive styling
- Beautiful tables with proper formatting and alignment
- Progress bars showing model performance across datasets
- Syntax-highlighted model architecture display

📊 **Data Analysis**
- Comprehensive metrics comparison across datasets
- Confusion matrix statistics (TP, TN, FP, FN)
- Class distribution analysis
- Performance assessment with color-coded ratings

📈 **Summary Statistics**
- Training completion status
- Average, best, and worst accuracy across datasets
- Overall performance assessment
- Export capabilities to text format

🖥️ **Multiple Usage Modes**
- Command-line interface with arguments
- Interactive mode for exploration
- Batch processing capabilities

## Installation & Requirements

The tool requires the following Python packages:
```bash
pip install rich numpy
```

## Usage

### Basic Usage
```bash
# Analyze all results in the results directory
python results_visualizer.py

# Analyze results from a specific directory
python results_visualizer.py /path/to/results

# Analyze a specific result file
python results_visualizer.py results/ --file experiment_results.json

# Export summary to text file
python results_visualizer.py results/ --export summary.txt
```

### Interactive Mode
```bash
python results_visualizer.py results/ --interactive
```

Interactive mode provides:
1. Show all results
2. Show specific file
3. Export summary
4. List available files
5. Quit

### Demo Mode
```bash
python demo_visualizer.py
```

## Output Examples

### Performance Metrics Table
```
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ Dataset             ┃ Accuracy ┃ Precision ┃ Recall ┃ F1-Score ┃ AUC-ROC ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ BoT-IoT             │   0.497  │   0.497   │ 0.497  │  0.497   │  0.498  │
│ CSE-CIC-IDS2018     │   0.501  │   0.501   │ 0.501  │  0.501   │  0.516  │
└─────────────────────┴──────────┴───────────┴────────┴──────────┴─────────┘
```

### Performance Bars
```
BoT-IoT
██████████████░░░░░░░░░░░░░░░░ 49.7%

CSE-CIC-IDS2018
███████████████░░░░░░░░░░░░░░░ 50.1%
```

### Configuration Summary
```
╭─────────────── Experiment Configuration ───────────────╮
│  Model                 MoMLNIDS                       │
│  Batch Size            128                            │
│  Learning Rate         0.01                           │
│  Epochs                10                             │
│  Random Seed           42                             │
│  Datasets              NF-BoT-IoT-v2                  │
│                        NF-CSE-CIC-IDS2018-v2          │
╰────────────────────────────────────────────────────────╯
```

## File Structure

```
results_visualizer.py    # Main visualization tool
demo_visualizer.py       # Demo script
experiment_summary.txt   # Generated text export
results/                 # Results directory
├── quick_test/
│   └── experiment_results.json
```

## Customization

The tool can be easily extended to:
- Support additional metrics
- Add new visualization types
- Export to different formats (CSV, HTML, etc.)
- Include statistical significance tests
- Add trend analysis for multiple experiments

## API Usage

```python
from results_visualizer import MoMLNIDSResultsAnalyzer

# Initialize analyzer
analyzer = MoMLNIDSResultsAnalyzer("results")

# Load results
results = analyzer.load_results()

# Display analysis
analyzer.display_results()

# Export summary
analyzer.export_summary("my_summary.txt")
```

## Command Line Arguments

- `results_dir`: Path to results directory (default: "results")
- `--file, -f`: Specific result file to analyze
- `--export, -e`: Export summary to text file
- `--interactive, -i`: Interactive mode for exploring results

## Color Coding

The tool uses intuitive color coding:
- 🟢 **Green**: Excellent performance (≥80%)
- 🟡 **Yellow**: Good performance (60-79%)
- 🔴 **Red**: Needs improvement (<60%)

## Tips

1. Use the interactive mode to explore multiple result files
2. Export summaries for documentation and reporting
3. The tool automatically handles different result file formats
4. Performance bars give quick visual assessment
5. Model architecture view helps understand the network structure

## Troubleshooting

- Ensure Rich library is installed: `pip install rich`
- Check that result files are valid JSON format
- Verify file paths are correct
- For large result files, the tool may take a moment to load