# MoMLNIDS Textual TUI - Usage Guide

A modern, interactive Terminal User Interface (TUI) for exploring MoMLNIDS experiment results using Textual.

## 🚀 **What is the Textual TUI?**

The Textual TUI provides a **sophisticated terminal-based graphical interface** that feels like a desktop application but runs entirely in your terminal. It offers:

- **Modern UI Elements**: Buttons, tabs, trees, tables, modals, progress bars
- **Mouse Support**: Click and interact with elements
- **Keyboard Navigation**: Full keyboard shortcuts and navigation
- **Real-time Updates**: Live data loading and refresh
- **Multi-pane Layout**: Split-screen with tree navigation and details
- **Modal Dialogs**: Pop-up windows for detailed analysis

## 🎯 **Key Features**

### 📱 **Split-Screen Interface**
- **Left Panel**: Experiment tree with hierarchical navigation
- **Right Panel**: Tabbed interface for details (Overview, Performance, Configuration, Analysis)
- **Status Bar**: Progress indicators and experiment count
- **Control Buttons**: Refresh, Compare, Best performers

### 🌳 **Interactive Experiment Tree**
- **📁 Folder Icons**: Training_results, ProperTraining, ProperTraining50Epoch, results
- **🗂️ Dataset Groups**: Organized by NF-UNSW-NB15-v2, NF-CSE-CIC-IDS2018-v2, etc.
- **Performance Indicators**: 
  - 🟢 High performance (≥80% accuracy)
  - 🟡 Medium performance (60-79% accuracy)  
  - 🔴 Low performance (<60% accuracy)
  - 📄 JSON results

### 📊 **Rich Details Panel**
- **Overview Tab**: Configuration summary and basic info
- **Performance Tab**: Best metrics, training progress, sparklines
- **Configuration Tab**: Experiment settings and parameters
- **Analysis Tab**: Detailed statistical analysis

### 🎪 **Modal Windows**
- Press **Enter** on any experiment to open detailed modal
- **Tabbed Content**: Configuration, Performance, Training Progress, Raw Data
- **Data Tables**: Epoch-by-epoch training details
- **Sparkline Charts**: Visual accuracy trends
- **Export Function**: Save individual experiment data

### 📋 **Comparison Screens**
- Select experiment groups and press **'c'** to compare
- **Full-screen Tables**: Side-by-side performance comparison
- **Export Comparisons**: Save comparison data
- **Back Navigation**: Return to main interface

## 🎮 **Controls & Navigation**

### **Keyboard Shortcuts**
- **q** - Quit application
- **r** - Refresh experiments  
- **c** - Compare selected group/dataset
- **e** - Export all results
- **h** - Show help
- **Enter** - Open detailed modal for selected experiment
- **Arrow Keys** - Navigate experiment tree
- **Tab** - Switch between interface elements

### **Mouse Controls**
- **Click** buttons for actions (Refresh, Compare, Best)
- **Click** tree nodes to select experiments
- **Click** tabs to switch views
- **Scroll** through content

### **Button Functions**
- **🔄 Refresh** - Reload all experiments from directories
- **📊 Compare** - Compare experiments in selected group
- **📈 Best** - Show top 10 best performing experiments

## 🚀 **Getting Started**

### **Run the TUI**
```bash
# Using the sk_kc environment 
cd /home/dzakirm/Research/MoMLNIDS
sk_kc/bin/python momlnids_results_tui.py

# Or if textual is globally available
python momlnids_results_tui.py
```

### **Basic Workflow**
1. **Launch TUI** - Interface loads with experiment discovery
2. **Navigate Tree** - Use arrows to explore experiment groups
3. **Select Experiments** - Click or use Enter for details
4. **View Details** - Switch between tabs for different views
5. **Compare Groups** - Select folder and press 'c'
6. **Export Data** - Use 'e' to save all results

## 🎯 **What You'll See**

### **Main Interface**
```
┌──────────────────────┐  ┌──────────────────────────────────────────────────┐
│📊 MoMLNIDS           │  │ Overview  Performance  Configuration  Analysis   │
│Experiments           │  │━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
│                      │  │Select an experiment to view details             │
│ ┌──────────────────┐ │  │                                                  │
│ │✅ Loaded 35      │ │  │                                                  │
│ │experiments from 4│ │  │                                                  │
│ └──────────────────┘ │  │                                                  │
│                      │  │                                                  │
│🔄 Refresh 📊 Compare │  │                                                  │
│▼ Experiments         │  │                                                  │
│├── ▶ 📁 Training_res │  │                                                  │
│├── ▶ 📁 ProperTrain  │  │                                                  │
│├── ▶ 📁 ProperTrain50│  │                                                  │
│└── ▶ 📁 results (2)  │  │                                                  │
└──────────────────────┘  └──────────────────────────────────────────────────┘
```

### **Expanded Tree View**
```
▼ 📁 Training_results (15)
  ├── ▶ 🗂️ UNSW-NB15 (5)
  │   ├── 🟢 NF-UNSW-NB15_NSingleLayer|1-N (Acc: 0.948)
  │   ├── 🟢 NF-UNSW-NB15_NSingleLayer|DomainWeight (Acc: 0.954)
  │   └── 🟢 NF-UNSW-NB15_NSingleLayer|No-Weighting (Acc: 0.944)
  ├── ▶ 🗂️ CSE-CIC-IDS2018 (5)  
  │   ├── 🔴 NF-CSE-CIC-IDS2018_NSingleLayer|1-N (Acc: 0.323)
  │   └── 🟡 NF-CSE-CIC-IDS2018_NSingleLayer|DomainWeight (Acc: 0.542)
  └── ▶ 🗂️ ToN-IoT (5)
      ├── 🔴 NF-ToN-IoT_NSingleLayer|1-N (Acc: 0.474)
      └── 🔴 NF-ToN-IoT_NSingleLayer|DomainWeight (Acc: 0.465)
```

## 🎉 **Advanced Features**

### **Performance Sparklines**
- Mini-charts showing accuracy trends over epochs
- Visual representation of training progress
- Hover/selection shows exact values

### **Real-time Statistics**
- Live experiment counting and status
- Progress bars during data loading
- Status updates and notifications

### **Export Capabilities**
- Individual experiment JSON export
- Group comparison exports
- Full dataset exports with timestamps

### **Responsive Design**
- Adapts to terminal size
- Scrollable content areas
- Collapsible interface elements

## 🆚 **TUI vs CLI Comparison**

| **Textual TUI** | **Rich CLI** |
|-----------------|--------------|
| ✅ Interactive navigation | ❌ Linear command execution |
| ✅ Real-time updates | ❌ Static output |
| ✅ Mouse + keyboard support | ✅ Keyboard only |
| ✅ Multi-pane interface | ❌ Single view |
| ✅ Modal dialogs | ❌ Inline display |
| ✅ Visual tree navigation | ❌ Text-based lists |
| ✅ Live data loading | ❌ Batch processing |
| ✅ Export from interface | ✅ Command-line export |

## 🔧 **Troubleshooting**

**TUI doesn't start:**
- Ensure textual is installed: `sk_kc/bin/pip list | grep textual`
- Check terminal compatibility (most modern terminals work)

**Performance issues:**
- The TUI loads 35 experiments efficiently
- Use Refresh button if data seems stale
- Tree expansion/collapse manages memory usage

**Display issues:**
- Ensure terminal is at least 80x25 characters
- Some features require color terminal support
- SSH sessions: ensure proper terminal forwarding

The Textual TUI provides a **modern, intuitive interface** for exploring your MoMLNIDS experiments with the power of a desktop application in your terminal! 🚀