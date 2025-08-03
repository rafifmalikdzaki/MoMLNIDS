#!/usr/bin/env python3
"""Test table creation with all enhanced metrics and color coding."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.skripsi_code.TrainEval.TrainEval import create_dynamic_metrics_table
from rich.console import Console


def test_enhanced_table():
    """Test the enhanced metrics table creation with all metrics and colors."""
    train_metrics = {
        "loss_class": 0.7550,
        "loss_domain": 0.8576,
        "loss_entropy": 0.6292,
        "acc_class": 0.5340,
        "acc_domain": 0.6774,
        "f1": 0.5371,
        "precision": 0.5439,
        "recall": 0.5340,
        "mcc": 0.2156,
        "auroc": 0.6892,
    }

    val_metrics = {
        "loss": 0.8100,  # This should show in LClass row
        "accuracy": 0.5889,
        "f1": 0.5757,
        "precision": 0.6100,
        "recall": 0.5500,
        "mcc": 0.2456,
        "auroc": 0.7123,
    }

    test_metrics = {
        "loss": 0.8500,  # This should show in LClass row
        "accuracy": 0.6419,
        "f1": 0.6970,
        "precision": 0.7200,
        "recall": 0.6800,
        "mcc": 0.3201,
        "auroc": 0.7456,
    }

    console = Console()
    print("🎨 Testing Enhanced Metrics Table with Color Coding & Additional Metrics")
    print("=" * 80)

    table = create_dynamic_metrics_table(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        epoch=0,
        num_epoch=20,
        best_accuracy=0.6419,
        best_epoch=0,
        clustering_info="",
    )

    if table:
        console.print(table)
        print("\n✅ Table Features:")
        print(
            "   🎨 Enhanced color coding (bright green ↗, bright red ↘, yellow →, blue ●)"
        )
        print("   📊 Added MCC and ROC-AUC metrics")
        print("   🔗 Val/Test losses shown in LClass row")
        print("   ✨ Improved styling and visual hierarchy")
    else:
        print("❌ Table creation failed")


if __name__ == "__main__":
    test_enhanced_table()
