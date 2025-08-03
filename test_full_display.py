#!/usr/bin/env python3
"""Test full enhanced training display."""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.skripsi_code.TrainEval.TrainEval import (
    setup_dynamic_training_display,
    update_epoch_progress,
    update_dynamic_display,
    advance_overall_progress,
    finish_dynamic_display,
    reset_metrics_tracking,
    display_training_summary,
)


def test_full_enhanced_display():
    """Test the complete enhanced training display with all metrics."""
    print("🚀 Testing Full Enhanced Training Display")
    print("=" * 50)

    # Reset metrics tracking
    reset_metrics_tracking()

    # Setup display
    live_display, progress, overall_task_id, epoch_task_id = (
        setup_dynamic_training_display(
            num_epochs=2, experiment_name="Enhanced Test Training"
        )
    )

    if live_display:
        live_display.start()

    try:
        for epoch in range(2):
            # Start epoch
            total_batches = 5
            update_epoch_progress("start", epoch=epoch, total_batches=total_batches)

            # Simulate training batches
            for batch in range(total_batches):
                # Simulate ALL your actual metrics during training
                train_metrics = {
                    "loss_class": 0.8 - epoch * 0.1 - batch * 0.02,
                    "loss_domain": 1.2 - epoch * 0.15 - batch * 0.03,
                    "loss_entropy": 0.6 - epoch * 0.05 - batch * 0.01,
                    "acc_class": 0.4 + epoch * 0.1 + batch * 0.03,
                    "acc_domain": 0.3 + epoch * 0.08 + batch * 0.02,
                    "f1": 0.35 + epoch * 0.12 + batch * 0.025,
                    "precision": 0.38 + epoch * 0.11 + batch * 0.02,
                    "recall": 0.33 + epoch * 0.13 + batch * 0.027,
                    "mcc": 0.1 + epoch * 0.15 + batch * 0.03,
                    "auroc": 0.55 + epoch * 0.08 + batch * 0.02,
                }

                # Update batch progress
                update_epoch_progress(
                    "batch",
                    epoch=epoch,
                    total_epochs=2,
                    total_batches=total_batches,
                    current_batch=batch,
                    train_metrics=train_metrics,
                )

                time.sleep(0.3)  # Simulate training time

            # Complete training for this epoch
            update_epoch_progress(
                "training_done", epoch=epoch, train_metrics=train_metrics
            )

            # Simulate evaluation metrics with ALL metrics
            val_metrics = {
                "loss": train_metrics["loss_class"] + 0.1,
                "accuracy": train_metrics["acc_class"] - 0.05,
                "f1": train_metrics["f1"] - 0.03,
                "precision": train_metrics["precision"] - 0.04,
                "recall": train_metrics["recall"] - 0.02,
                "auroc": train_metrics["auroc"] - 0.02,
                "mcc": train_metrics["mcc"] - 0.05,
            }

            test_metrics = {
                "loss": val_metrics["loss"] + 0.05,
                "accuracy": val_metrics["accuracy"] - 0.03,
                "f1": val_metrics["f1"] - 0.02,
                "precision": val_metrics["precision"] - 0.01,
                "recall": val_metrics["recall"] - 0.01,
                "auroc": val_metrics["auroc"] - 0.03,
                "mcc": val_metrics["mcc"] - 0.02,
            }

            # Update dynamic display with ALL metrics
            update_dynamic_display(
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                epoch=epoch,
                num_epoch=2,
                best_accuracy=max(0.45, val_metrics["accuracy"]),
                best_epoch=epoch,
                clustering_info=" | Enhanced Mode",
            )

            # Complete epoch
            update_epoch_progress(
                "evaluation_done",
                epoch=epoch,
                total_batches=total_batches,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
            )

            # Advance overall progress
            advance_overall_progress()

            time.sleep(1)  # Pause between epochs

    finally:
        # Finish display
        finish_dynamic_display()

    # Show summary
    display_training_summary(
        experiment_dir="enhanced_test_experiment",
        best_accuracy=0.75,
        best_epoch=1,
        test_accuracy=0.68,
        wandb_enabled=False,
    )

    print("\n✅ Complete enhanced display test finished!")
    print("🎨 Features tested:")
    print("   ✅ Enhanced color coding (bright green ↗, bright red ↘, yellow →)")
    print("   ✅ All metrics: LClass, LDomain, LEntropy, AccClass, AccDomain")
    print("   ✅ Performance metrics: F1, Precision, Recall, ROC-AUC, MCC")
    print("   ✅ Val/Test losses in LClass row")
    print("   ✅ Real-time progress bar updates")
    print("   ✅ Trend indicators and color transitions")


if __name__ == "__main__":
    test_full_enhanced_display()
