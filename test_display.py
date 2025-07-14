#!/usr/bin/env python3
"""Test the enhanced training display with sample metrics."""

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


def test_enhanced_display():
    """Test the enhanced training display with sample metrics."""
    print("🚀 Testing Enhanced Training Display")
    print("=" * 50)

    # Reset metrics tracking
    reset_metrics_tracking()

    # Setup display
    live_display, progress, overall_task_id, epoch_task_id = (
        setup_dynamic_training_display(num_epochs=3, experiment_name="Test Training")
    )

    if live_display:
        live_display.start()

    try:
        for epoch in range(3):
            # Start epoch
            total_batches = 10
            update_epoch_progress("start", epoch=epoch, total_batches=total_batches)

            # Simulate training batches
            for batch in range(total_batches):
                # Simulate your actual metrics during training
                train_metrics = {
                    "loss_class": 0.8 - epoch * 0.1 - batch * 0.01,
                    "loss_domain": 1.2 - epoch * 0.15 - batch * 0.02,
                    "loss_entropy": 0.6 - epoch * 0.05 - batch * 0.005,
                    "acc_class": 0.4 + epoch * 0.1 + batch * 0.02,
                    "acc_domain": 0.3 + epoch * 0.08 + batch * 0.015,
                    "f1": 0.35 + epoch * 0.12 + batch * 0.018,
                    "precision": 0.38 + epoch * 0.11 + batch * 0.016,
                    "recall": 0.33 + epoch * 0.13 + batch * 0.017,
                }

                # Update batch progress
                update_epoch_progress(
                    "batch",
                    epoch=epoch,
                    total_epochs=3,
                    total_batches=total_batches,
                    current_batch=batch,
                    train_metrics=train_metrics,
                )

                time.sleep(0.2)  # Simulate training time

            # Complete training for this epoch
            update_epoch_progress(
                "training_done", epoch=epoch, train_metrics=train_metrics
            )

            # Simulate evaluation metrics
            val_metrics = {
                "loss": train_metrics["loss_class"] + 0.1,
                "accuracy": train_metrics["acc_class"] - 0.05,
                "f1": train_metrics["f1"] - 0.03,
                "precision": train_metrics["precision"] - 0.04,
                "recall": train_metrics["recall"] - 0.02,
                "auroc": 0.65 + epoch * 0.08,
                "mcc": 0.25 + epoch * 0.1,
            }

            test_metrics = {
                "loss": val_metrics["loss"] + 0.05,
                "accuracy": val_metrics["accuracy"] - 0.03,
                "f1": val_metrics["f1"] - 0.02,
                "precision": val_metrics["precision"] - 0.01,
                "recall": val_metrics["recall"] - 0.01,
                "auroc": val_metrics["auroc"] - 0.05,
                "mcc": val_metrics["mcc"] - 0.03,
            }

            # Update dynamic display
            update_dynamic_display(
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                epoch=epoch,
                num_epoch=3,
                best_accuracy=max(0.45, val_metrics["accuracy"]),
                best_epoch=epoch,
                clustering_info=" | Test Mode",
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
        experiment_dir="test_experiment",
        best_accuracy=0.65,
        best_epoch=2,
        test_accuracy=0.58,
        wandb_enabled=False,
    )

    print("\n✅ Enhanced display test completed!")


if __name__ == "__main__":
    test_enhanced_display()
