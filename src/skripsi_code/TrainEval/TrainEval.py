import wandb
from torch import nn
from skripsi_code.utils.loss import EntropyLoss, MaximumSquareLoss
import torch
from torchmetrics import (
    F1Score,
    Precision,
    Recall,
    ROC,
    AUROC,
    AveragePrecision,
    MatthewsCorrCoef,
    ConfusionMatrix,
    Specificity,
)
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

# Import click and rich with fallback
try:
    import click

    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False

try:
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        MofNCompleteColumn,
    )
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.layout import Layout
    from rich.align import Align

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Global variables for tracking metrics and live display
_previous_metrics = {}
_progress_instance = None
_console_instance = None
_live_instance = None
_overall_task_id = None
_epoch_task_id = None
_current_table = None


def get_metric_arrow(current_value, previous_value, higher_is_better=True):
    """Get arrow indicating metric direction and color - using high contrast colors."""
    if previous_value is None:
        return "●", "bright_blue"  # First time, neutral - blue is generally safe

    diff = current_value - previous_value
    if abs(diff) < 1e-6:  # No significant change
        return "→", "white"  # Stable - white for neutral

    if higher_is_better:
        if diff > 0:
            return "↗", "bright_green"  # Improved - bright green
        else:
            return "↘", "bright_red"  # Worsened - bright red
    else:  # Lower is better (for loss)
        if diff < 0:
            return "↗", "bright_green"  # Improved (decreased) - bright green
        else:
            return "↘", "bright_red"  # Worsened (increased) - bright red


def create_metrics_table(
    train_metrics,
    val_metrics,
    test_metrics,
    epoch,
    num_epoch,
    best_accuracy,
    best_epoch,
    clustering_info="",
):
    """Create a beautiful metrics table with Rich."""
    if not RICH_AVAILABLE:
        return None

    global _previous_metrics

    # Create table
    table = Table(
        title=f"🚀 Epoch {epoch + 1}/{num_epoch}{clustering_info}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan", width=12)
    table.add_column("Train", justify="center", width=15)
    table.add_column("Val", justify="center", width=15)
    table.add_column("Test", justify="center", width=15)

    # Define metrics to display
    metrics_config = [
        ("Accuracy", "acc_class", "accuracy", "accuracy", True),
        ("F1 Score", "f1", "f1", "f1", True),
        ("AUROC", "auroc", "auroc", "auroc", True),
        ("Precision", "precision", "precision", "precision", True),
        ("Recall", "recall", "recall", "recall", True),
        ("MCC", "mcc", "mcc", "mcc", True),
        ("Loss Class", "loss_class", None, None, False),
        ("Loss Domain", "loss_domain", None, None, False),
        ("Loss Entropy", "loss_entropy", None, None, False),
        ("Loss", None, "loss", "loss", False),
    ]

    for display_name, train_key, val_key, test_key, higher_is_better in metrics_config:
        # Get current values
        train_val = (
            train_metrics.get(train_key) if train_key and train_metrics else None
        )
        val_val = val_metrics.get(val_key) if val_key and val_metrics else None
        test_val = test_metrics.get(test_key) if test_key and test_metrics else None

        # Get previous values for arrows
        prev_key = f"{display_name.lower().replace(' ', '_')}"
        prev_train = _previous_metrics.get(f"train_{prev_key}")
        prev_val = _previous_metrics.get(f"val_{prev_key}")
        prev_test = _previous_metrics.get(f"test_{prev_key}")

        # Create formatted strings with arrows
        def format_metric(value, prev_value, higher_is_better):
            if value is None:
                return Text("N/A", style="dim")

            arrow, color = get_metric_arrow(value, prev_value, higher_is_better)
            return Text(f"{arrow} {value:.4f}", style=color)

        train_text = format_metric(train_val, prev_train, higher_is_better)
        val_text = format_metric(val_val, prev_val, higher_is_better)
        test_text = format_metric(test_val, prev_test, higher_is_better)

        table.add_row(display_name, train_text, val_text, test_text)

        # Update previous metrics
        if train_val is not None:
            _previous_metrics[f"train_{prev_key}"] = train_val
        if val_val is not None:
            _previous_metrics[f"val_{prev_key}"] = val_val
        if test_val is not None:
            _previous_metrics[f"test_{prev_key}"] = test_val

    # Add best model info
    if val_metrics and best_accuracy is not None:
        table.add_section()
        best_text = Text(
            f"🏆 {best_accuracy:.4f} @ Epoch {best_epoch + 1}", style="bold gold1"
        )
        improved_text = Text(
            "💾 New Best!" if val_metrics.get("accuracy", 0) >= best_accuracy else "",
            style="bold green",
        )
        table.add_row("Best Model", "", best_text, improved_text)

    return table


def display_training_progress(
    epoch,
    num_epoch,
    train_metrics,
    val_metrics=None,
    test_metrics=None,
    best_accuracy=None,
    best_epoch=None,
    clustering_info="",
):
    """Display training progress with Rich formatting."""
    if not RICH_AVAILABLE:
        # Fallback to simple text
        if val_metrics and test_metrics:
            print(
                f"📈 Epoch {epoch + 1:2d}/{num_epoch} | Train Acc: {train_metrics.get('acc_class', 0):.4f} | Val Acc: {val_metrics.get('accuracy', 0):.4f} | Test Acc: {test_metrics.get('accuracy', 0):.4f}"
            )
        return

    global _console_instance
    if _console_instance is None:
        _console_instance = Console()

    # Create and display metrics table
    if val_metrics and test_metrics:
        table = create_metrics_table(
            train_metrics,
            val_metrics,
            test_metrics,
            epoch,
            num_epoch,
            best_accuracy,
            best_epoch,
            clustering_info,
        )
        if table:
            _console_instance.print(table)
            _console_instance.print()  # Add spacing


def setup_rich_progress(num_epochs, experiment_name="Training"):
    """Setup Rich progress bar for training."""
    if not RICH_AVAILABLE:
        return None

    global _progress_instance, _console_instance

    if _console_instance is None:
        _console_instance = Console()

    # Create progress bar
    _progress_instance = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=_console_instance,
        expand=True,
    )

    # Add task for epochs
    task_id = _progress_instance.add_task(f"🚀 {experiment_name}", total=num_epochs)

    return _progress_instance, task_id


def update_rich_progress(progress, task_id, advance=1):
    """Update Rich progress bar."""
    if progress and RICH_AVAILABLE:
        progress.advance(task_id, advance)


def finish_rich_progress(progress):
    """Finish Rich progress bar."""
    if progress and RICH_AVAILABLE:
        progress.stop()


def reset_metrics_tracking():
    """Reset metrics tracking for new experiment."""
    global _previous_metrics, _progress_instance, _console_instance, _live_instance
    global _overall_task_id, _epoch_task_id, _current_table
    _previous_metrics = {}
    _progress_instance = None
    _console_instance = None
    _live_instance = None
    _overall_task_id = None
    _epoch_task_id = None
    _current_table = None


def setup_dynamic_training_display(num_epochs, experiment_name="Training"):
    """Setup dynamic Rich display with dual progress bars and live updating table."""
    if not RICH_AVAILABLE:
        return None, None, None, None

    global \
        _progress_instance, \
        _console_instance, \
        _live_instance, \
        _overall_task_id, \
        _epoch_task_id

    if _console_instance is None:
        _console_instance = Console()

    # Create progress bar with colorblind-friendly styling
    _progress_instance = Progress(
        SpinnerColumn(),
        TextColumn("[bold white]{task.description}"),  # White for high contrast
        BarColumn(
            bar_width=40,
            style="green",  # Standard green for incomplete
            complete_style="bright_green",  # Bright green for completed
            finished_style="bold bright_green",  # Bold bright green for finished
        ),
        TaskProgressColumn(show_speed=False),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("[bold yellow]{task.fields[metrics]}"),  # Bold yellow for metrics
        console=_console_instance,
        expand=True,
    )

    # Add overall progress task with more descriptive format
    _overall_task_id = _progress_instance.add_task(
        "Overall", total=num_epochs, metrics=""
    )

    # Add per-epoch task with batches progress
    _epoch_task_id = _progress_instance.add_task(
        "Epoch 0/0", total=100, visible=True, metrics=""
    )

    # Create layout with progress bars and enhanced metrics table
    from rich.layout import Layout

    layout = Layout()
    layout.split_column(Layout(name="progress", size=3), Layout(name="table", ratio=1))

    # Initialize with progress bars
    layout["progress"].update(_progress_instance)
    layout["table"].update(
        Panel(
            "🚀 Starting training...",
            title="📊 Training Metrics",
            border_style="bright_blue",  # Blue border
        )
    )

    # Create live display with higher refresh rate for smooth updates
    _live_instance = Live(layout, console=_console_instance, refresh_per_second=20)

    return _live_instance, _progress_instance, _overall_task_id, _epoch_task_id


def format_metrics_for_progress(
    train_metrics=None, val_metrics=None, test_metrics=None, best_metrics=None
):
    """Format metrics string for progress bar display using your actual metrics."""
    if not train_metrics:
        return ""

    metrics_parts = []

    # Your actual loss metrics
    if "loss_class" in train_metrics:
        loss_class = train_metrics["loss_class"]
        metrics_parts.append(f"LClass: {loss_class:.4f}")

    if "loss_domain" in train_metrics:
        loss_domain = train_metrics["loss_domain"]
        metrics_parts.append(f"LDomain: {loss_domain:.4f}")

    if "loss_entropy" in train_metrics:
        loss_entropy = train_metrics["loss_entropy"]
        metrics_parts.append(f"LEntropy: {loss_entropy:.4f}")

    # Your actual accuracy metrics
    if "acc_class" in train_metrics:
        acc_class = train_metrics["acc_class"]
        metrics_parts.append(f"AccClass: {acc_class:.4f}")

    if "acc_domain" in train_metrics:
        acc_domain = train_metrics["acc_domain"]
        metrics_parts.append(f"AccDomain: {acc_domain:.4f}")

    # Performance metrics - only show if they are meaningful (> 0)
    if "f1" in train_metrics and train_metrics["f1"] > 0:
        f1 = train_metrics["f1"]
        metrics_parts.append(f"F1: {f1:.4f}")

    if "precision" in train_metrics and train_metrics["precision"] > 0:
        precision = train_metrics["precision"]
        metrics_parts.append(f"Precision: {precision:.4f}")

    if "recall" in train_metrics and train_metrics["recall"] > 0:
        recall = train_metrics["recall"]
        metrics_parts.append(f"Recall: {recall:.4f}")

    # Add MCC and AUROC if available
    if "mcc" in train_metrics and train_metrics["mcc"] > -1:  # MCC can be negative
        mcc = train_metrics["mcc"]
        metrics_parts.append(f"MCC: {mcc:.4f}")

    if "auroc" in train_metrics and train_metrics["auroc"] > 0:
        auroc = train_metrics["auroc"]
        metrics_parts.append(f"AUROC: {auroc:.4f}")

    return " | ".join(metrics_parts)


def update_epoch_progress(
    phase,
    progress_percent=None,
    epoch=None,
    total_epochs=None,
    total_batches=None,
    current_batch=None,
    train_metrics=None,
    val_metrics=None,
    test_metrics=None,
):
    """Update the per-epoch progress bar with metrics."""
    if not RICH_AVAILABLE or not _progress_instance:
        return

    global _epoch_task_id, _overall_task_id

    if phase == "start":
        # Reset and show epoch progress with batch info
        _progress_instance.reset(_epoch_task_id)
        epoch_desc = f"Epoch {epoch + 1}" if epoch is not None else "Training"
        total = total_batches if total_batches else 100

        _progress_instance.update(
            _epoch_task_id,
            description=epoch_desc,
            visible=True,
            total=total,
            metrics="",
        )

    elif phase == "batch":
        # Update per-batch progress with metrics
        if current_batch is not None and total_batches is not None:
            epoch_desc = (
                f"Epoch {epoch + 1}/{total_epochs}" if epoch is not None else "Training"
            )
            metrics_str = format_metrics_for_progress(
                train_metrics, val_metrics, test_metrics
            )

            _progress_instance.update(
                _epoch_task_id,
                description=epoch_desc,
                completed=current_batch + 1,
                metrics=metrics_str,
            )

    elif phase == "training_done":
        # Update with training complete metrics
        epoch_desc = f"Epoch {epoch + 1}" if epoch is not None else "Evaluating"
        metrics_str = format_metrics_for_progress(
            train_metrics, val_metrics, test_metrics
        )

        _progress_instance.update(
            _epoch_task_id, description=epoch_desc, metrics=metrics_str
        )

    elif phase == "evaluation_done":
        # Complete epoch with final metrics
        epoch_desc = f"Epoch {epoch + 1}" if epoch is not None else "Complete"
        metrics_str = format_metrics_for_progress(
            train_metrics, val_metrics, test_metrics
        )

        _progress_instance.update(
            _epoch_task_id,
            description=epoch_desc,
            completed=total_batches if total_batches else 100,
            metrics=metrics_str,
        )

        # Update overall progress with summary metrics
        if train_metrics:
            overall_metrics = format_metrics_for_progress(
                train_metrics, val_metrics, test_metrics
            )
            _progress_instance.update(_overall_task_id, metrics=overall_metrics)

    elif phase == "end":
        # Hide epoch progress
        _progress_instance.update(_epoch_task_id, visible=False)

    elif phase == "custom" and progress_percent is not None:
        _progress_instance.update(_epoch_task_id, completed=progress_percent)


def create_dynamic_metrics_table(
    train_metrics,
    val_metrics,
    test_metrics,
    epoch,
    num_epoch,
    best_accuracy,
    best_epoch,
    clustering_info="",
    show_interruption=False,
):
    """Create a dynamic updating metrics table with Rich - compact side-by-side layout."""
    if not RICH_AVAILABLE:
        return None

    global _previous_metrics, _current_table

    title = f"Epoch {epoch + 1}/{num_epoch}{clustering_info}"
    if show_interruption:
        title = f"Training interrupted by user. Checkpoints are saved up to the last finished epoch."

    # Create two compact tables side by side
    from rich.columns import Columns
    from rich.console import Group

    # Table 1: Loss and Accuracy metrics
    table1 = Table(
        show_header=True,
        header_style="bold bright_blue",  # Blue instead of magenta
        border_style="bright_blue",
        box=None,
        padding=(0, 0),
        width=35,  # Fixed narrow width
    )
    table1.add_column(
        "Metric", style="bright_blue bold", width=7
    )  # Blue instead of cyan
    table1.add_column("Train", justify="center", width=8)
    table1.add_column("Val", justify="center", width=8)
    table1.add_column("Test", justify="center", width=8)

    # Table 2: Performance metrics
    table2 = Table(
        show_header=True,
        header_style="bold bright_yellow",  # Yellow instead of green
        border_style="bright_yellow",
        box=None,
        padding=(0, 0),
        width=35,  # Fixed narrow width
    )
    table2.add_column(
        "Metric", style="bright_yellow bold", width=7
    )  # Yellow instead of green
    table2.add_column("Train", justify="center", width=8)
    table2.add_column("Val", justify="center", width=8)
    table2.add_column("Test", justify="center", width=8)
    # Loss and accuracy metrics for table1
    loss_acc_metrics = [
        ("LClass", "loss_class", "loss", "loss", False),
        ("LDomain", "loss_domain", None, None, False),
        ("LEntrop", "loss_entropy", None, None, False),  # Shortened
        ("AccCls", "acc_class", "accuracy", "accuracy", True),  # Shortened
        ("AccDom", "acc_domain", None, None, True),
    ]

    # Performance metrics for table2
    perf_metrics = [
        ("F1", "f1", "f1", "f1", True),
        ("Precis", "precision", "precision", "precision", True),
        ("Recall", "recall", "recall", "recall", True),
        ("AUROC", "auroc", "auroc", "auroc", True),
        ("MCC", "mcc", "mcc", "mcc", True),
    ]

    def add_metrics_to_table(table, metrics_config):
        for (
            display_name,
            train_key,
            val_key,
            test_key,
            higher_is_better,
        ) in metrics_config:
            # Get current values
            train_val = (
                train_metrics.get(train_key) if train_key and train_metrics else None
            )
            val_val = val_metrics.get(val_key) if val_key and val_metrics else None
            test_val = test_metrics.get(test_key) if test_key and test_metrics else None

            # Get previous values for arrows
            prev_key = f"{display_name.lower().replace(' ', '_').replace('-', '_')}"
            prev_train = _previous_metrics.get(f"train_{prev_key}")
            prev_val = _previous_metrics.get(f"val_{prev_key}")
            prev_test = _previous_metrics.get(f"test_{prev_key}")

            # Create formatted strings with colorblind-friendly colors and arrows
            def format_metric(
                value, prev_value, higher_is_better, metric_type="default"
            ):
                if value is None:
                    return Text("N/A", style="dim white")

                arrow, color = get_metric_arrow(value, prev_value, higher_is_better)

                # High contrast color coding
                if metric_type == "loss":
                    if arrow == "↗":  # Loss improved (decreased)
                        color = "bright_green bold"  # Green for improvement
                    elif arrow == "↘":  # Loss worsened (increased)
                        color = "bright_red bold"  # Red for decline
                    else:
                        color = "white"  # White for stable
                elif metric_type == "accuracy":
                    if arrow == "↗":  # Accuracy improved
                        color = "bright_green bold"  # Green for improvement
                    elif arrow == "↘":  # Accuracy decreased
                        color = "bright_red bold"  # Red for decline
                    else:
                        color = "bright_blue"  # Blue for stable
                else:
                    if arrow == "↗":  # General improvement
                        color = "bright_green bold"  # Green for improvement
                    elif arrow == "↘":  # General decline
                        color = "bright_red bold"  # Red for decline
                    else:
                        color = "bright_white"  # White for stable

                # Very compact format - no space between arrow and number
                return Text(f"{arrow}{value:.3f}", style=color)

            # Determine metric type for color coding
            if "loss" in display_name.lower() or display_name == "LClass":
                metric_type = "loss"
            elif "acc" in display_name.lower():
                metric_type = "accuracy"
            else:
                metric_type = "performance"

            train_text = format_metric(
                train_val, prev_train, higher_is_better, metric_type
            )
            val_text = format_metric(val_val, prev_val, higher_is_better, metric_type)
            test_text = format_metric(
                test_val, prev_test, higher_is_better, metric_type
            )

            table.add_row(display_name, train_text, val_text, test_text)

            # Update previous metrics
            if train_val is not None:
                _previous_metrics[f"train_{prev_key}"] = train_val
            if val_val is not None:
                _previous_metrics[f"val_{prev_key}"] = val_val
            if test_val is not None:
                _previous_metrics[f"test_{prev_key}"] = test_val

    # Fill both tables
    add_metrics_to_table(table1, loss_acc_metrics)
    add_metrics_to_table(table2, perf_metrics)

    # Create compact side-by-side layout with minimal spacing
    combined_tables = Columns([table1, table2], equal=False, expand=False)

    # Add best model info if available - compact format
    best_info = ""
    if val_metrics and best_accuracy is not None:
        improved = "💾New!" if val_metrics.get("accuracy", 0) >= best_accuracy else ""
        best_info = f"🏆Best: {best_accuracy:.3f}@E{best_epoch + 1} {improved}"

    # Combine everything into a compact group
    combined_display = Group(
        Text(title, style="bold bright_blue", justify="center"),  # Blue instead of cyan
        combined_tables,
        Text(best_info, style="bold bright_yellow", justify="center")
        if best_info
        else "",  # Yellow instead of gold
    )

    _current_table = combined_display
    return combined_display
    # Just the main tables without best model
    combined_display = Group(
        Text(title, style="bold bright_cyan", justify="center"), "", combined_table
    )
    _current_table = combined_display
    return combined_display


def display_training_summary(
    experiment_dir, best_accuracy, best_epoch, test_accuracy, wandb_enabled=False
):
    """Display training completion summary with colorblind-friendly colors."""
    if not RICH_AVAILABLE:
        print(f"🎉 Training completed!")
        print(
            f"🏆 Best validation accuracy: {best_accuracy:.4f} at epoch {best_epoch + 1}"
        )
        print(f"🎯 Final test accuracy: {test_accuracy:.4f}")
        return

    console = Console()

    # Create summary table with colorblind-friendly colors
    summary_table = Table(
        title="🎉 Training Completed!",
        show_header=True,
        header_style="bold bright_blue",  # Blue instead of green
        title_style="bold bright_blue",  # Blue instead of green
        border_style="bright_blue",  # Blue instead of bright_green
    )
    summary_table.add_column("Metric", style="bright_blue")  # Blue instead of cyan
    summary_table.add_column("Value", style="bold white")

    summary_table.add_row(
        "Best Val Accuracy", f"🏆 {best_accuracy:.4f} @ Epoch {best_epoch + 1}"
    )
    summary_table.add_row("Final Test Accuracy", f"🎯 {test_accuracy:.4f}")
    summary_table.add_row("Experiment Directory", f"📁 {experiment_dir}")

    if wandb_enabled:
        summary_table.add_row("Wandb Integration", "✅ Enabled")
        # Add wandb-style run info if available
        try:
            import wandb

            if wandb.run is not None:
                run_name = wandb.run.name
                run_url = wandb.run.url
                summary_table.add_row("Wandb Run", f"🚀 {run_name}")
                summary_table.add_row("View Run", f"🌐 {run_url}")
        except:
            pass

    # Display with panel using colorblind-friendly colors
    console.print(
        Panel(
            summary_table, title="📊 Training Summary", border_style="bright_blue"
        )  # Blue border
    )

    # Add wandb-style footer
    if wandb_enabled:
        console.print("\nwandb:")
        try:
            import wandb

            if wandb.run is not None:
                console.print(
                    f"wandb: 🚀 View run {wandb.run.name} at: {wandb.run.url}"
                )
                console.print(f"wandb: Find logs at: wandb/run-{wandb.run.id}/logs")
        except:
            pass


def show_interruption_message():
    """Show training interruption message."""
    if not RICH_AVAILABLE:
        print(
            "Training interrupted by user. Checkpoints are saved up to the last finished epoch."
        )
        return

    console = Console()
    console.print(
        Panel(
            "⚠️ Training interrupted by user. Checkpoints are saved up to the last finished epoch.",
            title="🛑 Training Interrupted",
            border_style="yellow",
        )
    )


def update_dynamic_display(
    train_metrics,
    val_metrics=None,
    test_metrics=None,
    epoch=0,
    num_epoch=1,
    best_accuracy=None,
    best_epoch=None,
    clustering_info="",
):
    """Update the dynamic display with new metrics."""
    if not RICH_AVAILABLE or not _live_instance:
        return

    global _current_table

    # Create updated table
    table = create_dynamic_metrics_table(
        train_metrics,
        val_metrics,
        test_metrics,
        epoch,
        num_epoch,
        best_accuracy,
        best_epoch,
        clustering_info,
    )

    if table:
        # Update the table in the layout
        _live_instance.renderable["table"].update(
            Panel(table, title="📊 Training Metrics", border_style="bright_blue")
        )
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", justify="right", width=12)

    # Training metrics breakdown like in your example
    if train_metrics:
        table.add_row(f"Train: Epoch: {epoch + 1}/{num_epoch}", "")

        # Loss metrics
        if "loss_class" in train_metrics:
            table.add_row("LClass", f"{train_metrics['loss_class']:.4f}")
        if "loss_domain" in train_metrics:
            table.add_row("LDomain", f"{train_metrics['loss_domain']:.4f}")
        if "loss_entropy" in train_metrics:
            table.add_row("Loss Entropy", f"{train_metrics['loss_entropy']:.4f}")

        # Accuracy metrics
        if "acc_class" in train_metrics:
            table.add_row("Acc Class", f"{train_metrics['acc_class']:.4f}")
        if "acc_domain" in train_metrics:
            table.add_row("Acc Domain", f"{train_metrics['acc_domain']:.4f}")

        # Performance metrics
        if "f1" in train_metrics:
            table.add_row("F1 Score", f"{train_metrics['f1']:.4f}")
        if "precision" in train_metrics:
            table.add_row("Precision", f"{train_metrics['precision']:.4f}")
        if "recall" in train_metrics:
            table.add_row("Recall", f"{train_metrics['recall']:.4f}")
        if "auroc" in train_metrics:
            table.add_row("AUROC", f"{train_metrics['auroc']:.4f}")
        if "mcc" in train_metrics:
            table.add_row("MCC", f"{train_metrics['mcc']:.4f}")

    # Validation metrics
    if val_metrics:
        table.add_section()
        table.add_row("Validation", "")
        if "accuracy" in val_metrics:
            table.add_row("Val Accuracy", f"{val_metrics['accuracy']:.4f}")
        if "f1" in val_metrics:
            table.add_row("Val F1", f"{val_metrics['f1']:.4f}")

    # Test metrics
    if test_metrics:
        table.add_section()
        table.add_row("Test", "")
        if "accuracy" in test_metrics:
            table.add_row("Test Accuracy", f"{test_metrics['accuracy']:.4f}")

    # Best model info
    if best_accuracy is not None:
        table.add_section()
        table.add_row("Best Model", f"{best_accuracy:.4f} @ Epoch {best_epoch + 1}")

    return table


def update_detailed_metrics_display(
    train_metrics,
    val_metrics=None,
    test_metrics=None,
    epoch=0,
    num_epoch=1,
    best_accuracy=None,
    best_epoch=None,
):
    """Update the detailed metrics display."""
    if not RICH_AVAILABLE or not _live_instance:
        return

    # Create updated table
    table = create_detailed_metrics_table(
        train_metrics,
        val_metrics,
        test_metrics,
        epoch,
        num_epoch,
        best_accuracy,
        best_epoch,
    )

    if table:
        # Update the table in the layout
        _live_instance.renderable["table"].update(
            Panel(table, title="📊 Detailed Metrics", border_style="bright_blue")
        )
    table.add_column("Metric", style="cyan", width=10)
    table.add_column("Train", justify="center", width=12)
    table.add_column("Val", justify="center", width=12)
    table.add_column("Test", justify="center", width=12)

    # Define metrics to display - compact list
    metrics_config = [
        ("Accuracy", "acc_class", "accuracy", "accuracy", True),
        ("F1", "f1", "f1", "f1", True),
        ("AUROC", "auroc", "auroc", "auroc", True),
        ("MCC", "mcc", "mcc", "mcc", True),
        ("Loss", "loss_class", "loss", "loss", False),
    ]

    for display_name, train_key, val_key, test_key, higher_is_better in metrics_config:
        # Get current values
        train_val = (
            train_metrics.get(train_key) if train_key and train_metrics else None
        )
        val_val = val_metrics.get(val_key) if val_key and val_metrics else None
        test_val = test_metrics.get(test_key) if test_key and test_metrics else None

        # Get previous values for arrows
        prev_key = f"{display_name.lower().replace(' ', '_')}"
        prev_train = _previous_metrics.get(f"train_{prev_key}")
        prev_val = _previous_metrics.get(f"val_{prev_key}")
        prev_test = _previous_metrics.get(f"test_{prev_key}")

        # Create formatted strings with arrows
        def format_metric(value, prev_value, higher_is_better):
            if value is None:
                return Text("N/A", style="dim")

            arrow, color = get_metric_arrow(value, prev_value, higher_is_better)
            return Text(f"{arrow} {value:.4f}", style=color)

        train_text = format_metric(train_val, prev_train, higher_is_better)
        val_text = format_metric(val_val, prev_val, higher_is_better)
        test_text = format_metric(test_val, prev_test, higher_is_better)

        table.add_row(display_name, train_text, val_text, test_text)

        # Update previous metrics
        if train_val is not None:
            _previous_metrics[f"train_{prev_key}"] = train_val
        if val_val is not None:
            _previous_metrics[f"val_{prev_key}"] = val_val
        if test_val is not None:
            _previous_metrics[f"test_{prev_key}"] = test_val

    # Add best model info
    if val_metrics and best_accuracy is not None:
        table.add_section()
        best_text = Text(
            f"🏆 {best_accuracy:.4f} @ Epoch {best_epoch + 1}", style="bold gold1"
        )
        improved_text = Text(
            "💾 New Best!" if val_metrics.get("accuracy", 0) >= best_accuracy else "",
            style="bold green",
        )
        table.add_row("Best Model", "", best_text, improved_text)

    _current_table = table
    return table


def update_dynamic_display(
    train_metrics,
    val_metrics=None,
    test_metrics=None,
    epoch=0,
    num_epoch=1,
    best_accuracy=None,
    best_epoch=None,
    clustering_info="",
):
    """Update the dynamic display with new metrics."""
    if not RICH_AVAILABLE or not _live_instance:
        return

    global _current_table

    # Create updated table
    table = create_dynamic_metrics_table(
        train_metrics,
        val_metrics,
        test_metrics,
        epoch,
        num_epoch,
        best_accuracy,
        best_epoch,
        clustering_info,
    )

    if table:
        # Update the table in the layout with compact panel
        _live_instance.renderable["table"].update(
            Panel(
                table,
                title="📊 Training Metrics",
                border_style="bright_blue",
                height=10,
            )
        )


def advance_overall_progress():
    """Advance the overall progress by one epoch."""
    if RICH_AVAILABLE and _progress_instance and _overall_task_id is not None:
        _progress_instance.advance(_overall_task_id, 1)


def finish_dynamic_display():
    """Finish the dynamic display."""
    if RICH_AVAILABLE and _live_instance:
        # Final update
        update_epoch_progress("end")
        _live_instance.stop()


def check_gradient_norm(model, threshold=1e4, verbose=True):
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)  # L2 norm
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    if total_norm > threshold and verbose:
        print(f"Warning: Gradient norm {total_norm} exceeds threshold {threshold}")
    return total_norm


def check_nan_inf_in_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN detected in gradient of parameter: {name}")
            if torch.isinf(param.grad).any():
                print(f"Inf detected in gradient of parameter: {name}")


def train(
    model,
    train_data,
    optimizers,
    device,
    epoch,
    num_epoch,
    filename,
    disc_weight=None,
    class_weight=None,
    label_smooth=None,
    entropy_weight=1.0,
    grl_weight=1.0,
    max_batches=None,
    wandb_enabled=False,
    clip_grad_norm=None,
    verbose=True,  # Add verbose parameter
    total_batches=None,  # Add total_batches parameter for progress tracking
):
    scaler = GradScaler()

    # Reloading buffer
    # train_data.dataset.dataset.reload_buffer()

    # Loss Functions
    label_smooth = label_smooth if label_smooth is not None else 0.0
    class_criterion = nn.CrossEntropyLoss(
        weight=class_weight, label_smoothing=label_smooth
    )
    domain_criterion = nn.CrossEntropyLoss(
        weight=disc_weight, label_smoothing=label_smooth
    )
    entropy_criterion = EntropyLoss()

    P = epoch / num_epoch
    alpha = (2.0 / (1.0 + np.exp(-10 * P)) - 1) * grl_weight
    beta = (2.0 / (1.0 + np.exp(-10 * P)) - 1) * entropy_weight

    model.DomainClassifier.set_lambd(alpha)
    model.train()

    running_loss_class = 0.0
    running_correct_class = 0.0

    running_loss_domain = 0.0
    running_correct_domain = 0.0

    running_loss_entropy = 0.0

    data_size = 0

    num_classes = (
        model.LabelClassifier.output_nodes
    )  # Assuming this is how to get num_classes

    # Initialize TorchMetrics for training - moved to CPU to save GPU memory
    f1_metric_train = F1Score(
        task="multiclass", num_classes=num_classes, average="weighted"
    ).to("cpu")
    precision_metric_train = Precision(
        task="multiclass", num_classes=num_classes, average="weighted"
    ).to("cpu")
    recall_metric_train = Recall(
        task="multiclass", num_classes=num_classes, average="weighted"
    ).to("cpu")
    auroc_metric_train = AUROC(
        task="multiclass", num_classes=num_classes, average="weighted"
    ).to("cpu")
    avg_precision_metric_train = AveragePrecision(
        task="multiclass", num_classes=num_classes, average="weighted"
    ).to("cpu")
    mcc_metric_train = MatthewsCorrCoef(task="multiclass", num_classes=num_classes).to(
        "cpu"
    )
    sensitivity_metric_train = Recall(
        task="multiclass", num_classes=num_classes, average="weighted"
    ).to("cpu")  # Sensitivity is Recall
    specificity_metric_train = Specificity(
        task="multiclass", num_classes=num_classes, average="weighted"
    ).to("cpu")
    conf_matrix_train = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(
        "cpu"
    )

    # Use lists instead of tensors to avoid GPU memory accumulation
    all_labels_list = []
    all_predictions_list = []
    all_predictions_proba_list = []

    # Calculate total batches for progress tracking
    total_train_batches = total_batches or (
        len(train_data) if hasattr(train_data, "__len__") else None
    )
    if max_batches and total_train_batches:
        total_train_batches = min(total_train_batches, max_batches)

    batch_count = 0
    for data, class_label, domain_label in train_data:
        # Break if max_batches limit is reached
        if max_batches is not None and batch_count >= max_batches:
            break

        batch_count += 1
        data, class_label, domain_label = (
            data.squeeze().double().to(device),
            class_label.squeeze().long().to(device),
            domain_label.squeeze().long().to(device),
        )

        # Numerical instability is possible
        # data = torch.clamp(data, min=-1e4, max=1e4)  # Example clamping

        data_size += data.size(0)

        if torch.isnan(data).any() or torch.isinf(data).any():
            print("Input contains NaN or Inf values.")

        for optimizer in optimizers:
            optimizer.zero_grad()

        with autocast():
            output_class, output_domain = model(data)

            loss_class = class_criterion(output_class, class_label)
            loss_domain = domain_criterion(output_domain, domain_label)
            loss_entropy = entropy_criterion(output_class)

            total_loss = loss_class + loss_domain + loss_entropy * beta

        pred_class = output_class.squeeze().argmax(dim=1)
        pred_domain = output_domain.squeeze().argmax(dim=1)

        # Update per-batch progress with current metrics including domain metrics
        current_acc = (
            torch.sum(pred_class == class_label.data).double() / data.size(0)
        ).item()
        current_acc_domain = (
            torch.sum(pred_domain == domain_label.data).double() / data.size(0)
        ).item()

        # Calculate simple batch-level approximations without heavy TorchMetrics
        # These will be more accurate at epoch level, but provide some indication during training
        batch_f1 = 0.0
        batch_precision = 0.0
        batch_recall = 0.0
        batch_mcc = 0.0
        batch_auroc = 0.0

        try:
            # Simple batch-level calculations for display purposes (CPU-based to save GPU memory)
            pred_class_cpu = pred_class.cpu()
            class_label_cpu = class_label.cpu()

            # Calculate per-class metrics for this batch using simple calculations
            unique_labels = torch.unique(class_label_cpu)
            if len(unique_labels) > 1:  # Only if we have multiple classes
                # Simple precision, recall, f1 calculation without TorchMetrics
                tp = torch.sum((pred_class_cpu == 1) & (class_label_cpu == 1)).float()
                fp = torch.sum((pred_class_cpu == 1) & (class_label_cpu == 0)).float()
                tn = torch.sum((pred_class_cpu == 0) & (class_label_cpu == 0)).float()
                fn = torch.sum((pred_class_cpu == 0) & (class_label_cpu == 1)).float()

                # Avoid division by zero
                if tp + fp > 0:
                    batch_precision = (tp / (tp + fp)).item()
                else:
                    batch_precision = 0.0

                if tp + fn > 0:
                    batch_recall = (tp / (tp + fn)).item()
                else:
                    batch_recall = 0.0

                if batch_precision + batch_recall > 0:
                    batch_f1 = (
                        2
                        * (batch_precision * batch_recall)
                        / (batch_precision + batch_recall)
                    )
                else:
                    batch_f1 = 0.0

                # Simple MCC calculation
                numerator = tp * tn - fp * fn
                denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                if denominator > 0:
                    batch_mcc = (numerator / denominator).item()
                else:
                    batch_mcc = 0.0

                # Simple AUROC approximation (just use accuracy as a proxy to avoid GPU memory issues)
                batch_auroc = (
                    current_acc  # Use accuracy as proxy for AUROC during batches
                )
            else:
                # Single class in batch - use accuracy as fallback
                batch_f1 = current_acc
                batch_precision = current_acc
                batch_recall = current_acc
                batch_mcc = 0.0
                batch_auroc = current_acc

        except Exception as e:
            # Fallback to simple accuracy if any calculation fails
            batch_f1 = current_acc
            batch_precision = current_acc
            batch_recall = current_acc
            batch_mcc = 0.0
            batch_auroc = current_acc

        current_metrics = {
            "loss_class": loss_class.item(),
            "loss_domain": loss_domain.item(),
            "loss_entropy": loss_entropy.item(),
            "acc_class": current_acc,
            "acc_domain": current_acc_domain,
            "f1": batch_f1,  # Batch-level approximation
            "precision": batch_precision,  # Batch-level approximation
            "recall": batch_recall,  # Batch-level approximation
            "mcc": batch_mcc,  # Batch-level approximation
            "auroc": batch_auroc,  # Batch-level approximation
        }

        update_epoch_progress(
            "batch",
            epoch=epoch,
            total_epochs=num_epoch,
            total_batches=total_train_batches,
            current_batch=batch_count - 1,  # -1 because we incremented earlier
            train_metrics=current_metrics,
        )

        # print(f"{loss_class.item():.4f}, {loss_domain.item():.4f}, {loss_entropy.item():.4f}")

        scaler.scale(total_loss).backward()

        check_gradient_norm(model, verbose=verbose)
        check_nan_inf_in_gradients(model)

        for optimizer in optimizers:
            scaler.step(optimizer)
        scaler.update()

        # Store predictions and labels in CPU lists instead of concatenating GPU tensors
        all_labels_list.append(class_label.cpu())
        all_predictions_list.append(pred_class.cpu())
        all_predictions_proba_list.append(
            torch.nn.functional.softmax(output_class, dim=1).cpu()
        )

        # Update metrics with CPU tensors to avoid GPU memory issues
        f1_metric_train.update(pred_class.cpu(), class_label.cpu())
        precision_metric_train.update(pred_class.cpu(), class_label.cpu())
        recall_metric_train.update(pred_class.cpu(), class_label.cpu())
        auroc_metric_train.update(output_class.cpu(), class_label.cpu())
        avg_precision_metric_train.update(output_class.cpu(), class_label.cpu())
        mcc_metric_train.update(pred_class.cpu(), class_label.cpu())
        sensitivity_metric_train.update(pred_class.cpu(), class_label.cpu())
        specificity_metric_train.update(pred_class.cpu(), class_label.cpu())
        conf_matrix_train.update(pred_class.cpu(), class_label.cpu())

        # Classifier Loss
        running_loss_class += loss_class.item() * data.size(0)
        running_correct_class += torch.sum(pred_class == class_label.data)

        # Domain Loss
        running_loss_domain += loss_domain.item() * data.size(0)
        running_correct_domain += torch.sum(pred_domain == domain_label.data)

        # Entropy Loss
        running_loss_entropy += loss_entropy.item() * data.size(0)

        # Clear GPU cache more frequently to prevent memory buildup
        if batch_count % 3 == 0:  # Clear every 3 batches instead of 10
            torch.cuda.empty_cache()

    # Class Loss on Epoch
    epoch_loss_class = running_loss_class / data_size
    epoch_acc_class = running_correct_class.double() / data_size

    # Domain Loss on Epoch
    epoch_loss_domain = running_loss_domain / data_size
    epoch_acc_domain = running_correct_domain.double() / data_size

    # Entropy Loss on Epoch
    epoch_loss_entropy = running_loss_entropy / data_size

    # Calculate metrics using TorchMetrics
    f1_train = f1_metric_train.compute()
    precision_train = precision_metric_train.compute()
    recall_train = recall_metric_train.compute()
    auroc_train = auroc_metric_train.compute()
    avg_precision_train = avg_precision_metric_train.compute()
    mcc_train = mcc_metric_train.compute()
    sensitivity_train = sensitivity_metric_train.compute()
    specificity_train = specificity_metric_train.compute()

    log = (
        f"Train: Epoch: {epoch}/{num_epoch} | Alpha: {alpha:.4f} |\n"
        f"LClass: {epoch_loss_class:.4f} | Acc Class: {epoch_acc_class:.4f} |\n"
        f"LDomain: {epoch_loss_domain:.4f} | Acc Domain: {epoch_acc_domain:.4f} |\n"
        f"Loss Entropy: {epoch_loss_entropy:.4f} |\n"
        f"F1 Score: {f1_train:.4f} Precision: {precision_train:.4f} Recall: {recall_train:.4f}\n"
        f"AUROC: {auroc_train:.4f} Avg Precision: {avg_precision_train:.4f} MCC: {mcc_train:.4f}\n"
        f"Sensitivity: {sensitivity_train:.4f} Specificity: {specificity_train:.4f}"
    )

    if verbose:
        print(log)

    # Return metrics for Rich display
    train_metrics = {
        "loss_class": epoch_loss_class,
        "acc_class": epoch_acc_class.item()
        if hasattr(epoch_acc_class, "item")
        else float(epoch_acc_class),
        "loss_domain": epoch_loss_domain,
        "acc_domain": epoch_acc_domain.item()
        if hasattr(epoch_acc_domain, "item")
        else float(epoch_acc_domain),
        "loss_entropy": epoch_loss_entropy,
        "f1": f1_train.item() if hasattr(f1_train, "item") else float(f1_train),
        "precision": precision_train.item()
        if hasattr(precision_train, "item")
        else float(precision_train),
        "recall": recall_train.item()
        if hasattr(recall_train, "item")
        else float(recall_train),
        "auroc": auroc_train.item()
        if hasattr(auroc_train, "item")
        else float(auroc_train),
        "avg_precision": avg_precision_train.item()
        if hasattr(avg_precision_train, "item")
        else float(avg_precision_train),
        "mcc": mcc_train.item() if hasattr(mcc_train, "item") else float(mcc_train),
    }
    with open(filename, "a") as f:
        f.write(log + "\n")

    if wandb_enabled:
        wandb.log(
            {
                "Train/Loss_Class": epoch_loss_class,
                "Train/Acc_Class": epoch_acc_class,
                "Train/Loss_Domain": epoch_loss_domain,
                "Train/Acc_Domain": epoch_acc_domain,
                "Train/Loss_Entropy": epoch_loss_entropy,
                "Train/F1_Score": f1_train,
                "Train/Precision": precision_train,
                "Train/Recall": recall_train,
                "Train/AUROC": auroc_train,
                "Train/Average_Precision": avg_precision_train,
                "Train/MCC": mcc_train,
                "Train/Sensitivity": sensitivity_train,
                "Train/Specificity": specificity_train,
                "Train/Alpha": alpha,
            },
            step=epoch,
        )

    # Return actual number of batches processed (considering max_batches limit)
    actual_batches = (
        min(len(train_data), max_batches)
        if max_batches is not None
        else len(train_data)
    )

    return model, optimizers, train_metrics


def eval(
    model,
    eval_data,
    criterion,
    device,
    num_epoch,
    filename,
    wandb_enabled=False,
    epoch=0,
    verbose=True,  # Add verbose parameter
):
    criterion = nn.CrossEntropyLoss()

    model.eval()
    running_loss = 0.0
    running_corrects = 0
    data_size = 0

    num_classes = model.LabelClassifier.output_nodes

    # Initialize TorchMetrics for evaluation
    f1_metric_eval = F1Score(
        task="multiclass", num_classes=num_classes, average="weighted"
    ).to(device)
    precision_metric_eval = Precision(
        task="multiclass", num_classes=num_classes, average="weighted"
    ).to(device)
    recall_metric_eval = Recall(
        task="multiclass", num_classes=num_classes, average="weighted"
    ).to(device)
    auroc_metric_eval = AUROC(
        task="multiclass", num_classes=num_classes, average="weighted"
    ).to(device)
    avg_precision_metric_eval = AveragePrecision(
        task="multiclass", num_classes=num_classes, average="weighted"
    ).to(device)
    mcc_metric_eval = MatthewsCorrCoef(task="multiclass", num_classes=num_classes).to(
        device
    )
    sensitivity_metric_eval = Recall(
        task="multiclass", num_classes=num_classes, average="weighted"
    ).to(device)  # Sensitivity is Recall
    specificity_metric_eval = Specificity(
        task="multiclass", num_classes=num_classes, average="weighted"
    ).to(device)
    conf_matrix_eval = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(
        device
    )

    with torch.inference_mode():
        for data, labels, *_ in eval_data:
            data, labels = (
                data.squeeze().double().to(device),
                labels.squeeze().long().to(device),
            )

            # get class labels
            output = model(data)[0].squeeze()
            prediction = output.argmax(dim=1)

            loss = criterion(output, labels)

            running_loss += loss.item() * data.size(0)
            running_corrects += torch.sum(prediction == labels.data).item()
            data_size += data.size(0)

            f1_metric_eval.update(prediction, labels)
            precision_metric_eval.update(prediction, labels)
            recall_metric_eval.update(prediction, labels)
            auroc_metric_eval.update(output, labels)
            avg_precision_metric_eval.update(output, labels)
            mcc_metric_eval.update(prediction, labels)
            sensitivity_metric_eval.update(prediction, labels)
            specificity_metric_eval.update(prediction, labels)
            conf_matrix_eval.update(prediction, labels)

    epoch_loss = running_loss / data_size
    epoch_acc = running_corrects / data_size

    # Calculate metrics using TorchMetrics
    f1_eval = f1_metric_eval.compute()
    precision_eval = precision_metric_eval.compute()
    recall_eval = recall_metric_eval.compute()
    auroc_eval = auroc_metric_eval.compute()
    avg_precision_eval = avg_precision_metric_eval.compute()
    mcc_eval = mcc_metric_eval.compute()
    sensitivity_eval = sensitivity_metric_eval.compute()
    specificity_eval = specificity_metric_eval.compute()

    # Enhanced logging format to match your desired style
    log = (
        f"Eval: Epoch: {epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} "
        f"F1 Score: {f1_eval:.4f} Precision: {precision_eval:.4f} Recall: {recall_eval:.4f} "
        f"AUROC: {auroc_eval:.4f} MCC: {mcc_eval:.4f}"
    )

    if verbose:
        print(log)

    with open(filename, "a") as f:
        f.write(log + "\n")

    # Log metrics to wandb
    if wandb_enabled:
        wandb.log(
            {
                "Val/Loss": epoch_loss,
                "Val/Accuracy": epoch_acc,
                "Val/F1_Score": f1_eval,
                "Val/Precision": precision_eval,
                "Val/Recall": recall_eval,
                "Val/AUROC": auroc_eval,
                "Val/Average_Precision": avg_precision_eval,
                "Val/MCC": mcc_eval,
                "Val/Sensitivity": sensitivity_eval,
                "Val/Specificity": specificity_eval,
            },
            step=epoch,
        )

    # Return metrics for Rich display
    eval_metrics = {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "f1": f1_eval.item() if hasattr(f1_eval, "item") else float(f1_eval),
        "precision": precision_eval.item()
        if hasattr(precision_eval, "item")
        else float(precision_eval),
        "recall": recall_eval.item()
        if hasattr(recall_eval, "item")
        else float(recall_eval),
        "auroc": auroc_eval.item()
        if hasattr(auroc_eval, "item")
        else float(auroc_eval),
        "avg_precision": avg_precision_eval.item()
        if hasattr(avg_precision_eval, "item")
        else float(avg_precision_eval),
        "mcc": mcc_eval.item() if hasattr(mcc_eval, "item") else float(mcc_eval),
    }

    return epoch_acc, eval_metrics


def demo_train_eval():
    """Demo function to test training and evaluation functionality."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from torch.utils.data import TensorDataset, DataLoader

        RICH_AVAILABLE = True
    except ImportError:
        RICH_AVAILABLE = False

    if RICH_AVAILABLE:
        console = Console()
        console.print(Panel.fit("🚀 Training & Evaluation Demo", style="bold blue"))
    else:
        print("🚀 Training & Evaluation Demo")

    try:
        # Create dummy model and data
        from skripsi_code.model.MoMLNIDS import momlnids

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Create model
        model = (
            momlnids(
                input_nodes=10,
                hidden_nodes=[32, 16],
                classifier_nodes=[16],
                num_domains=3,
                num_class=2,
                single_layer=True,
            )
            .double()
            .to(device)
        )

        print("✅ Model created successfully")

        # Create dummy data
        n_samples = 100
        n_features = 10

        # Training data
        X_train = torch.randn(n_samples, n_features, dtype=torch.double)
        y_train = torch.randint(0, 2, (n_samples,))
        d_train = torch.randint(0, 3, (n_samples,))

        train_dataset = TensorDataset(X_train, y_train, d_train)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        # Validation data
        X_val = torch.randn(50, n_features, dtype=torch.double)
        y_val = torch.randint(0, 2, (50,))
        d_val = torch.randint(0, 3, (50,))

        val_dataset = TensorDataset(X_val, y_val, d_val)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        print("✅ Dummy datasets created")

        # Create optimizers
        from skripsi_code.utils.utils import get_model_learning_rate, get_optimizer

        model_learning_rate = get_model_learning_rate(model, 1.0, 1.0, 1.0)
        optimizers = [
            get_optimizer(module, 0.001 * alpha)
            for module, alpha in model_learning_rate
        ]

        print("✅ Optimizers created")

        # Test training for one epoch
        print("🔄 Testing training function...")

        model_trained, optimizers_updated, num_batches = train(
            model=model,
            train_data=train_loader,
            optimizers=optimizers,
            device=device,
            epoch=1,
            num_epoch=5,
            filename="temp_train_log.txt",
            max_batches=3,  # Limit for demo
            wandb_enabled=False,
        )

        print(f"✅ Training completed: {num_batches} batches processed")

        # Test evaluation
        print("📊 Testing evaluation function...")

        criterion = nn.CrossEntropyLoss()
        accuracy = eval(
            model=model_trained,
            eval_data=val_loader,
            criterion=criterion,
            device=device,
            num_epoch=5,
            filename="temp_eval_log.txt",
            wandb_enabled=False,
            epoch=1,
        )

        print(f"✅ Evaluation completed: Accuracy = {accuracy:.4f}")

        # Clean up temp files
        import os

        for temp_file in ["temp_train_log.txt", "temp_eval_log.txt"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        print("✨ Training & Evaluation demo completed!")

    except Exception as e:
        print(f"❌ Error in demo: {e}")
        import traceback

        traceback.print_exc()


# Only add click decorator if click is available
if CLICK_AVAILABLE:

    @click.command()
    @click.option(
        "--demo", is_flag=True, help="Run training and evaluation demonstration"
    )
    @click.option("--test-metrics", is_flag=True, help="Test metrics calculation")
    def main(demo, test_metrics):
        """
        Test and demonstrate training and evaluation functionality.
        """
        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        if demo:
            demo_train_eval()
        elif test_metrics:
            console.print(Panel.fit("📊 Testing Metrics", style="bold blue"))

            # Test metrics computation
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Create dummy predictions and labels
            num_classes = 2
            batch_size = 32

            predictions = torch.randint(0, num_classes, (batch_size,)).to(device)
            labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            logits = torch.randn(batch_size, num_classes).to(device)

            # Test individual metrics
            f1_metric = F1Score(
                task="multiclass", num_classes=num_classes, average="weighted"
            ).to(device)
            precision_metric = Precision(
                task="multiclass", num_classes=num_classes, average="weighted"
            ).to(device)
            recall_metric = Recall(
                task="multiclass", num_classes=num_classes, average="weighted"
            ).to(device)

            f1_score = f1_metric(predictions, labels)
            precision_score = precision_metric(predictions, labels)
            recall_score = recall_metric(predictions, labels)

            console.print(f"✅ F1 Score: {f1_score:.4f}")
            console.print(f"✅ Precision: {precision_score:.4f}")
            console.print(f"✅ Recall: {recall_score:.4f}")

            console.print("✨ Metrics test completed!")
        else:
            console.print(
                "Use --demo to run demonstration or --test-metrics to test metrics"
            )
else:

    def main():
        """
        Test and demonstrate training and evaluation functionality.
        """
        print("Training & Evaluation Module")
        print("Available functions:")
        print("- demo_train_eval(): Run training and evaluation demo")
        print("- train(): Training function")
        print("- eval(): Evaluation function")

        # Run demo by default when click is not available
        demo_train_eval()
