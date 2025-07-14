import sys
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import wandb
import uuid
from torch import nn

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import project modules
from skripsi_code.utils.utils import (
    get_model_learning_rate,
    get_learning_rate_scheduler,
    get_optimizer,
    split_domain,
)
from skripsi_code.utils.dataloader import random_split_dataloader
from skripsi_code.clustering.cluster_utils import pseudolabeling
import sys
import importlib

# Force reload of TrainEval module to ensure we get the latest version
if "skripsi_code.TrainEval.TrainEval" in sys.modules:
    importlib.reload(sys.modules["skripsi_code.TrainEval.TrainEval"])

from skripsi_code.TrainEval.TrainEval import (
    train,
    eval,
    setup_dynamic_training_display,
    update_epoch_progress,
    finish_dynamic_display,
    update_dynamic_display,
    advance_overall_progress,
    reset_metrics_tracking,
)
from skripsi_code.model.MoMLNIDS import momlnids
from skripsi_code.config import load_config, get_config
from skripsi_code.config.config_manager import config_manager

DOMAIN_LIST = [
    "NF-CSE-CIC-IDS2018-v2",
    "NF-ToN-IoT-v2",
    "NF-UNSW-NB15-v2",
]


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_normal_(m.weight)


def main():
    # Set environment variable for better CUDA memory management
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment_config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    # Load configuration
    config = get_config()
    if args.config != "config/default_config.yaml":
        with open(args.config, "r") as f:
            override_config = yaml.safe_load(f)
        config_manager.update_config(override_config)
        config = get_config()  # Reload config after update

    # Extract parameters from config
    PROJECT_NAME = config.project.name
    PROJECT_DESCRIPTION = getattr(config.project, "description", "MoMLNIDS Project")
    PROJECT_VERSION = getattr(config.project, "version", "1.0.0")

    # Wandb settings
    WANDB_ENABLED = config.wandb.enabled
    WANDB_PROJECT = getattr(config.wandb, "project", PROJECT_NAME)
    WANDB_ENTITY = getattr(config.wandb, "entity", None)

    # Training settings
    NUM_EPOCH = config.training.epochs
    BATCH_SIZE = config.training.batch_size
    LEARNING_RATE = config.training.learning_rate
    TARGET_INDEX = config.training.target_index
    EXPERIMENT_NUM = config.training.experiment_num
    USE_CLUSTER = config.training.use_cluster
    EVAL_STEP = config.training.eval_step
    SAVE_STEP = config.training.save_step
    MAX_BATCHES = getattr(config.training, "max_batches", None)

    # Model settings
    INPUT_NODES = config.model.input_nodes
    SINGLE_LAYER = config.model.single_layer

    # Data settings
    PROCESSED_DATA_PATH = config.data.processed_data_path
    DOMAIN_REWEIGHTING = config.data.domain_reweighting
    LABEL_REWEIGHTING = config.data.label_reweighting

    # Device settings
    NUM_WORKERS = config.device.num_workers

    # Output settings - create structured experiment directory
    from skripsi_code.utils.utils import (
        create_experiment_directory,
        get_experiment_log_files,
        get_model_checkpoint_path,
    )

    # Create structured experiment directory
    target_domain_name = (
        DOMAIN_LIST[TARGET_INDEX] if TARGET_INDEX < len(DOMAIN_LIST) else "Unknown"
    )

    # Safely get cluster_num from config
    cluster_num = None
    try:
        if hasattr(config, "experiment"):
            cluster_num = getattr(config.experiment, "cluster_num", None)
    except:
        pass

    if cluster_num is None:
        try:
            if hasattr(config, "training"):
                cluster_num = getattr(config.training, "cluster_num", None)
        except:
            pass

    experiment_dir = create_experiment_directory(
        base_dir="ProperTraining",
        target_domain=target_domain_name,
        experiment_name=EXPERIMENT_NUM,
        use_clustering=USE_CLUSTER,
        cluster_num=cluster_num,
        epochs=NUM_EPOCH,
    )
    experiment_dir = create_experiment_directory(
        base_dir="ProperTraining",
        target_domain=target_domain_name,
        experiment_name=EXPERIMENT_NUM,
        use_clustering=USE_CLUSTER,
        cluster_num=cluster_num,
        epochs=NUM_EPOCH,
    )

    # Get log file paths
    log_files = get_experiment_log_files(experiment_dir)

    # Legacy compatibility
    MODELS_DIR = str(experiment_dir)
    LOGS_DIR = str(experiment_dir)

    print(f"📁 Experiment Directory: {experiment_dir}")
    print(f"📂 Log Files: {list(log_files.keys())}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb if enabled
    if WANDB_ENABLED:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f"{PROJECT_NAME}_{EXPERIMENT_NUM}_{uuid.uuid4().hex[:8]}",
            config=dict(config),
        )

    try:
        # Print configuration summary
        print("=" * 80)
        print(f"🚀 MoMLNIDS Training Configuration")
        print("=" * 80)
        print(f"📊 Project: {PROJECT_NAME}")
        print(
            f"🎯 Target Index: {TARGET_INDEX} ({DOMAIN_LIST[TARGET_INDEX] if TARGET_INDEX < len(DOMAIN_LIST) else 'Unknown'})"
        )
        print(f"🔬 Experiment: {EXPERIMENT_NUM}")
        print(f"📈 Epochs: {NUM_EPOCH}")
        print(f"📦 Batch Size: {BATCH_SIZE}")
        print(f"⚡ Learning Rate: {LEARNING_RATE}")
        print(f"🧠 Model: Single Layer={SINGLE_LAYER}, Input Nodes={INPUT_NODES}")
        print(f"🔄 Use Clustering: {USE_CLUSTER}")
        print(f"💾 Models Dir: {MODELS_DIR}")
        print(f"📝 Logs Dir: {LOGS_DIR}")
        print("=" * 80)

        # Load data
        print(f"📂 Loading data from: {PROCESSED_DATA_PATH}")

        # Check if real data exists and try to load it
        datapath = Path(PROCESSED_DATA_PATH)
        real_data_loaded = False

        if datapath.exists():
            try:
                print(f"✅ Data path exists: {datapath}")

                # Try to use the real dataloader
                from skripsi_code.utils.dataloader import random_split_dataloader

                # Check available datasets
                available_datasets = []
                for domain in DOMAIN_LIST:
                    domain_path = datapath / domain
                    if domain_path.exists():
                        parquet_files = list(domain_path.glob("*.parquet"))
                        if parquet_files:
                            available_datasets.append(domain)
                            print(
                                f"  📁 Found {domain}: {len(parquet_files)} parquet files"
                            )

                if available_datasets:
                    print(f"🎯 Available datasets: {available_datasets}")

                    # Use available datasets for training
                    domains_to_use = available_datasets[:3]  # Limit for demo
                    print(f"📊 Using domains for training: {domains_to_use}")

                    # Prepare source and target domains
                    target_domain_name = domains_to_use[
                        min(TARGET_INDEX, len(domains_to_use) - 1)
                    ]
                    source_domains = [
                        d for d in domains_to_use if d != target_domain_name
                    ]

                    print(f"🎯 Target domain: {target_domain_name}")
                    print(f"📦 Source domains: {source_domains}")

                    # Try to load real data using the project's dataloader
                    source_train, source_val, target_test = random_split_dataloader(
                        dir_path=str(datapath),
                        source_dir=source_domains,
                        target_dir=[target_domain_name],
                        source_domain=list(range(len(source_domains))),
                        target_domain=[
                            len(source_domains)
                        ],  # Target domain gets next index
                        get_domain=True,
                        get_cluster=USE_CLUSTER,
                        batch_size=BATCH_SIZE,
                        n_workers=NUM_WORKERS,
                        chunk=True,
                    )

                    print(f"✅ Successfully loaded real datasets!")
                    print(f"  🚂 Train batches: {len(source_train)}")
                    print(f"  🔍 Val batches: {len(source_val)}")
                    print(f"  🎯 Test batches: {len(target_test)}")

                    # Sample a batch to check data shape
                    for batch_data, batch_labels, batch_domains in source_train:
                        print(f"  📊 Data shape: {batch_data.shape}")
                        print(f"  🏷️  Labels shape: {batch_labels.shape}")
                        print(f"  🌐 Domains shape: {batch_domains.shape}")
                        # Flatten labels and domains for bincount (needs 1D tensors)
                        flat_labels = batch_labels.flatten()
                        flat_domains = batch_domains.flatten()
                        print(f"  📈 Label distribution: {torch.bincount(flat_labels)}")
                        print(
                            f"  🗺️  Domain distribution: {torch.bincount(flat_domains)}"
                        )
                        break

                    real_data_loaded = True

                else:
                    print("⚠️  No parquet files found in any domain directories")

            except Exception as e:
                print(f"⚠️  Failed to load real data: {e}")
                print("🔄 Falling back to dummy data for demonstration")

        if not real_data_loaded:
            print(f"⚠️  Data path not found or loading failed: {datapath}")
            print("🎭 Creating dummy data for demonstration...")

            # Create dummy data with proper domain distribution
            dummy_data = torch.randn(100, INPUT_NODES).double()
            dummy_labels = torch.randint(0, 2, (100,)).long()
            dummy_domains = torch.randint(0, len(DOMAIN_LIST), (100,)).long()

            # Create simple train/val/test split
            from torch.utils.data import TensorDataset, DataLoader

            dataset = TensorDataset(dummy_data, dummy_labels, dummy_domains)

            train_size = int(0.6 * len(dataset))
            val_size = int(0.2 * len(dataset))
            test_size = len(dataset) - train_size - val_size

            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size]
            )

            source_train = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,  # Use config batch size consistently
                shuffle=True,
                num_workers=NUM_WORKERS,
                drop_last=True,  # Important for BatchNorm
            )

            source_val = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,  # Use config batch size consistently
                shuffle=False,
                num_workers=NUM_WORKERS,
                drop_last=True,
            )
            target_test = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,  # Use config batch size consistently
                shuffle=False,
                num_workers=NUM_WORKERS,
                drop_last=True,
            )

            print(f"🎭 Created dummy data:")
            print(f"  🚂 Train batches: {len(source_train)}")
            print(f"  🔍 Val batches: {len(source_val)}")
            print(f"  🎯 Test batches: {len(target_test)}")

            # Sample a batch to show data shape
            for batch_data, batch_labels, batch_domains in source_train:
                print(f"  📊 Data shape: {batch_data.shape}")
                print(f"  🏷️  Labels shape: {batch_labels.shape}")
                print(f"  🌐 Domains shape: {batch_domains.shape}")
                break

        # Initialize model
        print("\n🧠 Initializing model...")
        hidden_nodes = [64, 32, 16, 10]
        classifier_nodes = [64, 32, 16]  # Always provide classifier nodes

        model = momlnids(
            input_nodes=INPUT_NODES,
            hidden_nodes=hidden_nodes,
            classifier_nodes=classifier_nodes,
            num_domains=len(DOMAIN_LIST),
            num_class=2,
            single_layer=SINGLE_LAYER,
        )
        model.apply(init_weights)
        model = model.double().to(device)

        print(f"✅ Model initialized successfully!")
        print(f"  📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(
            f"  🎯 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

        # Initialize optimizers and schedulers
        print("\n⚙️  Initializing optimizers and schedulers...")
        model_optimizer = get_optimizer(
            model=model,
            init_lr=LEARNING_RATE,
            amsgrad=getattr(config.training, "amsgrad", False),
        )

        model_scheduler = get_learning_rate_scheduler(
            optimizer=model_optimizer, t_max=getattr(config.training, "t_max", 10)
        )

        print(f"✅ Optimizer: {type(model_optimizer).__name__}")
        print(f"✅ Scheduler: {type(model_scheduler).__name__}")

        # Training loop
        best_accuracy = 0.0
        test_accuracy = 0.0
        best_epoch = 0

        print(f"\n🚀 Starting training for {NUM_EPOCH} epochs...")
        print("=" * 80)

        # Reset metrics tracking and setup dynamic Rich display
        reset_metrics_tracking()
        live_display, progress, overall_task_id, epoch_task_id = (
            setup_dynamic_training_display(
                NUM_EPOCH, f"{PROJECT_NAME} - {EXPERIMENT_NUM}"
            )
        )

        if live_display:
            live_display.start()

        try:
            for epoch in range(NUM_EPOCH):
                # Start epoch progress with batch count info
                total_batches = (
                    len(source_train) if hasattr(source_train, "__len__") else None
                )
                if MAX_BATCHES and total_batches:
                    total_batches = min(total_batches, MAX_BATCHES)

                update_epoch_progress("start", epoch=epoch, total_batches=total_batches)

                # Limit batches for demo if specified
                if MAX_BATCHES and epoch == 0:
                    print(f"🎭 Demo mode: limiting to {MAX_BATCHES} batches per epoch")

                # Training
                model, optimizers, train_metrics = train(
                    model=model,
                    train_data=source_train,
                    optimizers=[model_optimizer],  # Wrap in list
                    device=device,
                    epoch=epoch,
                    num_epoch=NUM_EPOCH,
                    filename=log_files["source_trained"],
                    max_batches=MAX_BATCHES,
                    wandb_enabled=WANDB_ENABLED,
                    label_smooth=0.0,  # Add label smoothing parameter
                    verbose=False,  # Reduce training output spam
                    total_batches=total_batches,  # Pass total batches for progress tracking
                )

                # Update progress after training
                update_epoch_progress(
                    "training_done", epoch=epoch, train_metrics=train_metrics
                )

                # Evaluation
                if (epoch + 1) % EVAL_STEP == 0:
                    import torch.nn as nn

                    criterion = nn.CrossEntropyLoss()

                    val_accuracy, val_metrics = eval(
                        model=model,
                        eval_data=source_val,
                        criterion=criterion,
                        device=device,
                        num_epoch=NUM_EPOCH,
                        filename=log_files["val_performance"],
                        wandb_enabled=WANDB_ENABLED,
                        epoch=epoch,
                        verbose=False,  # Reduce eval output spam
                    )

                    target_accuracy, test_metrics = eval(
                        model=model,
                        eval_data=target_test,
                        criterion=criterion,
                        device=device,
                        num_epoch=NUM_EPOCH,
                        filename=log_files["target_performance"],
                        wandb_enabled=WANDB_ENABLED,
                        epoch=epoch,
                        verbose=False,  # Reduce eval output spam
                    )

                    if val_accuracy >= best_accuracy:
                        best_accuracy = val_accuracy
                        test_accuracy = target_accuracy
                        best_epoch = epoch
                        model_save_path = get_model_checkpoint_path(
                            experiment_dir, is_best=True
                        )
                        torch.save(model.state_dict(), model_save_path)

                        # Also save epoch-specific checkpoint
                        epoch_model_path = get_model_checkpoint_path(
                            experiment_dir, epoch=epoch
                        )
                        torch.save(model.state_dict(), epoch_model_path)

                    # Update dynamic display with enhanced class/domain metrics
                    clustering_info = (
                        f" | Clustering: {USE_CLUSTER}" if USE_CLUSTER else ""
                    )
                    update_dynamic_display(
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        test_metrics=test_metrics,
                        epoch=epoch,
                        num_epoch=NUM_EPOCH,
                        best_accuracy=best_accuracy,
                        best_epoch=best_epoch,
                        clustering_info=clustering_info,
                    )
                else:
                    # For non-evaluation epochs, show training metrics with class/domain details
                    update_dynamic_display(
                        train_metrics=train_metrics,
                        epoch=epoch,
                        num_epoch=NUM_EPOCH,
                        clustering_info=f" | Clustering: {USE_CLUSTER}"
                        if USE_CLUSTER
                        else "",
                    )

                # Complete epoch progress
                update_epoch_progress(
                    "evaluation_done",
                    epoch=epoch,
                    total_batches=total_batches,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    test_metrics=test_metrics,
                )

                # Save model checkpoint every 2 epochs (adjustable)
                if epoch % 2 == 0:
                    epoch_model_path = get_model_checkpoint_path(
                        experiment_dir, epoch=epoch
                    )
                    torch.save(model.state_dict(), epoch_model_path)

                # Advance overall progress
                advance_overall_progress()

                # Update schedulers
                model_scheduler.step()

        except KeyboardInterrupt:
            # Handle interruption gracefully
            from skripsi_code.TrainEval.TrainEval import show_interruption_message

            show_interruption_message()
            print("Finishing current epoch...")

        finally:
            # Ensure display is properly closed
            finish_dynamic_display()

        # Display training summary
        from skripsi_code.TrainEval.TrainEval import display_training_summary

        display_training_summary(
            experiment_dir=experiment_dir,
            best_accuracy=best_accuracy,
            best_epoch=best_epoch,
            test_accuracy=test_accuracy,
            wandb_enabled=WANDB_ENABLED,
        )

    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        # Clean up wandb
        if WANDB_ENABLED:
            wandb.finish()


if __name__ == "__main__":
    main()
