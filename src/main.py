#!/usr/bin/env python3
"""
MoMLNIDS Main Training Script - Source-Only Baseline

This script runs the baseline training without pseudo-labeling clustering.
It trains on source domains and evaluates on target domains.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import wandb
import uuid
from torch import nn
from tqdm import tqdm

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
    create_experiment_directory,
    get_experiment_log_files,
    get_model_checkpoint_path,
)
from skripsi_code.utils.dataloader import random_split_dataloader
from skripsi_code.clustering.cluster_utils import pseudolabeling
from skripsi_code.TrainEval.TrainEval import train, eval
from skripsi_code.model.MoMLNIDS import momlnids

# Domain configuration
DOMAIN_LIST = [
    "NF-UNSW-NB15-v2",
    "NF-CSE-CIC-IDS2018-v2",
    "NF-ToN-IoT-v2",
]


def init_weights(m):
    """Initialize model weights using Xavier normal initialization."""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_normal_(m.weight)


def main():
    """Main training function for source-only baseline."""

    # Configuration
    VERBOSE = False
    SINGLE_LAYER = True

    # Domain reweighting based on dataset sizes
    DOMAIN_REWEIGHTING = [23, 187, 169]  # Approximate relative sizes

    # Label reweighting (benign, attack ratios)
    LABEL_REWEIGHTING = [
        (0.8735, 0.1265),  # UNSW-NB15
        (0.8307, 0.1693),  # CSE-CIC-IDS2018
        (0.0356, 0.9644),  # ToN-IoT
    ]

    # Model architecture
    HIDDEN_NODES = [64, 32, 16, 10]
    CLASS_NODES = [64, 32, 16]

    # Training configuration
    BATCH_SIZE = 1
    NUM_EPOCH = 20
    EVAL_STEP = 1
    SAVE_STEP = 2
    INIT_LEARNING_RATE = 0.0015

    # Training weights
    CLASSIFIER_WEIGHT = 1
    DISCRIMINATOR_WEIGHT = 1
    EXTRACTOR_WEIGHT = 1

    # Loss configuration
    ENTROPY_WEIGHT = 1
    GRL_WEIGHT = 1.25
    LABEL_SMOOTH = 0.1

    # Optimizer settings
    WEIGHT_DECAY = 5e-4
    AMSGRAD = True
    T_MAX = 10

    # Domain and clustering settings
    USE_CLUSTER = True  # Use pseudo-labeling
    USE_DOMAIN = not USE_CLUSTER
    NUM_CLUSTERS = 4
    NUM_CLASSES = 2
    CLUSTERING_STEP = 2

    EXPERIMENT_NUM = "|PseudoLabelling|"
    DATA_PATH = "./src/skripsi_code/data/parquet/"

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run training for each target domain
    for TARGET_INDEX in range(len(DOMAIN_LIST)):
        print("=" * 100)
        print(f"🚀 MoMLNIDS Training - Source-Only Baseline")
        print("=" * 100)
        print(f"📊 Target Domain: {DOMAIN_LIST[TARGET_INDEX]} (Index: {TARGET_INDEX})")
        print(
            f"📦 Source Domains: {[d for i, d in enumerate(DOMAIN_LIST) if i != TARGET_INDEX]}"
        )
        print(f"🔬 Experiment: {EXPERIMENT_NUM}")
        print(f"📈 Epochs: {NUM_EPOCH}")
        print(f"🧠 Model: Single Layer={SINGLE_LAYER}")
        print(f"🔄 Use Clustering: {USE_CLUSTER}")
        print(f"🎯 Device: {device}")
        print("=" * 100)

        # Initialize Wandb
        wandb.init(
            project="MoMLNIDS_Training",
            group=DOMAIN_LIST[TARGET_INDEX],
            tags=[EXPERIMENT_NUM],
            name=f"{DOMAIN_LIST[TARGET_INDEX]}_{EXPERIMENT_NUM}_{uuid.uuid4().hex[:8]}",
            config={
                "target_domain": DOMAIN_LIST[TARGET_INDEX],
                "source_domains": [
                    d for idx, d in enumerate(DOMAIN_LIST) if idx != TARGET_INDEX
                ],
                "batch_size": BATCH_SIZE,
                "num_epoch": NUM_EPOCH,
                "init_learning_rate": INIT_LEARNING_RATE,
                "hidden_nodes": HIDDEN_NODES,
                "class_nodes": CLASS_NODES,
                "use_cluster": USE_CLUSTER,
                "use_domain": USE_DOMAIN,
                "num_clusters": NUM_CLUSTERS,
                "num_classes": NUM_CLASSES,
                "clustering_step": CLUSTERING_STEP,
                "classifier_weight": CLASSIFIER_WEIGHT,
                "discriminator_weight": DISCRIMINATOR_WEIGHT,
                "extractor_weight": EXTRACTOR_WEIGHT,
                "entropy_weight": ENTROPY_WEIGHT,
                "grl_weight": GRL_WEIGHT,
                "label_smooth": LABEL_SMOOTH,
                "weight_decay": WEIGHT_DECAY,
                "amsgrad": AMSGRAD,
                "t_max": T_MAX,
                "single_layer": SINGLE_LAYER,
                "experiment_num": EXPERIMENT_NUM,
            },
        )

        try:
            # Setup paths
            # Create structured experiment directory
            experiment_dir = create_experiment_directory(
                base_dir="ProperTraining",
                target_domain=DOMAIN_LIST[TARGET_INDEX],
                experiment_name=EXPERIMENT_NUM,
                use_clustering=False,  # main.py doesn't use clustering
                epochs=NUM_EPOCH,
            )

            # Get log file paths
            log_files = get_experiment_log_files(experiment_dir)

            # Legacy compatibility
            SAVE_PATH = experiment_dir

            print(f"📁 Experiment Directory: {experiment_dir}")
            print(f"📂 Log Files: {list(log_files.keys())}")

            # Calculate domain and label weights (excluding target domain)
            domain_weights_filtered = [
                w for i, w in enumerate(DOMAIN_REWEIGHTING) if i != TARGET_INDEX
            ]
            label_weights_filtered = [
                w for i, w in enumerate(LABEL_REWEIGHTING) if i != TARGET_INDEX
            ]

            DOMAIN_WEIGHT = np.array(domain_weights_filtered)
            DOMAIN_WEIGHT = 1 - (DOMAIN_WEIGHT / DOMAIN_WEIGHT.sum())

            LABEL_WEIGHT = np.array(label_weights_filtered)
            LABEL_WEIGHT = LABEL_WEIGHT.mean(axis=0)

            print(f"📊 Domain weights: {DOMAIN_WEIGHT}")
            print(f"🏷️  Label weights: {LABEL_WEIGHT}")

            # Prepare domain splits
            source_domain, target_domain = split_domain(DOMAIN_LIST, TARGET_INDEX)

            print(f"📂 Loading datasets...")
            print(f"  📦 Source domains: {source_domain}")
            print(f"  🎯 Target domain: {target_domain}")

            # Load data
            source_train, source_val, target_test = random_split_dataloader(
                dir_path=DATA_PATH,
                source_dir=source_domain,
                target_dir=target_domain,
                source_domain=list(range(len(source_domain))),
                target_domain=[len(source_domain)],
                batch_size=BATCH_SIZE,
                get_cluster=USE_CLUSTER,
                get_domain=USE_DOMAIN,
                chunk=True,
                n_workers=0,
            )

            print(f"✅ Data loaded successfully!")
            print(f"  🚂 Train batches: {len(source_train)}")
            print(f"  🔍 Val batches: {len(source_val)}")
            print(f"  🎯 Test batches: {len(target_test)}")

            # Initialize model
            discriminator_dimensions = (
                NUM_CLUSTERS if USE_CLUSTER else len(source_domain)
            )

            model = (
                momlnids(
                    input_nodes=39,
                    hidden_nodes=HIDDEN_NODES,
                    classifier_nodes=CLASS_NODES,
                    num_domains=discriminator_dimensions,
                    num_class=NUM_CLASSES,
                    single_layer=SINGLE_LAYER,
                )
                .double()
                .to(device)
            )

            model.apply(init_weights)

            print(f"🧠 Model initialized!")
            print(
                f"  📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}"
            )
            print(f"  🎯 Discriminator dims: {discriminator_dimensions}")

            # Setup loss weights
            DISCRIMINATOR_LOSS_WEIGHT = (
                torch.tensor(DOMAIN_WEIGHT, dtype=torch.double, device=device)
                if USE_DOMAIN
                else None
            )
            CLASS_LOSS_WEIGHT = torch.tensor(
                LABEL_WEIGHT, dtype=torch.double, device=device
            )

            # Initialize optimizers
            model_learning_rate = get_model_learning_rate(
                model,
                extractor_weight=EXTRACTOR_WEIGHT,
                classifier_weight=CLASSIFIER_WEIGHT,
                discriminator_weight=DISCRIMINATOR_WEIGHT,
            )

            optimizers = [
                get_optimizer(
                    model_module,
                    INIT_LEARNING_RATE * alpha,
                    weight_decay=WEIGHT_DECAY,
                    amsgrad=AMSGRAD,
                )
                for model_module, alpha in model_learning_rate
            ]

            model_schedulers = [
                get_learning_rate_scheduler(optimizer=opt, t_max=T_MAX)
                for opt in optimizers
            ]

            print(f"⚙️  Optimizers and schedulers initialized!")

            # Training loop
            best_accuracy = 0.0
            test_accuracy = 0.0
            best_epoch = 0

            print(f"\n🚀 Starting training...")
            print("=" * 80)

            for epoch in tqdm(range(NUM_EPOCH), disable=VERBOSE):
                print(f"\n📈 Epoch {epoch + 1}/{NUM_EPOCH}")
                print(f"  ⚡ Learning Rate: {optimizers[0].param_groups[0]['lr']:.6f}")
                print(
                    f"  🏆 Current Best: {best_accuracy:.4f} (Test: {test_accuracy:.4f}) at Epoch {best_epoch + 1}"
                )

                source_dataset = source_train.dataset.dataset

                # Pseudo-labeling clustering
                if USE_CLUSTER and epoch % CLUSTERING_STEP == 0:
                    print(f"  🔄 Running clustering step...")
                    pseudo_domain_label = pseudolabeling(
                        dataset=source_dataset,
                        model=model,
                        device=device,
                        previous_cluster=source_dataset.cluster_label,
                        log_file=log_files["clustering"],
                        epoch=epoch,
                        n_clusters=NUM_CLUSTERS,
                        method="MiniK",
                        data_reduction=False,
                        reduced_dimentions=48,
                        batch_size=BATCH_SIZE,
                    )
                    source_dataset.cluster_label = pseudo_domain_label
                    print(f"  ✅ Clustering completed!")

                # Training
                model, optimizers = train(
                    model=model,
                    train_data=source_train,
                    optimizers=optimizers,
                    device=device,
                    epoch=epoch,
                    num_epoch=NUM_EPOCH,
                    filename=log_files["source_trained"],
                    disc_weight=DISCRIMINATOR_LOSS_WEIGHT,
                    class_weight=CLASS_LOSS_WEIGHT,
                    label_smooth=LABEL_SMOOTH,
                    entropy_weight=ENTROPY_WEIGHT,
                    grl_weight=GRL_WEIGHT,
                    wandb_enabled=True,
                    verbose=False,  # Reduce training spam
                )

                # Evaluation
                if epoch % EVAL_STEP == 0:
                    val_accuracy = eval(
                        model=model,
                        eval_data=source_val,
                        device=device,
                        epoch=epoch,
                        filename=log_files["val_performance"],
                        wandb_enabled=True,
                        verbose=False,  # Reduce eval spam
                    )

                    target_accuracy = eval(
                        model=model,
                        eval_data=target_test,
                        device=device,
                        epoch=epoch,
                        filename=log_files["target_performance"],
                        wandb_enabled=True,
                        verbose=False,  # Reduce eval spam
                    )

                    # Save best model
                    if val_accuracy >= best_accuracy:
                        best_accuracy = val_accuracy
                        test_accuracy = target_accuracy
                        best_epoch = epoch
                        model_save_path = get_model_checkpoint_path(
                            experiment_dir, is_best=True
                        )
                        torch.save(model.state_dict(), model_save_path)

                    # Save epoch checkpoint every 2 epochs
                    if epoch % 2 == 0:
                        epoch_model_path = get_model_checkpoint_path(
                            experiment_dir, epoch=epoch
                        )
                        torch.save(model.state_dict(), epoch_model_path)

                    # Clean summary per epoch
                    cluster_info = (
                        f" | Clustering: K={NUM_CLUSTERS}" if USE_CLUSTER else ""
                    )
                    print(
                        f"📈 Epoch {epoch + 1:2d}/{NUM_EPOCH} | Val: {val_accuracy:.4f} | Test: {target_accuracy:.4f} | Best: {best_accuracy:.4f}{cluster_info}"
                    )
                    if val_accuracy >= best_accuracy:
                        print(f"   💾 New best model!")

                # Update learning rate schedulers
                for scheduler in model_schedulers:
                    scheduler.step()

            print("\n" + "=" * 80)
            print(f"🎉 Training completed for {DOMAIN_LIST[TARGET_INDEX]}!")
            print(
                f"🏆 Best validation accuracy: {best_accuracy:.4f} at epoch {best_epoch + 1}"
            )
            print(f"🎯 Final test accuracy: {test_accuracy:.4f}")
            print(f"📁 Experiment directory: {experiment_dir}")
            print(
                f"💾 Best model saved: {get_model_checkpoint_path(experiment_dir, is_best=True)}"
            )
            print(f"📋 Logs saved in: {list(log_files.values())}")
            print("=" * 80)

        except Exception as e:
            print(f"❌ Error during training for {DOMAIN_LIST[TARGET_INDEX]}: {e}")
            raise
        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
