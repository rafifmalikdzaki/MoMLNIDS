#!/usr/bin/env python3
"""
MoMLNIDS Extended Pseudo-Labeling Training Script (50 Epochs)

This script runs extended training with pseudo-labeling clustering for 50 epochs
to achieve better convergence and more stable results.
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
    """Main extended pseudo-labeling training function (50 epochs)."""

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

    # Extended training configuration
    BATCH_SIZE = 1
    NUM_EPOCH = 50  # Extended to 50 epochs
    EVAL_STEP = 2  # Evaluate every 2 epochs for efficiency
    SAVE_STEP = 5  # Save every 5 epochs
    INIT_LEARNING_RATE = 0.0015

    # Training weights
    CLASSIFIER_WEIGHT = 1
    DISCRIMINATOR_WEIGHT = 1
    EXTRACTOR_WEIGHT = 1

    # Loss configuration
    ENTROPY_WEIGHT = 1
    GRL_WEIGHT = 1.25
    LABEL_SMOOTH = 0.1

    # Optimizer settings (adjusted for longer training)
    WEIGHT_DECAY = 5e-4
    AMSGRAD = True
    T_MAX = 20  # Longer cosine annealing cycle

    # Domain and clustering settings
    USE_CLUSTER = True  # Enable pseudo-labeling
    USE_DOMAIN = not USE_CLUSTER
    NUM_CLUSTERS = 4  # Fixed to 4 clusters for extended training
    NUM_CLASSES = 2
    CLUSTERING_STEP = 3  # Cluster every 3 epochs for stability

    EXPERIMENT_NUM = "|PseudoLabelling50Epoch|"
    DATA_PATH = "./src/skripsi_code/data/parquet/"

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print(f"🚀 MoMLNIDS Extended Pseudo-Labeling Training (50 Epochs)")
    print("=" * 100)
    print(f"📈 Extended Epochs: {NUM_EPOCH}")
    print(f"🔢 Clusters: {NUM_CLUSTERS}")
    print(f"🎯 Target domains: {DOMAIN_LIST}")
    print(f"🔬 Experiment: {EXPERIMENT_NUM}")
    print(f"🧠 Model: Single Layer={SINGLE_LAYER}")
    print(f"🎯 Device: {device}")
    print(f"📊 Evaluation every {EVAL_STEP} epochs")
    print(f"💾 Checkpoints every {SAVE_STEP} epochs")
    print("=" * 100)

    # Run training for each target domain
    for TARGET_INDEX in range(len(DOMAIN_LIST)):
        print(
            f"\n📊 Extended Training - Target: {DOMAIN_LIST[TARGET_INDEX]} (Index: {TARGET_INDEX})"
        )
        print(
            f"📦 Source Domains: {[d for i, d in enumerate(DOMAIN_LIST) if i != TARGET_INDEX]}"
        )

        # Initialize Wandb
        wandb.init(
            project="MoMLNIDS_Extended_Training",
            group=f"{DOMAIN_LIST[TARGET_INDEX]}_50EP",
            tags=[EXPERIMENT_NUM, "extended", "50_epochs"],
            name=f"{DOMAIN_LIST[TARGET_INDEX]}_50EP_{uuid.uuid4().hex[:8]}",
            config={
                "target_domain": DOMAIN_LIST[TARGET_INDEX],
                "source_domains": [
                    d for idx, d in enumerate(DOMAIN_LIST) if idx != TARGET_INDEX
                ],
                "num_clusters": NUM_CLUSTERS,
                "batch_size": BATCH_SIZE,
                "num_epoch": NUM_EPOCH,
                "init_learning_rate": INIT_LEARNING_RATE,
                "hidden_nodes": HIDDEN_NODES,
                "class_nodes": CLASS_NODES,
                "use_cluster": USE_CLUSTER,
                "use_domain": USE_DOMAIN,
                "num_classes": NUM_CLASSES,
                "clustering_step": CLUSTERING_STEP,
                "eval_step": EVAL_STEP,
                "save_step": SAVE_STEP,
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
            RESULT_FOLDER = f"ProperTraining50Epoch/{DOMAIN_LIST[TARGET_INDEX]}"
            SAVE_PATH = Path(
                f"./{RESULT_FOLDER}/{DOMAIN_LIST[TARGET_INDEX]}_N{EXPERIMENT_NUM}"
            )
            SAVE_PATH.mkdir(parents=True, exist_ok=True)

            print(f"💾 Extended results: {SAVE_PATH}")

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

            print(f"📂 Loading datasets for extended training...")
            print(f"  📦 Source: {source_domain}")
            print(f"  🎯 Target: {target_domain}")

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

            print(f"✅ Data loaded for extended training!")
            print(f"  🚂 Train: {len(source_train)} batches")
            print(f"  🔍 Val: {len(source_val)} batches")
            print(f"  🎯 Test: {len(target_test)} batches")

            # Initialize model
            model = (
                momlnids(
                    input_nodes=39,
                    hidden_nodes=HIDDEN_NODES,
                    classifier_nodes=CLASS_NODES,
                    num_domains=NUM_CLUSTERS,  # Use cluster count for discriminator
                    num_class=NUM_CLASSES,
                    single_layer=SINGLE_LAYER,
                )
                .double()
                .to(device)
            )

            model.apply(init_weights)

            print(f"🧠 Extended model initialized!")
            print(f"  📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  🎯 Discriminator dims: {NUM_CLUSTERS}")

            # Setup loss weights
            DISCRIMINATOR_LOSS_WEIGHT = None  # Not used with clustering
            CLASS_LOSS_WEIGHT = torch.tensor(
                LABEL_WEIGHT, dtype=torch.double, device=device
            )

            # Initialize optimizers (with longer annealing)
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

            print(f"⚙️  Extended optimizers ready! (T_max={T_MAX})")

            # Training loop (extended)
            best_accuracy = 0.0
            test_accuracy = 0.0
            best_epoch = 0

            print(f"\n🚀 Starting extended training ({NUM_EPOCH} epochs)...")

            for epoch in tqdm(
                range(NUM_EPOCH), disable=VERBOSE, desc="Extended Training"
            ):
                print(f"\n📈 Epoch {epoch + 1}/{NUM_EPOCH}")
                print(f"  ⚡ LR: {optimizers[0].param_groups[0]['lr']:.6f}")
                print(
                    f"  🏆 Best: {best_accuracy:.4f} (Test: {test_accuracy:.4f}) @ Epoch {best_epoch + 1}"
                )

                source_dataset = source_train.dataset.dataset

                # Pseudo-labeling clustering (less frequent for stability)
                if epoch % CLUSTERING_STEP == 0:
                    try:
                        pseudo_domain_label = pseudolabeling(
                            dataset=source_dataset,
                            model=model,
                            device=device,
                            previous_cluster=source_dataset.cluster_label,
                            log_file=SAVE_PATH / "clustering.log",
                            epoch=epoch,
                            n_clusters=NUM_CLUSTERS,
                            method="MiniK",
                            data_reduction=False,
                            reduced_dimentions=48,
                            batch_size=BATCH_SIZE,
                        )
                        source_dataset.cluster_label = pseudo_domain_label
                    except Exception as e:
                        print(f"  ⚠️  Clustering failed: {e}")

                # Training
                model, optimizers = train(
                    model=model,
                    train_data=source_train,
                    optimizers=optimizers,
                    device=device,
                    epoch=epoch,
                    num_epoch=NUM_EPOCH,
                    filename=SAVE_PATH / "source_trained.log",
                    disc_weight=DISCRIMINATOR_LOSS_WEIGHT,
                    class_weight=CLASS_LOSS_WEIGHT,
                    label_smooth=LABEL_SMOOTH,
                    entropy_weight=ENTROPY_WEIGHT,
                    grl_weight=GRL_WEIGHT,
                    wandb_enabled=True,
                    verbose=False,  # Reduce training spam
                )

                # Evaluation (less frequent for efficiency)
                if epoch % EVAL_STEP == 0:
                    val_accuracy = eval(
                        model=model,
                        eval_data=source_val,
                        device=device,
                        epoch=epoch,
                        filename=SAVE_PATH / "val_performance.log",
                        wandb_enabled=True,
                        verbose=False,  # Reduce eval spam
                    )

                    target_accuracy = eval(
                        model=model,
                        eval_data=target_test,
                        device=device,
                        epoch=epoch,
                        filename=SAVE_PATH / "target_performance.log",
                        wandb_enabled=True,
                        verbose=False,  # Reduce eval spam
                    )

                    # Save best model
                    if val_accuracy >= best_accuracy:
                        best_accuracy = val_accuracy
                        test_accuracy = target_accuracy
                        best_epoch = epoch
                        torch.save(model.state_dict(), SAVE_PATH / "model_best.pt")

                    # Clean summary per epoch (only on eval steps)
                    print(
                        f"📈 Epoch {epoch + 1:2d}/{NUM_EPOCH} | Extended | Val: {val_accuracy:.4f} | Test: {target_accuracy:.4f} | Best: {best_accuracy:.4f}"
                    )
                    if val_accuracy >= best_accuracy:
                        print(f"   💾 New best model!")

                # Save periodic checkpoints (less frequent)
                if epoch % SAVE_STEP == 0:
                    torch.save(model.state_dict(), SAVE_PATH / f"model_{epoch}.pt")

                # Update learning rate schedulers
                for scheduler in model_schedulers:
                    scheduler.step()

            # Final evaluation
            print(f"\n🏁 Final evaluation...")
            final_val_accuracy = eval(
                model=model,
                eval_data=source_val,
                device=device,
                epoch=NUM_EPOCH - 1,
                filename=SAVE_PATH / "final_val_performance.log",
                wandb_enabled=True,
            )

            final_test_accuracy = eval(
                model=model,
                eval_data=target_test,
                device=device,
                epoch=NUM_EPOCH - 1,
                filename=SAVE_PATH / "final_target_performance.log",
                wandb_enabled=True,
            )

            # Save final model
            torch.save(model.state_dict(), SAVE_PATH / "model_final.pt")

            print(f"\n✅ Extended training completed!")
            print(f"🏆 Best val: {best_accuracy:.4f} @ epoch {best_epoch + 1}")
            print(f"🎯 Best test: {test_accuracy:.4f}")
            print(f"📊 Final val: {final_val_accuracy:.4f}")
            print(f"🎯 Final test: {final_test_accuracy:.4f}")
            print(f"💾 Models saved: {SAVE_PATH}")

        except Exception as e:
            print(
                f"❌ Error during extended training for {DOMAIN_LIST[TARGET_INDEX]}: {e}"
            )
            continue
        finally:
            wandb.finish()

    print("\n" + "=" * 100)
    print(f"🎉 All extended training experiments completed!")
    print(f"📈 Ran {NUM_EPOCH} epochs per domain")
    print(f"📊 Check Wandb project 'MoMLNIDS_Extended_Training' for results")
    print("=" * 100)


if __name__ == "__main__":
    main()
