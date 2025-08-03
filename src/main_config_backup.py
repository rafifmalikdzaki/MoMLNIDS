import numpy as np
import torch

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
from pathlib import Path
from tqdm import tqdm
from torch import nn

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_normal_(m.weight)

import wandb
import uuid

    for i in range(3):
        wandb.init(project="MoMLNIDS_Training", group=DOMAIN_LIST[TARGET_INDEX], tags=[EXPERIMENT_NUM])
        wandb.config.update({
            "target_domain": DOMAIN_LIST[TARGET_INDEX],
            "source_domains": [d for idx, d in enumerate(DOMAIN_LIST) if idx != TARGET_INDEX],
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
        })
        wandb.config.update({
            "target_domain": DOMAIN_LIST[TARGET_INDEX],
            "source_domains": [d for idx, d in enumerate(DOMAIN_LIST) if idx != TARGET_INDEX],
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
        })
    # if i := 1:
        VERBOSE = False
        SINGLE_LAYER = True
        DOMAIN_LIST = [
            "NF-UNSW-NB15-v2",
            "NF-CSE-CIC-IDS2018-v2",
            "NF-ToN-IoT-v2",
            # "NF-BoT-IoT-v2",
        ]


        # use the probability
        DOMAIN_REWEIGHTING = [
            23,
            187,
            169,
        ]

        # Use arithmethic mean
        LABEL_REWEIGHTING = [
            # Benign, Attack
            (0.8735, 0.1265),
            (0.8307, 0.1693),
            (0.0356, 0.9644)
            # (0.01, 0.9999)
        ]

        HIDDEN_NODES = [64, 32, 16, 10]
        CLASS_NODES = [64, 32, 16]

        # Variables
        TARGET_INDEX = i
        BATCH_SIZE = 1

        EXPERIMENT_NUM = "|PseudoLabelling|"

        RESULT_FOLDER = f"ProperTraining/{DOMAIN_LIST[TARGET_INDEX]}"
        # RESULT_FOLDER = "Training_results"

        del DOMAIN_REWEIGHTING[TARGET_INDEX]
        del LABEL_REWEIGHTING[TARGET_INDEX]

        DOMAIN_WEIGHT = np.array(DOMAIN_REWEIGHTING)
        DOMAIN_WEIGHT = 1 - (DOMAIN_WEIGHT / DOMAIN_WEIGHT.sum())

        LABEL_WEIGHT = np.array(LABEL_REWEIGHTING)
        LABEL_WEIGHT = LABEL_WEIGHT.mean(axis=0)

        print(f"{DOMAIN_WEIGHT=}, {LABEL_WEIGHT=}")

        SAVE_PATH = Path(
            f"./{RESULT_FOLDER}/{DOMAIN_LIST[TARGET_INDEX]}_N{EXPERIMENT_NUM}"
        )
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        USE_CLUSTER = True
        USE_DOMAIN = not USE_CLUSTER
        print(f"{USE_DOMAIN=}, {USE_CLUSTER=}, {device=}\n")

        NUM_EPOCH = 20
        EVAL_STEP = 1 # Validation every epoch
        SAVE_STEP = 2
        INIT_LEARNING_RATE = 0.0015

        # Training Weights
        CLASSIFIER_WEIGHT = 1
        DISCRIMINATOR_WEIGHT = 1
        EXTRACTOR_WEIGHT = 1

        # CLASS_LOSS_WEIGHT = None
        # DISCRIMINATOR_LOSS_WEIGHT = None
        DISCRIMINATOR_LOSS_WEIGHT = torch.tensor(DOMAIN_WEIGHT, dtype=torch.double, device=device) if USE_DOMAIN else None
        CLASS_LOSS_WEIGHT = torch.tensor(LABEL_WEIGHT, dtype=torch.double, device=device)
        ENTROPY_WEIGHT = 1
        GRL_WEIGHT = 1.25
        LABEL_SMOOTH = 0.1

        # Optimizer
        WEIGHT_DECAY = 5e-4
        AMSGRAD = True

        # Learning Rate Scheduler
        T_MAX = 10

        # Domain Info
        NUM_DOMAINS = len(DOMAIN_LIST) - 1
        NUM_CLUSTERS = 4
        NUM_CLASSES = 2
        CLUSTERING_STEP = 2

        DATA_PATH = "./src/skripsi_code/data/parquet/"

        # Executions
        source_domain, target_domain = split_domain(DOMAIN_LIST, TARGET_INDEX)

        source_train, source_val, target_test = random_split_dataloader(
            dir_path=DATA_PATH,
            source_dir=source_domain,
            target_dir=target_domain,
            source_domain=source_domain,
            target_domain=target_domain,
            batch_size=BATCH_SIZE,
            get_cluster=USE_CLUSTER,
            get_domain=USE_DOMAIN,
            chunk=True,
            n_workers=0,
        )

        discriminator_dimensions = NUM_CLUSTERS if USE_CLUSTER else NUM_DOMAINS

        model = momlnids(
            input_nodes=39,
            hidden_nodes=HIDDEN_NODES,
            classifier_nodes=CLASS_NODES,
            num_domains=discriminator_dimensions,
            num_class=NUM_CLASSES,
            single_layer=SINGLE_LAYER
        ).double().to(device)

        model.apply(init_weights)

        model_learning_rate = get_model_learning_rate(
            model,
            extractor_weight=EXTRACTOR_WEIGHT,
            classifier_weight=CLASSIFIER_WEIGHT,
            discriminator_weight=DISCRIMINATOR_WEIGHT,
        )

        # AdamW optimizer
        optimizers = [
            get_optimizer(
                model_module,
                INIT_LEARNING_RATE * alpha,
                weight_decay=WEIGHT_DECAY,
                amsgrad=AMSGRAD,
            )
            for model_module, alpha in model_learning_rate
        ]

        # print(optimizers)

        model_schedulers = [
            get_learning_rate_scheduler(optimizer=opt, t_max=T_MAX) for opt in optimizers
        ]

        best_accuracy = 0.0
        test_accuracy = 0.0
        best_epoch = 0

        for epoch in tqdm(range(NUM_EPOCH), disable=VERBOSE):
            print(
                f"Epoch: {epoch + 1}/{NUM_EPOCH}, Learning Rate: {optimizers[0].param_groups[0]['lr']:.6f}"
            )
            print(
                f"Temporary Best Accuracy: {test_accuracy:.4f} (Best: {best_accuracy:.4f} at Epoch {best_epoch})"
            )

            source_dataset = source_train.dataset.dataset

            if USE_CLUSTER:
                if epoch % CLUSTERING_STEP == 0:
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
            )

            if epoch % EVAL_STEP == 0:
                val_accuracy = eval(
                    model=model,
                    eval_data=source_val,
                    device=device,
                    epoch=epoch,
                    filename=SAVE_PATH / "val_performance.log",
                )

                target_accuracy = eval(
                    model=model,
                    eval_data=target_test,
                    device=device,
                    epoch=epoch,
                    filename=SAVE_PATH / "target_performance.log",
                )

            if epoch % SAVE_STEP == 0:
                torch.save(model.state_dict(), SAVE_PATH / f"model_{epoch}.pt")

            if val_accuracy >= best_accuracy:
                best_accuracy = val_accuracy
                test_accuracy = target_accuracy
                best_epoch = epoch
                torch.save(model.state_dict(), SAVE_PATH / "model_best.pt")

            for scheduler in model_schedulers:
                scheduler.step()

        wandb.finish()

        # best_model = MoMLDNIDS(
        #     input_nodes=39,
        #     hidden_nodes=HIDDEN_NODES,
        #     classifier_nodes=CLASS_NODES,
        #     num_domains=discriminator_dimensions,
        #     num_class=NUM_CLASSES,
        # )

        # best_model.load_state_dict(
        #     torch.load(SAVE_PATH / "model_best.pt")
        # )
        # best_model = best_model.double().to(device)
        #
        # final_accuracy = eval(
        #     model=best_model,
        #     eval_data=target_test,
        #     device=device,
        #     epoch=best_epoch,
        #     filename=SAVE_PATH / "target_best_performance.log",
        # )
        #
        # print(
        #     f"Test Accuracy by the best model on the source domain is {final_accuracy} (at Epoch {best_epoch})"
        # )
