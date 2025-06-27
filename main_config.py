import numpy as np
import torch
import yaml
import argparse
import yaml
import argparse

from skripsi_code.utils.utils import (
    get_model_learning_rate,
    get_learning_rate_scheduler,
    get_optimizer,
    split_domain,
)
from skripsi_code.utils.dataloader import random_split_dataloader
from skripsi_code.clustering.cluster_utils import pseudolabeling
from skripsi_code.TrainEval.TrainEval import train, eval
from skripsi_code.model.MoMLNIDS import MoMLDNIDS
from pathlib import Path
from tqdm import tqdm
from torch import nn

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_normal_(m.weight)

import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MoMLNIDS training with a YAML configuration file.")
    parser.add_argument("--config", type=str, default="config/experiment_config.yaml",
                        help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    exp_config = config['experiment']
    domain_list_cfg = config['domain_list']
    domain_reweighting_cfg = config['domain_reweighting']
    label_reweighting_cfg = config['label_reweighting']
    hidden_nodes_cfg = config['hidden_nodes']
    class_nodes_cfg = config['class_nodes']

    # Extract parameters from config
    VERBOSE = exp_config['verbose']
    SINGLE_LAYER = exp_config['single_layer']
    TARGET_INDEX = exp_config['target_index']
    BATCH_SIZE = exp_config['batch_size']
    EXPERIMENT_NUM = exp_config['experiment_num']
    NUM_EPOCH = exp_config['num_epoch']
    EVAL_STEP = exp_config['eval_step']
    SAVE_STEP = exp_config['save_step']
    INIT_LEARNING_RATE = exp_config['init_learning_rate']
    CLASSIFIER_WEIGHT = exp_config['classifier_weight']
    DISCRIMINATOR_WEIGHT = exp_config['discriminator_weight']
    EXTRACTOR_WEIGHT = exp_config['extractor_weight']
    ENTROPY_WEIGHT = exp_config['entropy_weight']
    GRL_WEIGHT = exp_config['grl_weight']
    LABEL_SMOOTH = exp_config['label_smooth']
    WEIGHT_DECAY = exp_config['weight_decay']
    AMSGRAD = exp_config['amsgrad']
    T_MAX = exp_config['t_max']
    NUM_CLUSTERS = exp_config['num_clusters']
    NUM_CLASSES = exp_config['num_classes']
    CLUSTERING_STEP = exp_config['clustering_step']
    DATA_PATH = exp_config['data_path']
    INPUT_NODES = exp_config['input_nodes']
    EVAL_BATCH_FREQUENCY = exp_config['eval_batch_frequency']

    # Use the loaded domain_list
    DOMAIN_LIST = domain_list_cfg

    # Use the loaded reweighting values
    DOMAIN_REWEIGHTING = domain_reweighting_cfg
    LABEL_REWEIGHTING = label_reweighting_cfg

    # Use the loaded node configurations
    HIDDEN_NODES = hidden_nodes_cfg
    CLASS_NODES = class_nodes_cfg

    # The loop for `i` is removed as target_index is now read from config
    # The loop for `i` is removed as target_index is now read from config
    wandb.init(project="MoMLNIDS_Training", name=f"Experiment_{TARGET_INDEX}")

    RESULT_FOLDER = f"ProperTraining/{DOMAIN_LIST[TARGET_INDEX]}"

    # Adjust DOMAIN_REWEIGHTING and LABEL_REWEIGHTING based on TARGET_INDEX
    current_domain_reweighting = list(DOMAIN_REWEIGHTING)
    current_label_reweighting = [list(x) for x in LABEL_REWEIGHTING] # Convert tuples to lists for modification

    del current_domain_reweighting[TARGET_INDEX]
    del current_label_reweighting[TARGET_INDEX]

    DOMAIN_WEIGHT = np.array(current_domain_reweighting)
    DOMAIN_WEIGHT = 1 - (DOMAIN_WEIGHT / DOMAIN_WEIGHT.sum())

    LABEL_WEIGHT = np.array(current_label_reweighting)
    LABEL_WEIGHT = LABEL_WEIGHT.mean(axis=0)

    print(f"{DOMAIN_WEIGHT=}, {LABEL_WEIGHT=}")

    SAVE_PATH = Path(
        f"./{RESULT_FOLDER}/{DOMAIN_LIST[TARGET_INDEX]}_N{EXPERIMENT_NUM}"
    )
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    USE_CLUSTER = exp_config['use_cluster']
    USE_DOMAIN = not USE_CLUSTER
    print(f"{USE_DOMAIN=}, {USE_CLUSTER=}, {device=}\n")

    # CLASS_LOSS_WEIGHT = None
    # DISCRIMINATOR_LOSS_WEIGHT = None
    DISCRIMINATOR_LOSS_WEIGHT = torch.tensor(DOMAIN_WEIGHT, dtype=torch.double, device=device) if USE_DOMAIN else None
    CLASS_LOSS_WEIGHT = torch.tensor(LABEL_WEIGHT, dtype=torch.double, device=device)

    INPUT_NODES = exp_config['input_nodes']
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

    model = MoMLDNIDS(
        input_nodes=INPUT_NODES,
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

    model_schedulers = [
        get_learning_rate_scheduler(optimizer=opt, t_max=T_MAX) for opt in optimizers
    ]

    best_accuracy = 0.0
    test_accuracy = 0.0
    best_epoch = 0
    global_step = 0

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

        # Modify train function to return number of batches processed
        model, optimizers, num_batches = train(
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

        # Update global step after each training epoch
        global_step += num_batches

        if global_step % EVAL_BATCH_FREQUENCY == 0:
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

            if val_accuracy >= best_accuracy:
                best_accuracy = val_accuracy
                test_accuracy = target_accuracy
                best_epoch = epoch
                torch.save(model.state_dict(), SAVE_PATH / "model_best.pt")

        for scheduler in model_schedulers:
            scheduler.step()

    wandb.finish()

