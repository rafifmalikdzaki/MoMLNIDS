import sys
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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

import wandb
import uuid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MoMLNIDS training with a YAML configuration file.")
    parser.add_argument("--config", type=str, default="config/experiment_config.yaml",
                        help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config = get_config()
    if args.config != "config/default_config.yaml":
        with open(args.config, 'r') as f:
            override_config = yaml.safe_load(f)
        config_manager.update_config(override_config)
        config = get_config() # Reload config after update

    # Extract parameters from config
    # Project settings
    PROJECT_NAME = config.project.name
    PROJECT_DESCRIPTION = config.project.description
    PROJECT_VERSION = config.project.version

    # Wandb settings
    WANDB_ENABLED = config.wandb.enabled
    WANDB_PROJECT = config.wandb.project
    WANDB_ENTITY = config.wandb.entity
    WANDB_TAGS = config.wandb.tags

    # Model configuration
    MODEL_NAME = config.model.name
    FEATURE_EXTRACTOR_HIDDEN_LAYERS = config.model.feature_extractor.hidden_layers
    FEATURE_EXTRACTOR_DROPOUT_RATE = config.model.feature_extractor.dropout_rate
    FEATURE_EXTRACTOR_ACTIVATION = config.model.feature_extractor.activation
    CLASSIFIER_HIDDEN_LAYERS = config.model.classifier.hidden_layers
    NUM_CLASSES = config.model.classifier.num_classes
    CLASSIFIER_DROPOUT_RATE = config.model.classifier.dropout_rate
    DISCRIMINATOR_HIDDEN_LAYERS = config.model.discriminator.hidden_layers
    NUM_DOMAINS = config.model.discriminator.num_domains
    DISCRIMINATOR_DROPOUT_RATE = config.model.discriminator.dropout_rate

    # Training configuration
    BATCH_SIZE = config.training.batch_size
    NUM_EPOCH = config.training.epochs
    INIT_LEARNING_RATE = config.training.learning_rate
    WEIGHT_DECAY = float(config.training.weight_decay)
    DOMAIN_WEIGHT = config.training.domain_weight
    CLASSIFIER_WEIGHT = config.training.classification_weight
    MAX_BATCHES = config.training.get('max_batches', None)  # Optional parameter for limiting batches
    GRADIENT_CLIP_NORM = config.training.gradient_clip_norm
    
    # Early stopping
    EARLY_STOPPING_ENABLED = config.training.early_stopping.enabled
    EARLY_STOPPING_PATIENCE = config.training.early_stopping.patience
    EARLY_STOPPING_MIN_DELTA = config.training.early_stopping.min_delta

    # Checkpoint saving
    SAVE_EVERY = config.training.checkpoint.save_every
    SAVE_BEST = config.training.checkpoint.save_best

    # Data configuration
    DATASETS = config.data.datasets
    RAW_DATA_PATH = config.data.raw_data_path
    INTERIM_DATA_PATH = config.data.interim_data_path
    PROCESSED_DATA_PATH = config.data.processed_data_path
    NORMALIZE_DATA = config.data.preprocessing.normalize
    HANDLE_MISSING_DATA = config.data.preprocessing.handle_missing
    FEATURE_SELECTION = config.data.preprocessing.feature_selection
    TRAIN_RATIO = config.data.data_splits.train_ratio
    VAL_RATIO = config.data.data_splits.val_ratio
    TEST_RATIO = config.data.data_splits.test_ratio

    # Clustering configuration
    CLUSTERING_METHOD = config.data.clustering.method
    NUM_CLUSTERS = config.data.clustering.n_clusters
    CLUSTERING_RANDOM_STATE = config.data.clustering.random_state

    # Evaluation configuration
    EVALUATION_METRICS = config.evaluation.metrics
    CROSS_VALIDATION_ENABLED = config.evaluation.cross_validation.enabled
    CROSS_VALIDATION_FOLDS = config.evaluation.cross_validation.folds

    # Explainable AI configuration
    EXPLAINABLE_AI_ENABLED = config.explainable_ai.enabled
    EXPLAINABLE_AI_METHODS = config.explainable_ai.methods
    FEATURE_IMPORTANCE_ENABLED = config.explainable_ai.feature_importance.enabled
    TOP_K_FEATURES = config.explainable_ai.feature_importance.top_k_features
    SAVE_PLOTS = config.explainable_ai.visualization.save_plots
    PLOT_FORMAT = config.explainable_ai.visualization.plot_format
    DPI = config.explainable_ai.visualization.dpi

    # Logging configuration
    LOGGING_LEVEL = config.logging.level
    LOGGING_FORMAT = config.logging.format
    LOGGING_FILE = config.logging.file

    # Output paths
    MODELS_DIR = config.output.models_dir
    RESULTS_DIR = config.output.results_dir
    PLOTS_DIR = config.output.plots_dir
    LOGS_DIR = config.output.logs_dir

    # Reproducibility
    RANDOM_SEED = config.random_seed
    DETERMINISTIC = config.deterministic

    # Hardware configuration
    USE_CUDA = config.device.use_cuda
    CUDA_DEVICE = config.device.cuda_device
    NUM_WORKERS = config.device.num_workers

    TARGET_INDEX = config.training.target_index
    EXPERIMENT_NUM = config.training.experiment_num
    INPUT_NODES = config.model.input_nodes
    HIDDEN_NODES = config.model.feature_extractor.hidden_layers
    CLASS_NODES = config.model.classifier.hidden_layers
    SINGLE_LAYER = config.model.single_layer
    EXTRACTOR_WEIGHT = config.training.extractor_weight
    CLASSIFIER_WEIGHT = config.training.classifier_weight
    DISCRIMINATOR_WEIGHT = config.training.discriminator_weight
    AMSGRAD = config.training.amsgrad
    T_MAX = config.training.t_max
    VERBOSE = config.logging.verbose
    CLUSTERING_STEP = config.training.clustering_step
    LABEL_SMOOTH = config.training.label_smooth
    ENTROPY_WEIGHT = config.training.entropy_weight
    GRL_WEIGHT = config.training.grl_weight
    EVAL_BATCH_FREQUENCY = config.training.eval_batch_frequency

    # The loop for `i` is removed as target_index is now read from config
    # The loop for `i` is removed as target_index is now read from config
    if WANDB_ENABLED:
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, tags=WANDB_TAGS, group=DOMAIN_LIST[TARGET_INDEX])
        if wandb.run is None:
            print("Wandb initialization failed: wandb.run is None after init.")
        else:
            print(f"Wandb initialized successfully. Run ID: {wandb.run.id}")
        wandb.config.update(OmegaConf.to_container(config, resolve=True))

    RESULT_FOLDER = f"{config.output.results_dir}/{DOMAIN_LIST[TARGET_INDEX]}"

    # Adjust DOMAIN_REWEIGHTING and LABEL_REWEIGHTING based on TARGET_INDEX
    current_domain_reweighting = list(config.data.domain_reweighting)
    current_label_reweighting = [list(x) for x in config.data.label_reweighting] # Convert tuples to lists for modification

    del current_domain_reweighting[TARGET_INDEX]
    del current_label_reweighting[TARGET_INDEX]

    DOMAIN_WEIGHT = np.array(current_domain_reweighting)
    DOMAIN_WEIGHT = 1 - (DOMAIN_WEIGHT / DOMAIN_WEIGHT.sum())

    LABEL_WEIGHT = np.array(current_label_reweighting)
    LABEL_WEIGHT = LABEL_WEIGHT.mean(axis=0)

    print(f"{DOMAIN_WEIGHT=}, {LABEL_WEIGHT=}")

    SAVE_PATH = Path(
        f"{config.output.models_dir}/{DOMAIN_LIST[TARGET_INDEX]}_N{EXPERIMENT_NUM}"
    )
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if config.device.use_cuda and torch.cuda.is_available() else "cpu")

    USE_CLUSTER = config.data.clustering.method == "Kmeans"
    USE_DOMAIN = not USE_CLUSTER
    print(f"{USE_DOMAIN=}, {USE_CLUSTER=}, {device=}")

    # CLASS_LOSS_WEIGHT = None
    # DISCRIMINATOR_LOSS_WEIGHT = None
    DISCRIMINATOR_LOSS_WEIGHT = torch.tensor(DOMAIN_WEIGHT, dtype=torch.double, device=device) if USE_DOMAIN else None
    CLASS_LOSS_WEIGHT = torch.tensor(LABEL_WEIGHT, dtype=torch.double, device=device)

    # INPUT_NODES = exp_config['input_nodes'] # Already mapped above
    DATA_PATH = config.data.processed_data_path
    
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
        n_workers=NUM_WORKERS,
    )

    discriminator_dimensions = NUM_CLUSTERS if USE_CLUSTER else len(source_domain)

    model = momlnids(
        input_nodes=INPUT_NODES,
        hidden_nodes=HIDDEN_NODES,
        classifier_nodes=CLASS_NODES,
        num_domains=discriminator_dimensions,
        num_class=NUM_CLASSES,
        single_layer=SINGLE_LAYER
    ).double().to(device)

    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()

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
                    log_file=Path(config.output.logs_dir) / "clustering.log", # Use LOGS_DIR
                    epoch=epoch,
                    n_clusters=NUM_CLUSTERS,
                    method=CLUSTERING_METHOD,
                    data_reduction=False,
                    reduced_dimentions=48,
                    batch_size=BATCH_SIZE,
                )

                source_dataset.cluster_label = pseudo_domain_label

        # Modify train function to return number of batches batches processed
        model, optimizers, num_batches = train(
            model=model,
            train_data=source_train,
            optimizers=optimizers,
            device=device,
            epoch=epoch,
            num_epoch=NUM_EPOCH,
            filename=Path(config.output.logs_dir) / "source_trained.log", # Use LOGS_DIR
            disc_weight=DISCRIMINATOR_LOSS_WEIGHT,
            class_weight=CLASS_LOSS_WEIGHT,
            label_smooth=LABEL_SMOOTH,
            entropy_weight=ENTROPY_WEIGHT,
            grl_weight=GRL_WEIGHT,
            max_batches=MAX_BATCHES,
            wandb_enabled=WANDB_ENABLED,
            clip_grad_norm=GRADIENT_CLIP_NORM,
        )

        # Update global step after each training epoch
        global_step += num_batches

        if global_step % EVAL_BATCH_FREQUENCY == 0:
            val_loader = source_val
            val_accuracy = eval(
                model=model,
                eval_data=val_loader,
                criterion=criterion,
                device=device,
                num_epoch=NUM_EPOCH,
                filename=Path(config.output.logs_dir) / "validation.log", # Use LOGS_DIR
                wandb_enabled=WANDB_ENABLED,
                epoch=epoch,
            )

            target_accuracy = eval(
            model=model,
            eval_data=target_test,
            criterion=criterion,
            device=device,
            num_epoch=NUM_EPOCH,
            filename=Path(config.output.logs_dir) / "target_test.log", # Use LOGS_DIR
            wandb_enabled=WANDB_ENABLED,
            epoch=epoch,
        )

            if val_accuracy >= best_accuracy:
                best_accuracy = val_accuracy
                test_accuracy = target_accuracy
                best_epoch = epoch
                torch.save(model.state_dict(), Path(config.output.models_dir) / "model_best.pt") # Use MODELS_DIR

        for scheduler in model_schedulers:
            scheduler.step()

    if config.wandb.enabled:
        wandb.finish()
