#!/usr/bin/env python3
"""
Improved main training script for MoMLNIDS project.

This script demonstrates the integration of:
- Configuration management
- Experiment tracking with wandb
- Enhanced logging and formatting
- Better project organization
"""

import sys
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import wandb
import uuid
from torch import nn
from tqdm import tqdm
import logging

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
from skripsi_code.config import load_config, get_config
from skripsi_code.config.config_manager import config_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DOMAIN_LIST = [
    "NF-CSE-CIC-IDS2018-v2",
    "NF-ToN-IoT-v2",
    "NF-UNSW-NB15-v2",
]


def init_weights(m):
    """Initialize model weights using Xavier normal initialization."""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_normal_(m.weight)


def setup_reproducibility(seed: int = 42):
    """Setup reproducible training environment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Main improved training function with better organization."""
    parser = argparse.ArgumentParser(description="Improved MoMLNIDS Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment_config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=0,
        help="Target domain index (0-2)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom experiment name",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = get_config()
        if args.config != "config/default_config.yaml":
            with open(args.config, "r") as f:
                override_config = yaml.safe_load(f)
            config_manager.update_config(override_config)
            config = get_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Extract parameters
    PROJECT_NAME = config.project.name
    WANDB_ENABLED = config.wandb.enabled
    WANDB_PROJECT = getattr(config.wandb, "project", PROJECT_NAME)
    WANDB_ENTITY = getattr(config.wandb, "entity", None)

    # Training settings
    NUM_EPOCH = config.training.epochs
    BATCH_SIZE = config.training.batch_size
    LEARNING_RATE = config.training.learning_rate
    TARGET_INDEX = (
        args.target_index
        if args.target_index is not None
        else config.training.target_index
    )
    EXPERIMENT_NUM = args.experiment_name or config.training.experiment_num
    USE_CLUSTER = config.training.use_cluster
    EVAL_STEP = config.training.eval_step
    SAVE_STEP = config.training.save_step

    # Model settings
    INPUT_NODES = config.model.input_nodes
    SINGLE_LAYER = config.model.single_layer

    # Data settings
    PROCESSED_DATA_PATH = config.data.processed_data_path
    DOMAIN_REWEIGHTING = getattr(config.data, "domain_reweighting", [1.0, 1.0, 1.0])
    LABEL_REWEIGHTING = getattr(
        config.data, "label_reweighting", [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    )

    # Device settings
    NUM_WORKERS = config.device.num_workers

    # Output settings
    MODELS_DIR = getattr(config.output, "models_dir", "models")
    LOGS_DIR = getattr(config.output, "logs_dir", "logs")

    # Setup
    setup_reproducibility(getattr(config, "random_seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate target index
    if TARGET_INDEX >= len(DOMAIN_LIST):
        logger.error(
            f"Invalid target index {TARGET_INDEX}. Must be 0-{len(DOMAIN_LIST) - 1}"
        )
        return

    # Create output directories
    Path(MODELS_DIR).mkdir(exist_ok=True)
    Path(LOGS_DIR).mkdir(exist_ok=True)

    # Display configuration
    print("=" * 100)
    print(f"🚀 MoMLNIDS Improved Training Script")
    print("=" * 100)
    print(f"📊 Project: {PROJECT_NAME}")
    print(f"🎯 Target Domain: {DOMAIN_LIST[TARGET_INDEX]} (Index: {TARGET_INDEX})")
    print(
        f"📦 Source Domains: {[d for i, d in enumerate(DOMAIN_LIST) if i != TARGET_INDEX]}"
    )
    print(f"🔬 Experiment: {EXPERIMENT_NUM}")
    print(f"📈 Epochs: {NUM_EPOCH}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"⚡ Learning Rate: {LEARNING_RATE}")
    print(f"🧠 Model: Single Layer={SINGLE_LAYER}, Input Nodes={INPUT_NODES}")
    print(f"🔄 Use Clustering: {USE_CLUSTER}")
    print(f"🎯 Device: {device}")
    print(f"💾 Models Dir: {MODELS_DIR}")
    print(f"📝 Logs Dir: {LOGS_DIR}")
    print("=" * 100)

    # Initialize Wandb
    if WANDB_ENABLED:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            group=DOMAIN_LIST[TARGET_INDEX],
            tags=[EXPERIMENT_NUM, "improved"],
            name=f"{DOMAIN_LIST[TARGET_INDEX]}_{EXPERIMENT_NUM}_{uuid.uuid4().hex[:8]}",
            config=dict(config),
        )
        logger.info("Wandb initialized successfully")

    try:
        # Setup paths
        RESULT_FOLDER = f"{MODELS_DIR}/{DOMAIN_LIST[TARGET_INDEX]}"
        SAVE_PATH = Path(
            f"./{RESULT_FOLDER}/{DOMAIN_LIST[TARGET_INDEX]}_N{EXPERIMENT_NUM}"
        )
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

        logger.info(f"Results will be saved to: {SAVE_PATH}")

        # Calculate weights (excluding target domain)
        domain_weights_filtered = [
            w for i, w in enumerate(DOMAIN_REWEIGHTING) if i != TARGET_INDEX
        ]
        label_weights_filtered = [
            w for i, w in enumerate(LABEL_REWEIGHTING) if i != TARGET_INDEX
        ]

        DOMAIN_WEIGHT = np.array(domain_weights_filtered)
        DOMAIN_WEIGHT = (
            1 - (DOMAIN_WEIGHT / DOMAIN_WEIGHT.sum())
            if DOMAIN_WEIGHT.sum() > 0
            else DOMAIN_WEIGHT
        )

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
            dir_path=PROCESSED_DATA_PATH,
            source_dir=source_domain,
            target_dir=target_domain,
            source_domain=list(range(len(source_domain))),
            target_domain=[len(source_domain)],
            batch_size=BATCH_SIZE,
            get_cluster=USE_CLUSTER,
            get_domain=not USE_CLUSTER,
            chunk=True,
            n_workers=NUM_WORKERS,
        )

        print(f"✅ Data loaded successfully!")
        print(f"  🚂 Train batches: {len(source_train)}")
        print(f"  🔍 Val batches: {len(source_val)}")
        print(f"  🎯 Test batches: {len(target_test)}")

        # Initialize model
        hidden_nodes = getattr(
            config.model.feature_extractor, "hidden_layers", [64, 32, 16, 10]
        )
        classifier_nodes = getattr(
            config.model.classifier, "hidden_layers", [64, 32, 16]
        )

        discriminator_dimensions = (
            getattr(config.training, "num_clusters", 4)
            if USE_CLUSTER
            else len(source_domain)
        )

        model = (
            momlnids(
                input_nodes=INPUT_NODES,
                hidden_nodes=hidden_nodes,
                classifier_nodes=classifier_nodes,
                num_domains=discriminator_dimensions,
                num_class=2,
                single_layer=SINGLE_LAYER,
            )
            .double()
            .to(device)
        )

        model.apply(init_weights)

        print(f"🧠 Model initialized!")
        print(f"  📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  🎯 Discriminator dims: {discriminator_dimensions}")

        # Setup loss weights
        DISCRIMINATOR_LOSS_WEIGHT = (
            torch.tensor(DOMAIN_WEIGHT, dtype=torch.double, device=device)
            if not USE_CLUSTER
            else None
        )
        CLASS_LOSS_WEIGHT = torch.tensor(
            LABEL_WEIGHT, dtype=torch.double, device=device
        )

        # Initialize optimizers
        model_learning_rate = get_model_learning_rate(
            model,
            extractor_weight=getattr(config.training, "extractor_weight", 1.0),
            classifier_weight=getattr(config.training, "classifier_weight", 1.0),
            discriminator_weight=getattr(config.training, "discriminator_weight", 1.0),
        )

        optimizers = [
            get_optimizer(
                model_module,
                LEARNING_RATE * alpha,
                weight_decay=getattr(config.training, "weight_decay", 5e-4),
                amsgrad=getattr(config.training, "amsgrad", True),
            )
            for model_module, alpha in model_learning_rate
        ]

        model_schedulers = [
            get_learning_rate_scheduler(
                optimizer=opt, t_max=getattr(config.training, "t_max", 10)
            )
            for opt in optimizers
        ]

        print(f"⚙️  Optimizers and schedulers initialized!")

        # Training loop
        best_accuracy = 0.0
        test_accuracy = 0.0
        best_epoch = 0

        print(f"\n🚀 Starting improved training...")
        print("=" * 80)

        for epoch in tqdm(range(NUM_EPOCH), desc="Training Progress"):
            logger.info(f"Epoch {epoch + 1}/{NUM_EPOCH}")
            print(f"\n📈 Epoch {epoch + 1}/{NUM_EPOCH}")
            print(f"  ⚡ Learning Rate: {optimizers[0].param_groups[0]['lr']:.6f}")
            print(
                f"  🏆 Current Best: {best_accuracy:.4f} (Test: {test_accuracy:.4f}) at Epoch {best_epoch + 1}"
            )

            source_dataset = source_train.dataset.dataset

            # Pseudo-labeling clustering
            if (
                USE_CLUSTER
                and epoch % getattr(config.training, "clustering_step", 2) == 0
            ):
                try:
                    pseudo_domain_label = pseudolabeling(
                        dataset=source_dataset,
                        model=model,
                        device=device,
                        previous_cluster=source_dataset.cluster_label,
                        log_file=SAVE_PATH / "clustering.log",
                        epoch=epoch,
                        n_clusters=getattr(config.training, "num_clusters", 4),
                        method="MiniK",
                        data_reduction=False,
                        reduced_dimentions=48,
                        batch_size=BATCH_SIZE,
                    )
                    source_dataset.cluster_label = pseudo_domain_label
                except Exception as e:
                    logger.warning(f"Clustering failed: {e}")

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
                label_smooth=getattr(config.training, "label_smooth", 0.1),
                entropy_weight=getattr(config.training, "entropy_weight", 1.0),
                grl_weight=getattr(config.training, "grl_weight", 1.25),
                wandb_enabled=WANDB_ENABLED,
                verbose=False,  # Reduce training spam
            )

            # Evaluation
            if epoch % EVAL_STEP == 0:
                val_accuracy = eval(
                    model=model,
                    eval_data=source_val,
                    device=device,
                    epoch=epoch,
                    filename=SAVE_PATH / "val_performance.log",
                    wandb_enabled=WANDB_ENABLED,
                    verbose=False,  # Reduce eval spam
                )

                target_accuracy = eval(
                    model=model,
                    eval_data=target_test,
                    device=device,
                    epoch=epoch,
                    filename=SAVE_PATH / "target_performance.log",
                    wandb_enabled=WANDB_ENABLED,
                    verbose=False,  # Reduce eval spam
                )

                # Save best model
                if val_accuracy >= best_accuracy:
                    best_accuracy = val_accuracy
                    test_accuracy = target_accuracy
                    best_epoch = epoch
                    torch.save(model.state_dict(), SAVE_PATH / "model_best.pt")

                # Clean summary per epoch
                cluster_info = (
                    f" | Clustering: K={getattr(config.training, 'num_clusters', 4)}"
                    if USE_CLUSTER
                    else ""
                )
                print(
                    f"📈 Epoch {epoch + 1:2d}/{NUM_EPOCH} | Improved | Val: {val_accuracy:.4f} | Test: {target_accuracy:.4f} | Best: {best_accuracy:.4f}{cluster_info}"
                )
                if val_accuracy >= best_accuracy:
                    print(f"   💾 New best model!")

            # Save periodic checkpoints
            if epoch % SAVE_STEP == 0:
                torch.save(model.state_dict(), SAVE_PATH / f"model_{epoch}.pt")

            # Update learning rate schedulers
            for scheduler in model_schedulers:
                scheduler.step()

        print("\n" + "=" * 80)
        print(f"🎉 Improved training completed for {DOMAIN_LIST[TARGET_INDEX]}!")
        print(
            f"🏆 Best validation accuracy: {best_accuracy:.4f} at epoch {best_epoch + 1}"
        )
        print(f"🎯 Final test accuracy: {test_accuracy:.4f}")
        print(f"💾 Best model saved: {SAVE_PATH / 'model_best.pt'}")
        print("=" * 80)

        logger.info(
            f"Training completed successfully. Best accuracy: {best_accuracy:.4f}"
        )

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    finally:
        if WANDB_ENABLED:
            wandb.finish()


if __name__ == "__main__":
    main()

    if config.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed}")


def setup_device(config: Config) -> torch.device:
    """Setup training device."""
    if config.device.get("use_cuda", True) and torch.cuda.is_available():
        device_id = config.device.get("cuda_device", 0)
        device = torch.device(f"cuda:{device_id}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device


def load_and_prepare_data(config: Config) -> Dict[str, Any]:
    """Load and prepare datasets."""
    logger.info("Loading datasets...")

    datasets = {}
    scalers = {}

    for dataset_name in config.data["datasets"]:
        logger.info(f"Loading {dataset_name}")

        # Load dataset (you'll need to implement this based on your data structure)
        dataset_path = Path(config.data["interim_data_path"]) / dataset_name

        # For demonstration, create dummy data
        # Replace this with actual data loading logic
        n_samples = 10000
        n_features = 43  # Common number of features in network datasets

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)

        # Preprocessing
        if config.data["preprocessing"]["normalize"]:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            scalers[dataset_name] = scaler

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=(1 - config.data["train_ratio"]),
            random_state=config.random_seed,
            stratify=y,
        )

        val_ratio = config.data["val_ratio"] / (
            config.data["val_ratio"] + config.data["test_ratio"]
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(1 - val_ratio),
            random_state=config.random_seed,
            stratify=y_temp,
        )

        datasets[dataset_name] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "n_features": n_features,
        }

    logger.info(f"Loaded {len(datasets)} datasets")
    return datasets, scalers


def create_model(config: Config, n_features: int) -> nn.Module:
    """Create and initialize model."""
    logger.info("Creating model...")

    model_config = config.model

    # Create model based on configuration
    model = momlnids(
        input_nodes=n_features,
        hidden_nodes=model_config["feature_extractor"]["hidden_layers"],
        classifier_nodes=model_config["classifier"]["hidden_layers"],
        num_domains=model_config["discriminator"]["num_domains"],
        num_class=model_config["classifier"]["num_classes"],
        single_layer=True,
    )

    logger.info(f"Created model: {model_config['name']}")
    return model


def train_model(
    model: nn.Module,
    datasets: Dict[str, Any],
    config: Config,
    experiment_tracker: ExperimentTracker,
) -> Dict[str, Any]:
    """Train the model with experiment tracking."""
    logger.info("Starting model training...")

    device = setup_device(config)
    model = model.to(device).double()  # Set to double precision to match TrainEval

    # Create dummy data loaders for demonstration
    # In a real implementation, you would use the actual data loading functions
    from torch.utils.data import DataLoader, TensorDataset

    # Prepare combined dataset for multi-domain training
    all_train_data = []
    all_train_labels = []
    domain_labels = []

    for domain_idx, (dataset_name, data) in enumerate(datasets.items()):
        all_train_data.append(data["X_train"])
        all_train_labels.append(data["y_train"])
        domain_labels.extend([domain_idx] * len(data["y_train"]))

    X_train_combined = np.vstack(all_train_data)
    y_train_combined = np.hstack(all_train_labels)
    domain_labels = np.array(domain_labels)

    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_combined, dtype=torch.float64),
        torch.tensor(y_train_combined, dtype=torch.long),
        torch.tensor(domain_labels, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.training["batch_size"], shuffle=True
    )

    # Setup optimizers
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training["learning_rate"]
    )
    optimizers = [optimizer]

    # Training loop with experiment tracking
    training_results = {}
    epochs = config.training["epochs"]

    # Create logs directory
    logs_dir = Path(config.output.get("logs_dir", "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "training.log"

    logger.info(f"Training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # Train for one epoch
        model, optimizers = train(
            model=model,
            train_data=train_loader,
            optimizers=optimizers,
            device=device,
            epoch=epoch,
            num_epoch=epochs,
            filename=str(log_file),
            label_smooth=0.0,  # Add label smoothing parameter
        )

        # Log metrics to wandb
        experiment_tracker.log_metrics(
            {"epoch": epoch, "training_progress": epoch / epochs}
        )

        if epoch % 5 == 0:
            logger.info(f"Completed epoch {epoch}/{epochs}")

    training_results["epochs_completed"] = epochs
    training_results["final_model"] = model

    logger.info("Training completed")
    return training_results


def evaluate_model(
    model: nn.Module,
    datasets: Dict[str, Any],
    config: Config,
    experiment_tracker: ExperimentTracker,
) -> Dict[str, Any]:
    """Evaluate model on all datasets."""
    logger.info("Evaluating model...")

    device = setup_device(config)
    model.eval()

    all_results = {}

    for dataset_name, data in datasets.items():
        logger.info(f"Evaluating on {dataset_name}")

        X_test = torch.tensor(data["X_test"], dtype=torch.float64, device=device)
        y_test = data["y_test"]

        with torch.no_grad():
            outputs_class, outputs_domain = model(X_test)  # Model returns tuple
            y_pred = outputs_class.argmax(dim=1).cpu().numpy()
            y_prob = torch.softmax(outputs_class, dim=1).cpu().numpy()

        # Log performance metrics
        metrics = experiment_tracker.log_model_performance(
            y_test, y_pred, y_prob, prefix=f"{dataset_name}_"
        )

        # Log confusion matrix
        experiment_tracker.log_confusion_matrix(
            y_test, y_pred, class_names=["Normal", "Attack"]
        )

        all_results[dataset_name] = {
            "metrics": metrics,
            "predictions": y_pred,
            "probabilities": y_prob,
            "true_labels": y_test,
        }

    logger.info("Evaluation completed")
    return all_results


def explain_model(
    model: nn.Module,
    datasets: Dict[str, Any],
    config: Config,
    experiment_tracker: ExperimentTracker,
) -> None:
    """Generate model explanations."""
    if not config.explainable_ai.get("enabled", False):
        logger.info("Explainable AI is disabled")
        return

    logger.info("Generating model explanations...")

    # Create feature names (replace with actual feature names)
    feature_names = [
        f"feature_{i}" for i in range(datasets[list(datasets.keys())[0]]["n_features"])
    ]

    # Create a wrapper to handle type conversion
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            # Convert to double precision for the model
            if x.dtype != torch.float64:
                x = x.double()
            outputs_class, outputs_domain = self.model(x)
            return outputs_class  # Return only classification output for explainer

    wrapped_model = ModelWrapper(model)
    explainer = ModelExplainer(wrapped_model, feature_names)

    # Take first dataset for demonstration
    first_dataset = list(datasets.values())[0]
    X_sample = first_dataset["X_test"][:100].astype(
        np.float32
    )  # Sample for explanation, ensure float32

    # Global feature importance
    logger.info("Computing global feature importance...")
    global_importance = explainer.explain_global(X_sample, method="feature_importance")

    # Visualize and log feature importance
    importance_scores = global_importance["importance_scores"]

    # Create output directory
    plots_dir = Path(config.output.get("plots_dir", "plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save feature importance plot
    importance_plot_path = plots_dir / "feature_importance.png"
    visualize_feature_importance(
        importance_scores,
        feature_names,
        title="Global Feature Importance",
        top_k=config.explainable_ai["feature_importance"]["top_k_features"],
        save_path=str(importance_plot_path),
    )

    # Log to wandb
    experiment_tracker.log_feature_importance(
        feature_names, importance_scores, title="Global Feature Importance"
    )

    # Instance-level explanations for a few examples
    logger.info("Computing instance-level explanations...")
    for method in config.explainable_ai.get("methods", ["integrated_gradients"]):
        if method in ["integrated_gradients", "gradient_shap", "feature_ablation"]:
            for i in range(min(5, len(X_sample))):  # Explain first 5 instances
                explanation = explainer.explain_instance(X_sample[i], method=method)

                # Log explanation results
                experiment_tracker.log_metrics(
                    {
                        f"{method}_explanation_{i}_mean_attribution": np.mean(
                            explanation["attributions"]
                        ),
                        f"{method}_explanation_{i}_max_attribution": np.max(
                            np.abs(explanation["attributions"])
                        ),
                    }
                )

    logger.info("Model explanations completed")


def save_model_and_results(
    model: nn.Module,
    training_results: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    config: Config,
    experiment_tracker: ExperimentTracker,
) -> None:
    """Save model and results."""
    logger.info("Saving model and results...")

    # Create output directories
    models_dir = Path(config.output.get("models_dir", "models"))
    results_dir = Path(config.output.get("results_dir", "results"))

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = models_dir / "best_model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    # Log model as wandb artifact
    experiment_tracker.log_model_artifact(
        model,
        "best_model",
        metadata={
            "model_type": config.model["name"],
            "datasets": config.data["datasets"],
            "performance": evaluation_results,
        },
    )

    # Save experiment results
    all_results = {
        "config": experiment_tracker._prepare_config_for_wandb(),
        "training_results": training_results,
        "evaluation_results": evaluation_results,
    }

    experiment_tracker.save_experiment_results(all_results)

    logger.info("Model and results saved")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="NIDS Model Training with Enhanced Features"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (uses default if not specified)",
    )
    parser.add_argument("--experiment-name", type=str, help="Name for the experiment")
    parser.add_argument("--tags", nargs="*", help="Additional tags for the experiment")
    parser.add_argument("--notes", type=str, help="Notes about the experiment")

    args = parser.parse_args()

    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            config = get_config()

        logger.info(f"Loaded configuration: {config.project['name']}")

        # Setup reproducibility
        setup_reproducibility(config)

        # Initialize experiment tracking
        experiment_tracker = ExperimentTracker(config)
        experiment_tracker.init_experiment(
            experiment_name=args.experiment_name, tags=args.tags, notes=args.notes
        )

        # Load and prepare data
        datasets, scalers = load_and_prepare_data(config)

        # Create model
        n_features = datasets[list(datasets.keys())[0]]["n_features"]
        model = create_model(config, n_features)

        # Train model
        training_results = train_model(model, datasets, config, experiment_tracker)

        # Evaluate model
        evaluation_results = evaluate_model(model, datasets, config, experiment_tracker)

        # Generate explanations
        explain_model(model, datasets, config, experiment_tracker)

        # Save results
        save_model_and_results(
            model, training_results, evaluation_results, config, experiment_tracker
        )

        # Print summary
        logger.info("=== Experiment Summary ===")
        for dataset_name, results in evaluation_results.items():
            metrics = results["metrics"]
            accuracy = metrics.get("accuracy", "N/A")
            if isinstance(accuracy, (int, float)):
                logger.info(f"{dataset_name} - Accuracy: {accuracy:.4f}")
            else:
                logger.info(f"{dataset_name} - Accuracy: {accuracy}")

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    finally:
        # Finish experiment tracking
        if "experiment_tracker" in locals():
            experiment_tracker.finish_experiment()
