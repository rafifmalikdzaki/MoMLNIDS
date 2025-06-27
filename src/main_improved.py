#!/usr/bin/env python3
"""
Improved main training script for NIDS research project.

This script demonstrates the integration of:
- Configuration management
- Experiment tracking with wandb
- Enhanced logging
- Explainable AI capabilities
- Better project organization
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import project modules
from skripsi_code.config import Config, load_config, get_config
from skripsi_code.experiment import ExperimentTracker, MetricsLogger
from skripsi_code.explainability import ModelExplainer, visualize_feature_importance
from skripsi_code.model.MoMLNIDS import MoMLDNIDS
from skripsi_code.utils.dataloader import random_split_dataloader
from skripsi_code.utils.domain_dataset import MultiChunkDataset, MultiChunkParquet
from skripsi_code.TrainEval.TrainEval import train, eval

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_reproducibility(config: Config) -> None:
    """Setup reproducible training environment."""
    seed = config.random_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if config.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set random seed to {seed}")


def setup_device(config: Config) -> torch.device:
    """Setup training device."""
    if config.device.get('use_cuda', True) and torch.cuda.is_available():
        device_id = config.device.get('cuda_device', 0)
        device = torch.device(f'cuda:{device_id}')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device


def load_and_prepare_data(config: Config) -> Dict[str, Any]:
    """Load and prepare datasets."""
    logger.info("Loading datasets...")
    
    datasets = {}
    scalers = {}
    
    for dataset_name in config.data['datasets']:
        logger.info(f"Loading {dataset_name}")
        
        # Load dataset (you'll need to implement this based on your data structure)
        dataset_path = Path(config.data['interim_data_path']) / dataset_name
        
        # For demonstration, create dummy data
        # Replace this with actual data loading logic
        n_samples = 10000
        n_features = 43  # Common number of features in network datasets
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # Preprocessing
        if config.data['preprocessing']['normalize']:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            scalers[dataset_name] = scaler
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=(1 - config.data['train_ratio']),
            random_state=config.random_seed,
            stratify=y
        )
        
        val_ratio = config.data['val_ratio'] / (config.data['val_ratio'] + config.data['test_ratio'])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio),
            random_state=config.random_seed,
            stratify=y_temp
        )
        
        datasets[dataset_name] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'n_features': n_features
        }
    
    logger.info(f"Loaded {len(datasets)} datasets")
    return datasets, scalers


def create_model(config: Config, n_features: int) -> nn.Module:
    """Create and initialize model."""
    logger.info("Creating model...")
    
    model_config = config.model
    
    # Create model based on configuration
    model = MoMLDNIDS(
        input_nodes=n_features,
        hidden_nodes=model_config['feature_extractor']['hidden_layers'],
        classifier_nodes=model_config['classifier']['hidden_layers'],
        num_domains=model_config['discriminator']['num_domains'],
        num_class=model_config['classifier']['num_classes'],
        single_layer=True
    )
    
    logger.info(f"Created model: {model_config['name']}")
    return model


def train_model(model: nn.Module, 
               datasets: Dict[str, Any], 
               config: Config,
               experiment_tracker: ExperimentTracker) -> Dict[str, Any]:
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
        all_train_data.append(data['X_train'])
        all_train_labels.append(data['y_train'])
        domain_labels.extend([domain_idx] * len(data['y_train']))
    
    X_train_combined = np.vstack(all_train_data)
    y_train_combined = np.hstack(all_train_labels)
    domain_labels = np.array(domain_labels)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_combined, dtype=torch.float64),
        torch.tensor(y_train_combined, dtype=torch.long),
        torch.tensor(domain_labels, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=config.training['batch_size'], shuffle=True)
    
    # Setup optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training['learning_rate'])
    optimizers = [optimizer]
    
    # Training loop with experiment tracking
    training_results = {}
    epochs = config.training['epochs']
    
    # Create logs directory
    logs_dir = Path(config.output.get('logs_dir', 'logs'))
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / 'training.log'
    
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
            label_smooth=0.0  # Add label smoothing parameter
        )
        
        # Log metrics to wandb
        experiment_tracker.log_metrics({
            'epoch': epoch,
            'training_progress': epoch / epochs
        })
        
        if epoch % 5 == 0:
            logger.info(f"Completed epoch {epoch}/{epochs}")
    
    training_results['epochs_completed'] = epochs
    training_results['final_model'] = model
    
    logger.info("Training completed")
    return training_results


def evaluate_model(model: nn.Module,
                  datasets: Dict[str, Any],
                  config: Config,
                  experiment_tracker: ExperimentTracker) -> Dict[str, Any]:
    """Evaluate model on all datasets."""
    logger.info("Evaluating model...")
    
    device = setup_device(config)
    model.eval()
    
    all_results = {}
    
    for dataset_name, data in datasets.items():
        logger.info(f"Evaluating on {dataset_name}")
        
        X_test = torch.tensor(data['X_test'], dtype=torch.float64, device=device)
        y_test = data['y_test']
        
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
            y_test, y_pred, class_names=['Normal', 'Attack']
        )
        
        all_results[dataset_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_prob,
            'true_labels': y_test
        }
    
    logger.info("Evaluation completed")
    return all_results


def explain_model(model: nn.Module,
                 datasets: Dict[str, Any],
                 config: Config,
                 experiment_tracker: ExperimentTracker) -> None:
    """Generate model explanations."""
    if not config.explainable_ai.get('enabled', False):
        logger.info("Explainable AI is disabled")
        return
    
    logger.info("Generating model explanations...")
    
    # Create feature names (replace with actual feature names)
    feature_names = [f"feature_{i}" for i in range(datasets[list(datasets.keys())[0]]['n_features'])]
    
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
    X_sample = first_dataset['X_test'][:100].astype(np.float32)  # Sample for explanation, ensure float32
    
    # Global feature importance
    logger.info("Computing global feature importance...")
    global_importance = explainer.explain_global(X_sample, method="feature_importance")
    
    # Visualize and log feature importance
    importance_scores = global_importance['importance_scores']
    
    # Create output directory
    plots_dir = Path(config.output.get('plots_dir', 'plots'))
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save feature importance plot
    importance_plot_path = plots_dir / "feature_importance.png"
    visualize_feature_importance(
        importance_scores,
        feature_names,
        title="Global Feature Importance",
        top_k=config.explainable_ai['feature_importance']['top_k_features'],
        save_path=str(importance_plot_path)
    )
    
    # Log to wandb
    experiment_tracker.log_feature_importance(
        feature_names,
        importance_scores,
        title="Global Feature Importance"
    )
    
    # Instance-level explanations for a few examples
    logger.info("Computing instance-level explanations...")
    for method in config.explainable_ai.get('methods', ['integrated_gradients']):
        if method in ['integrated_gradients', 'gradient_shap', 'feature_ablation']:
            for i in range(min(5, len(X_sample))):  # Explain first 5 instances
                explanation = explainer.explain_instance(X_sample[i], method=method)
                
                # Log explanation results
                experiment_tracker.log_metrics({
                    f"{method}_explanation_{i}_mean_attribution": np.mean(explanation['attributions']),
                    f"{method}_explanation_{i}_max_attribution": np.max(np.abs(explanation['attributions']))
                })
    
    logger.info("Model explanations completed")


def save_model_and_results(model: nn.Module,
                          training_results: Dict[str, Any],
                          evaluation_results: Dict[str, Any],
                          config: Config,
                          experiment_tracker: ExperimentTracker) -> None:
    """Save model and results."""
    logger.info("Saving model and results...")
    
    # Create output directories
    models_dir = Path(config.output.get('models_dir', 'models'))
    results_dir = Path(config.output.get('results_dir', 'results'))
    
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
            'model_type': config.model['name'],
            'datasets': config.data['datasets'],
            'performance': evaluation_results
        }
    )
    
    # Save experiment results
    all_results = {
        'config': experiment_tracker._prepare_config_for_wandb(),
        'training_results': training_results,
        'evaluation_results': evaluation_results
    }
    
    experiment_tracker.save_experiment_results(all_results)
    
    logger.info("Model and results saved")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="NIDS Model Training with Enhanced Features")
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to configuration file (uses default if not specified)'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Name for the experiment'
    )
    parser.add_argument(
        '--tags',
        nargs='*',
        help='Additional tags for the experiment'
    )
    parser.add_argument(
        '--notes',
        type=str,
        help='Notes about the experiment'
    )
    
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
            experiment_name=args.experiment_name,
            tags=args.tags,
            notes=args.notes
        )
        
        # Load and prepare data
        datasets, scalers = load_and_prepare_data(config)
        
        # Create model
        n_features = datasets[list(datasets.keys())[0]]['n_features']
        model = create_model(config, n_features)
        
        # Train model
        training_results = train_model(model, datasets, config, experiment_tracker)
        
        # Evaluate model
        evaluation_results = evaluate_model(model, datasets, config, experiment_tracker)
        
        # Generate explanations
        explain_model(model, datasets, config, experiment_tracker)
        
        # Save results
        save_model_and_results(model, training_results, evaluation_results, config, experiment_tracker)
        
        # Print summary
        logger.info("=== Experiment Summary ===")
        for dataset_name, results in evaluation_results.items():
            metrics = results['metrics']
            accuracy = metrics.get('accuracy', 'N/A')
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
        if 'experiment_tracker' in locals():
            experiment_tracker.finish_experiment()


if __name__ == "__main__":
    main()
