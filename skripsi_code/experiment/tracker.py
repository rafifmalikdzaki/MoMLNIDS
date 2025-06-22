"""Experiment tracking utilities using Weights & Biases (wandb)."""

import os
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path
import json

import wandb
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ..config import Config

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Handles experiment tracking with wandb integration."""
    
    def __init__(self, config: Config):
        """
        Initialize experiment tracker.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.wandb_enabled = config.wandb.get('enabled', False)
        self.run = None
        
    def init_experiment(self, 
                       experiment_name: Optional[str] = None,
                       tags: Optional[list] = None,
                       notes: Optional[str] = None) -> None:
        """
        Initialize experiment tracking.
        
        Args:
            experiment_name: Name of the experiment
            tags: Additional tags for the experiment
            notes: Notes about the experiment
        """
        if not self.wandb_enabled:
            logger.info("Wandb tracking is disabled")
            return
            
        try:
            # Prepare wandb configuration
            wandb_config = {
                'project': self.config.wandb.get('project', 'nids-research'),
                'entity': self.config.wandb.get('entity'),
                'name': experiment_name,
                'tags': (self.config.wandb.get('tags', []) + (tags or [])),
                'notes': notes,
                'config': self._prepare_config_for_wandb(),
            }
            
            # Remove None values
            wandb_config = {k: v for k, v in wandb_config.items() if v is not None}
            
            # Initialize wandb run
            self.run = wandb.init(**wandb_config)
            
            logger.info(f"Initialized wandb experiment: {self.run.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            self.wandb_enabled = False
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number (optional)
        """
        if not self.wandb_enabled or self.run is None:
            return
            
        try:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
                
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_model_performance(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray, 
                            y_prob: Optional[np.ndarray] = None,
                            prefix: str = "") -> Dict[str, float]:
        """
        Log model performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        try:
            # Compute basic metrics
            metrics[f'{prefix}accuracy'] = accuracy_score(y_true, y_pred)
            metrics[f'{prefix}precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics[f'{prefix}recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics[f'{prefix}f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Compute AUC if probabilities are provided
            if y_prob is not None:
                try:
                    if y_prob.shape[1] == 2:  # Binary classification
                        metrics[f'{prefix}auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
                    else:  # Multi-class
                        metrics[f'{prefix}auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                except Exception as e:
                    logger.warning(f"Could not compute AUC: {e}")
            
            # Log metrics
            self.log_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to compute/log performance metrics: {e}")
            return {}
    
    def log_model_artifact(self, 
                          model: torch.nn.Module, 
                          name: str,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log model as wandb artifact.
        
        Args:
            model: PyTorch model to save
            name: Name for the model artifact
            metadata: Additional metadata
        """
        if not self.wandb_enabled or self.run is None:
            return
            
        try:
            # Create artifact
            artifact = wandb.Artifact(
                name=name,
                type="model",
                metadata=metadata or {}
            )
            
            # Save model temporarily
            temp_path = Path(f"temp_{name}.pth")
            torch.save(model.state_dict(), temp_path)
            
            # Add to artifact
            artifact.add_file(str(temp_path))
            
            # Log artifact
            self.run.log_artifact(artifact)
            
            # Clean up
            temp_path.unlink()
            
            logger.info(f"Logged model artifact: {name}")
            
        except Exception as e:
            logger.error(f"Failed to log model artifact: {e}")
    
    def log_confusion_matrix(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           class_names: Optional[list] = None) -> None:
        """
        Log confusion matrix to wandb.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes (optional)
        """
        if not self.wandb_enabled or self.run is None:
            return
            
        try:
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=y_pred,
                    class_names=class_names
                )
            })
            
        except Exception as e:
            logger.error(f"Failed to log confusion matrix: {e}")
    
    def log_feature_importance(self, 
                             feature_names: list, 
                             importance_scores: np.ndarray,
                             title: str = "Feature Importance") -> None:
        """
        Log feature importance plot.
        
        Args:
            feature_names: Names of features
            importance_scores: Importance scores for each feature
            title: Title for the plot
        """
        if not self.wandb_enabled or self.run is None:
            return
            
        try:
            # Create data for plotting
            data = [[name, score] for name, score in zip(feature_names, importance_scores)]
            
            # Log as table
            table = wandb.Table(data=data, columns=["Feature", "Importance"])
            
            wandb.log({
                f"{title.lower().replace(' ', '_')}": wandb.plot.bar(
                    table, "Feature", "Importance", title=title
                )
            })
            
        except Exception as e:
            logger.error(f"Failed to log feature importance: {e}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        if not self.wandb_enabled or self.run is None:
            return
            
        try:
            wandb.config.update(hyperparams)
        except Exception as e:
            logger.error(f"Failed to log hyperparameters: {e}")
    
    def save_experiment_results(self, 
                               results: Dict[str, Any], 
                               filename: str = "experiment_results.json") -> None:
        """
        Save experiment results to file and log as artifact.
        
        Args:
            results: Results dictionary
            filename: Filename for results
        """
        try:
            # Create results directory
            results_dir = Path(self.config.output.get('results_dir', 'results'))
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results locally
            results_path = results_dir / filename
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
            logger.info(f"Saved experiment results to: {results_path}")
            
            # Log as wandb artifact if enabled
            if self.wandb_enabled and self.run is not None:
                artifact = wandb.Artifact("experiment_results", type="results")
                artifact.add_file(str(results_path))
                self.run.log_artifact(artifact)
                
        except Exception as e:
            logger.error(f"Failed to save experiment results: {e}")
    
    def finish_experiment(self) -> None:
        """Finish the current experiment."""
        if self.wandb_enabled and self.run is not None:
            try:
                wandb.finish()
                logger.info("Finished wandb experiment")
            except Exception as e:
                logger.error(f"Failed to finish wandb experiment: {e}")
    
    def _prepare_config_for_wandb(self) -> Dict[str, Any]:
        """Prepare configuration for wandb logging."""
        # Create a simplified config dict for wandb
        config_dict = {
            'model_name': self.config.model.get('name'),
            'batch_size': self.config.training.get('batch_size'),
            'learning_rate': self.config.training.get('learning_rate'),
            'epochs': self.config.training.get('epochs'),
            'datasets': self.config.data.get('datasets'),
            'random_seed': self.config.random_seed,
        }
        
        # Remove None values
        return {k: v for k, v in config_dict.items() if v is not None}


class MetricsLogger:
    """Simple metrics logger for tracking training progress."""
    
    def __init__(self):
        """Initialize metrics logger."""
        self.metrics_history = {}
        
    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Step number (optional)
        """
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            
            if step is not None:
                self.metrics_history[name].append((step, value))
            else:
                self.metrics_history[name].append(value)
    
    def get_history(self, metric_name: str) -> list:
        """Get history for a specific metric."""
        return self.metrics_history.get(metric_name, [])
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric."""
        history = self.get_history(metric_name)
        if history:
            return history[-1] if isinstance(history[-1], (int, float)) else history[-1][1]
        return None
    
    def clear(self) -> None:
        """Clear all metrics history."""
        self.metrics_history.clear()
