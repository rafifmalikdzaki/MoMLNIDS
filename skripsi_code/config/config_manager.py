"""Configuration management for the NIDS research project."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration dataclass with nested structures for different components."""
    
    # Project settings
    project: Dict[str, Any] = field(default_factory=dict)
    
    # Experiment tracking
    wandb: Dict[str, Any] = field(default_factory=dict)
    
    # Model configuration
    model: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    training: Dict[str, Any] = field(default_factory=dict)
    
    # Data configuration
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation configuration
    evaluation: Dict[str, Any] = field(default_factory=dict)
    
    # Explainable AI configuration
    explainable_ai: Dict[str, Any] = field(default_factory=dict)
    
    # Logging configuration
    logging: Dict[str, Any] = field(default_factory=dict)
    
    # Output paths
    output: Dict[str, Any] = field(default_factory=dict)
    
    # Reproducibility settings
    random_seed: int = 42
    deterministic: bool = True
    
    # Hardware configuration
    device: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Manages configuration loading, merging, and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path
        self.config = None
        
    def load_config(self, config_path: Optional[str] = None) -> Config:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Config object with loaded configuration
        """
        if config_path is None:
            config_path = self.config_path
            
        if config_path is None:
            # Use default config
            default_config_path = self._get_default_config_path()
            config_path = default_config_path
            
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        logger.info(f"Loading configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        self.config = Config(**config_dict)
        return self.config
    
    def save_config(self, config: Config, save_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration object to save
            save_path: Path where to save the configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config object to dictionary
        config_dict = self._config_to_dict(config)
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
        logger.info(f"Configuration saved to: {save_path}")
    
    def merge_configs(self, base_config: Config, override_config: Dict[str, Any]) -> Config:
        """
        Merge configuration with override values.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration values
            
        Returns:
            Merged configuration
        """
        base_dict = self._config_to_dict(base_config)
        merged_dict = self._deep_merge(base_dict, override_config)
        
        return Config(**merged_dict)
    
    def get_config(self) -> Config:
        """Get current configuration."""
        if self.config is None:
            self.load_config()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update current configuration with new values.
        
        Args:
            updates: Dictionary with configuration updates
        """
        if self.config is None:
            self.load_config()
            
        self.config = self.merge_configs(self.config, updates)
    
    def validate_config(self, config: Optional[Config] = None) -> bool:
        """
        Validate configuration values.
        
        Args:
            config: Configuration to validate. If None, uses current config.
            
        Returns:
            True if configuration is valid
        """
        if config is None:
            config = self.get_config()
            
        try:
            # Validate required fields
            required_fields = ['project', 'model', 'training', 'data']
            for field in required_fields:
                if not hasattr(config, field) or getattr(config, field) is None:
                    raise ValueError(f"Required configuration field missing: {field}")
            
            # Validate data paths exist
            data_paths = [
                config.data.get('raw_data_path'),
                config.data.get('interim_data_path'),
            ]
            
            for path in data_paths:
                if path and not Path(path).exists():
                    logger.warning(f"Data path does not exist: {path}")
            
            # Validate training parameters
            if config.training.get('epochs', 0) <= 0:
                raise ValueError("Training epochs must be positive")
                
            if config.training.get('batch_size', 0) <= 0:
                raise ValueError("Batch size must be positive")
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _get_default_config_path(self) -> str:
        """Get path to default configuration file."""
        current_dir = Path(__file__).parent.parent.parent  # Go up to project root
        return current_dir / "config" / "default_config.yaml"
    
    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """Convert Config object to dictionary."""
        return {
            'project': config.project,
            'wandb': config.wandb,
            'model': config.model,
            'training': config.training,
            'data': config.data,
            'evaluation': config.evaluation,
            'explainable_ai': config.explainable_ai,
            'logging': config.logging,
            'output': config.output,
            'random_seed': config.random_seed,
            'deterministic': config.deterministic,
            'device': config.device,
        }
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> Config:
    """Get global configuration."""
    return config_manager.get_config()


def load_config(config_path: str) -> Config:
    """Load configuration from file."""
    global config_manager
    config_manager = ConfigManager(config_path)
    return config_manager.load_config()


def update_config(updates: Dict[str, Any]) -> None:
    """Update global configuration."""
    config_manager.update_config(updates)
