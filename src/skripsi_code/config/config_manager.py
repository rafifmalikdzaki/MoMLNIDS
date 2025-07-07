"""Configuration management for the NIDS research project."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import logging
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)





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
        
    def load_config(self, config_path: Optional[str] = None) -> DictConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            OmegaConf.DictConfig object with loaded configuration
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
            
        self.config = OmegaConf.create(config_dict)
        return self.config
    
    def save_config(self, config: DictConfig, save_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration object to save
            save_path: Path where to save the configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        OmegaConf.save(config, save_path)
            
        logger.info(f"Configuration saved to: {save_path}")
    
    def merge_configs(self, base_config: DictConfig, override_config: Dict[str, Any]) -> DictConfig:
        """
        Merge configuration with override values.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration values
            
        Returns:
            Merged configuration
        """
        return OmegaConf.merge(base_config, override_config)
    
    def get_config(self) -> DictConfig:
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
            
        self.config = OmegaConf.merge(self.config, updates)
    
    def validate_config(self, config: Optional[DictConfig] = None) -> bool:
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
        current_dir = Path(__file__).parent.parent.parent.parent  # Go up to project root
        return current_dir / "config" / "default_config.yaml"
    
    


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> DictConfig:
    """Get global configuration."""
    return config_manager.get_config()


def load_config(config_path: str) -> DictConfig:
    """Load configuration from file."""
    global config_manager
    config_manager = ConfigManager(config_path)
    return config_manager.load_config()


def update_config(updates: Dict[str, Any]) -> None:
    """Update global configuration."""
    config_manager.update_config(updates)
