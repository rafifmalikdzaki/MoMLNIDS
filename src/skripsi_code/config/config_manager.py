"""Configuration management for the NIDS research project."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import logging
from omegaconf import OmegaConf, DictConfig
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.syntax import Syntax
import json
import tempfile

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

        with open(config_path, "r") as f:
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

    def merge_configs(
        self, base_config: DictConfig, override_config: Dict[str, Any]
    ) -> DictConfig:
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
            required_fields = ["project", "model", "training", "data"]
            for field in required_fields:
                if not hasattr(config, field) or getattr(config, field) is None:
                    raise ValueError(f"Required configuration field missing: {field}")

            # Validate data paths exist
            data_paths = [
                config.data.get("raw_data_path"),
                config.data.get("interim_data_path"),
            ]

            for path in data_paths:
                if path and not Path(path).exists():
                    logger.warning(f"Data path does not exist: {path}")

            # Validate training parameters
            if config.training.get("epochs", 0) <= 0:
                raise ValueError("Training epochs must be positive")

            if config.training.get("batch_size", 0) <= 0:
                raise ValueError("Batch size must be positive")

            logger.info("Configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def _get_default_config_path(self) -> str:
        """Get path to default configuration file."""
        current_dir = Path(
            __file__
        ).parent.parent.parent.parent  # Go up to project root
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


def demo_config_management():
    """Demonstrate configuration management functionality."""
    console = Console()

    console.print(Panel.fit("⚙️ Configuration Management Demo", style="bold blue"))

    # Test configuration loading
    console.print("📁 Testing configuration loading...")

    try:
        # Try to load default config
        config_mgr = ConfigManager()
        config = config_mgr.load_config()
        console.print("✅ Default configuration loaded successfully")

        # Display configuration structure
        console.print("\n📋 Configuration Structure:")
        tree = Tree("🔧 Configuration")

        def add_to_tree(node, data, prefix=""):
            if isinstance(data, DictConfig):
                for key, value in data.items():
                    if isinstance(value, (DictConfig, dict)):
                        branch = node.add(f"📂 {key}")
                        add_to_tree(branch, value, f"{prefix}.{key}")
                    else:
                        node.add(f"📄 {key}: {value}")
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        branch = node.add(f"📂 {key}")
                        add_to_tree(branch, value, f"{prefix}.{key}")
                    else:
                        node.add(f"📄 {key}: {value}")

        add_to_tree(tree, config)
        console.print(tree)

        # Test configuration validation
        console.print("\n🔍 Testing configuration validation...")
        is_valid = config_mgr.validate_config()
        if is_valid:
            console.print("✅ Configuration validation passed")
        else:
            console.print("❌ Configuration validation failed")

    except FileNotFoundError:
        console.print("⚠️ Default configuration not found, creating sample config...")

        # Create sample configuration
        sample_config = {
            "project": {
                "name": "MoMLNIDS",
                "version": "1.0.0",
                "description": "Multi-domain Machine Learning NIDS",
            },
            "model": {
                "type": "MoMLNIDS",
                "feature_extractor": {"layers": [256, 128, 64], "activation": "relu"},
                "classifier": {"layers": [32, 16], "num_classes": 10},
            },
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "device": "cuda",
            },
            "data": {
                "raw_data_path": "/tmp/raw_data",
                "interim_data_path": "/tmp/interim_data",
                "processed_data_path": "/tmp/processed_data",
            },
        }

        # Save sample config to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config, f)
            temp_config_path = f.name

        try:
            config_mgr = ConfigManager(temp_config_path)
            config = config_mgr.load_config()
            console.print("✅ Sample configuration created and loaded")
        finally:
            os.unlink(temp_config_path)

    # Test configuration operations
    console.print("\n🔧 Testing configuration operations...")

    operations_table = Table(title="Configuration Operations")
    operations_table.add_column("Operation", style="cyan")
    operations_table.add_column("Status", style="green")
    operations_table.add_column("Details", style="yellow")

    try:
        # Test merging
        override_config = {"training": {"batch_size": 64, "new_param": "test"}}
        merged_config = config_mgr.merge_configs(config, override_config)
        operations_table.add_row(
            "Merge Config",
            "✅ Success",
            f"Batch size updated to {merged_config.training.batch_size}",
        )

        # Test update
        config_mgr.update_config({"training": {"epochs": 200}})
        updated_config = config_mgr.get_config()
        operations_table.add_row(
            "Update Config",
            "✅ Success",
            f"Epochs updated to {updated_config.training.epochs}",
        )

        # Test save (to temp file)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_save_path = f.name

        config_mgr.save_config(updated_config, temp_save_path)
        operations_table.add_row(
            "Save Config", "✅ Success", f"Saved to {temp_save_path}"
        )

        # Clean up
        os.unlink(temp_save_path)

    except Exception as e:
        operations_table.add_row("Operations", "❌ Error", str(e)[:50])

    console.print(operations_table)
    console.print("\n✨ Configuration management demo completed!")


def display_config_details(config: DictConfig):
    """Display detailed configuration information."""
    console = Console()

    # Convert to dict for display
    config_dict = OmegaConf.to_yaml(config)

    syntax = Syntax(config_dict, "yaml", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="📋 Configuration Details", border_style="green"))


@click.command()
@click.option("--demo", is_flag=True, help="Run configuration management demonstration")
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    help="Path to configuration file to load",
)
@click.option("--validate", is_flag=True, help="Validate configuration file")
@click.option("--show-config", is_flag=True, help="Display current configuration")
@click.option(
    "--create-sample", type=click.Path(), help="Create sample configuration file"
)
def main(demo, config_path, validate, show_config, create_sample):
    """
    Test and demonstrate configuration management functionality.

    This script provides comprehensive testing of configuration loading,
    validation, merging, and saving operations.
    """
    console = Console()

    if demo:
        demo_config_management()
    elif config_path:
        console.print(
            Panel.fit(f"📁 Loading Configuration: {config_path}", style="bold blue")
        )

        try:
            config_mgr = ConfigManager(config_path)
            config = config_mgr.load_config()
            console.print("✅ Configuration loaded successfully")

            if validate:
                console.print("\n🔍 Validating configuration...")
                is_valid = config_mgr.validate_config()
                if is_valid:
                    console.print("✅ Configuration is valid")
                else:
                    console.print("❌ Configuration validation failed")

            if show_config:
                display_config_details(config)

        except Exception as e:
            console.print(f"❌ Error loading configuration: {e}")
    elif create_sample:
        console.print(
            Panel.fit(
                f"📝 Creating Sample Configuration: {create_sample}", style="bold green"
            )
        )

        sample_config = {
            "project": {
                "name": "MoMLNIDS-Sample",
                "version": "1.0.0",
                "description": "Sample configuration for Multi-domain ML NIDS",
            },
            "model": {
                "type": "MoMLNIDS",
                "feature_extractor": {
                    "layers": [512, 256, 128],
                    "activation": "relu",
                    "dropout": 0.2,
                },
                "classifier": {
                    "layers": [64, 32],
                    "num_classes": 15,
                    "activation": "softmax",
                },
                "discriminator": {"layers": [128, 64], "activation": "relu"},
            },
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
                "device": "cuda",
                "num_workers": 4,
            },
            "data": {
                "raw_data_path": "./data/raw",
                "interim_data_path": "./data/interim",
                "processed_data_path": "./data/processed",
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1,
            },
            "clustering": {
                "method": "kmeans",
                "n_clusters": 5,
                "pca_dim": 48,
                "data_reduction": True,
            },
            "logging": {
                "level": "INFO",
                "log_dir": "./logs",
                "wandb_project": "MoMLNIDS",
            },
        }

        try:
            config_mgr = ConfigManager()
            config_mgr.save_config(OmegaConf.create(sample_config), create_sample)
            console.print(f"✅ Sample configuration created at: {create_sample}")
        except Exception as e:
            console.print(f"❌ Error creating sample configuration: {e}")
    else:
        console.print(
            "Use --demo to run demonstration, --config-path to load specific config,"
        )
        console.print("or --create-sample to create a sample configuration file")
        console.print("Use --help for more options")


