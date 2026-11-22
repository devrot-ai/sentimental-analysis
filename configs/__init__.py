"""Configuration loading utilities for ViMACSA / FCMF framework."""

import os
import yaml
import argparse
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


def load_yaml_config(config_path=None):
    """Load configuration from a YAML file.
    
    Falls back to configs/default.yaml if no path is provided.
    Returns a plain dictionary.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def merge_cli_args(config, args):
    """Merge CLI arguments into config dict. CLI values take precedence.
    
    Only overrides keys that are explicitly provided (not None) in args.
    """
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    
    for key, value in args.items():
        if value is not None:
            config[key] = value
    
    return config


def resolve_device(device_str="auto"):
    """Resolve device string to an actual torch device string."""
    import torch
    
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def get_config(config_path=None, cli_args=None):
    """Main entry point: load YAML defaults, merge CLI overrides, resolve device.
    
    Args:
        config_path: Optional path to a YAML config file.
        cli_args: Optional argparse.Namespace or dict with CLI overrides.
    
    Returns:
        dict with all configuration values.
    """
    config = load_yaml_config(config_path)
    
    if cli_args is not None:
        config = merge_cli_args(config, cli_args)
    
    # Resolve device
    config["device"] = resolve_device(config.get("device", "auto"))
    
    # Ensure output directories exist
    for dir_key in ["output_dir", "checkpoint_dir"]:
        if dir_key in config:
            os.makedirs(config[dir_key], exist_ok=True)
    
    return config
