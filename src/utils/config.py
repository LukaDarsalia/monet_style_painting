"""
YAML configuration utilities with inheritance support via _base_ directive.
"""

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dicts, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML config with inheritance support.
    
    If config contains '_base_' key, loads base config first and merges.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Merged configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    if '_base_' in config:
        base_path = config_path.parent / config['_base_']
        base_config = load_config(base_path)
        config = deep_merge(base_config, {k: v for k, v in config.items() if k != '_base_'})
    
    return config
