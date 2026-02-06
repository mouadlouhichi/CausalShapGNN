import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    default_path = os.path.join(os.path.dirname(__file__), 'default_config.yaml')
    return load_config(default_path)