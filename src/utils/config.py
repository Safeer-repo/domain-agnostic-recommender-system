import os
import yaml
from typing import Dict, Any

class Config:
    """Configuration class for the recommender system."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize with configuration dictionary."""
        for key, value in config_dict.items():
            setattr(self, key, value)

def load_config(config_path: str = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file, defaults to configs/default.yaml
        
    Returns:
        Config object
    """
    if config_path is None:
        env = os.environ.get("RECOMMENDER_ENV", "development")
        config_path = f"configs/{env}.yaml"
        
        # Fall back to default if environment-specific config doesn't exist
        if not os.path.exists(config_path):
            config_path = "configs/default.yaml"
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)
