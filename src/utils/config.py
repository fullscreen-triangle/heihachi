import os
import yaml
from typing import Dict, Any


class ConfigManager:
    def __init__(self, config_path: str = "../configs/default.yaml"):
        """Initialize the config manager.
        
        Args:
            config_path: Path to the config file (relative or absolute)
        """
        self.config_path = config_path
        self.config = self._load_config()



    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file.
        
        Returns:
            Dict: Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        # Load custom config if specified
        custom_config_path = os.environ.get('CUSTOM_CONFIG')
        if custom_config_path and os.path.exists(custom_config_path):
            with open(custom_config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                self._deep_update(config, custom_config)

        return config

    def _deep_update(self, d: dict, u: dict) -> dict:
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def get(self, *keys):
        """Retrieve nested config values using dot notation."""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value

