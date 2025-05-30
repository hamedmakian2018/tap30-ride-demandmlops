"""
Provides functionality for reading and parsing YAML configuration files.

Import `read_config` from this module to read YAML configuration files that
follow the project's configuration structure.
"""

from pathlib import Path

import yaml

from src.logger import get_logger

logger = get_logger(__name__)


def read_config(config_path):
    """
    Returns
    -------
    dict
        Dictionary containing the parsed YAML configuration.

    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Config file not found at {config_path}")
        raise FileNotFoundError(f"Config file not found at {config_path}")

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_file}")
        raise e


if __name__ == "__main__":
    config = read_config("config/config.yaml")
    print(config)
