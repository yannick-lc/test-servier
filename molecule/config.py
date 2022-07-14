import os
import json
import logging
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO)

PATH_TO_DEFAULT_CONFIG_FILE = "config/config.json"

def get_project_root_directory() -> Path:
    """Return project root directory"""
    return Path(__file__).parent.parent

def get_config(path_to_config_file: str=PATH_TO_DEFAULT_CONFIG_FILE) -> Dict[str, str]:
    """Return config information (e.g. default save path etc) in the form of a dictionary"""
    if not os.path.isabs(path_to_config_file):
       root_dir = get_project_root_directory()
       path_to_config_file = os.path.join(root_dir, path_to_config_file)
    with open(path_to_config_file) as f:
        config = json.load(f)
    return config

configuration = get_config()