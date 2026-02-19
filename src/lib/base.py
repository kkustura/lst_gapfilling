import yaml
import logging
import os
import argparse
from dataclasses import dataclass
from logger import setup_logger

@dataclass
class BaseConfig:
    """Dataclass for base configuration parameters."""
    @classmethod
    def from_yaml(cls, yaml_path):
        """Create a BaseDataConfig instance from a YAML file."""
        with open(yaml_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        # filter out any keys that are not defined in the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)
    
    def update_from_dict(self, config_dict):
        """
        Update configuration parameters from a dictionary.
        Dict key must match defined dataclass field names.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)
        return self
            
    def update_from_args(self, args: argparse.Namespace):
        """Update configuration parameters from keyword arguments."""
        for key, value in vars(args).items():
            setattr(self, key, value)
        return self
            

class BaseSetup:
    """Base class for loading config and preparing directories."""
    def __init__(self, cfg: BaseConfig, level = logging.INFO, pretty: bool = False):
        self.config = cfg  # config object
        self.logger = setup_logger(level=level, pretty=pretty)
        
        # unpack all config parameters as class attributes
        for field in cfg.__dataclass_fields__:
            setattr(self, field, getattr(cfg, field))
            
    def ensure_dirs_exist(self, *dirs):
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        self.logger.debug(f"Directories prepared:")
        for d in dirs:
            self.logger.debug(f"    {d}")
        return self