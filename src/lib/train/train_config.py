import sys
import os
from dataclasses import dataclass
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.base import BaseConfig

@dataclass
class BaseTrainConfig(BaseConfig):
    """Dataclass for base configuration parameters."""
    net_input_directory: str
    net_output_directory: str
    year: int
    month_range: List[int]
    hour_range: List[int]
    s2_datasets: List[str]
    inca_datasets: List[str]
    resolution: int
    inca_range: int
    inca_cumulative_datasets: List[str]
    inca_cumulative_window: int
    filter_by_SCL: bool
    patch_size: int
    max_epochs: int
    dropout_rate: float
    train_valid_split_shuffle: bool
    validation_split: float