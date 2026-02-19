import sys
import os
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.base import BaseConfig

@dataclass
class BaseDataConfig(BaseConfig):
    """Dataclass for base configuration parameters."""
    data_directory: str
    ecostress_nodata: int
    s2_nodata: int
    pct_valid_threshold: int
    cloud_cover_max: int
    inca_nodata: int
    inca_datasets: list
    bounding_box: list
    s2_datasets: list