import sys
import os
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.utils import list_filepaths
from lib.base import BaseSetup
from lib.train.train_config import BaseTrainConfig
from lib.data.image import ImageProcessor
    
class TimestampGenerator(BaseSetup):
    """
    Class for generating timestamps for train/test images.
    Time window for image search is defined in the config file (year, month_range, hour_range).    
    """
    def __init__(self, cfg: BaseTrainConfig):
        super().__init__(cfg)
        # initialize empty dict to be filled with tile-specific timestamps (key=tile, value=list of timestamps)
        self.timestamp_dict = {}
        # initialize empty supplementary dict for model eval during training
        self.timestamp_dict_supplementary = {}
        # directory for saving timestamp .txt files
        self.net_input_directory_txt = os.path.join(self.net_input_directory, 'timestamp_lists')

    def prepare_directories(self):
        self.ensure_dirs_exist(self.net_input_directory, self.net_input_directory_txt) 
        return self

    def generate_timestamps_from_dates(self, start_date, end_date, frequency='3-day'):
        """
        Generate a list of timestamps for a given date range.
        Note: data source independent method. \n
        Input:
        - start_date: start date string 'YYYY-MM-DD'
        - end_date: end date string 'YYYY-MM-DD'
        - frequency: frequency of timestamps ('daily', '3-day', 'hourly') \n
        Output:
        - list of timestamps (strings in 'YYYYMMDDTHHMMSS' format)
        """        
        def _time_increase(frequency):
            days = 0
            hours = 0
            if frequency == 'daily':
                days = 1
            elif frequency == '3-day':
                days = 3
            elif frequency == 'hourly':
                hours = 1
            else:
                raise ValueError(f"Invalid frequency: {frequency}. Supported frequencies are 'daily', '3-day', and 'hourly'.")
            return days, hours
        
        def _string_to_datetime(date_str, hour=0):
            try:
                return datetime.strptime(date_str, "%Y-%m-%d").replace(hour=hour, minute=0, second=0)
            except ValueError as e:
                self.logger.error(f"Invalid date format: {date_str}")
                raise ValueError(f"Invalid date format: {date_str}. Expected format: 'YYYY-MM-DD'.")
        
        ref_year = int(start_date[:4])
        if ref_year != int(end_date[:4]):
            self.logger.error(f"Year mismatch in date range: {start_date} and {end_date}")
            raise ValueError(f"Year mismatch in date range: {start_date} and {end_date}. Range should be within the same year.")
        
        date_range = []
        for d in [start_date, end_date]:
            if frequency in ['daily', '3-day']:
                date_range.append(_string_to_datetime(d, hour=15))
            elif frequency == 'hourly':
                date_range.append(_string_to_datetime(d, hour=0))
            else:
                raise ValueError(f"Invalid frequency: {frequency}. Supported frequencies are 'daily', '3-day', and 'hourly'.")
        
        timestamps = []
        current_date = date_range[0]
        increase_days, increase_hours = _time_increase(frequency)
        while current_date <= date_range[1]:
            timestamps.append(current_date.strftime("%Y%m%dT%H%M%S"))
            current_date += timedelta(days=increase_days, hours=increase_hours) 
        return timestamps

    def update_timestamps(self, key, values, update_type='main'):
        """
        Update timestamps dictionary with provided data. \n
        Input parameters:
        - key: S2 tile (string)
        - values: list of timestamps (strings in 'YYYYMMDDTHHMMSS' format)
                  values = [timestamp1, timestamp2, ...]
        - update_type: 'main' or 'supplementary' to update main or supplementary (eval) dict, respectively. \n
        Output:
        - None \n
        Result of running the method:
        - Updates self.timestamp_dict or self.timestamp_dict_supplementary depending on the 'update_type' parameter.
        """
        if update_type not in ['main', 'supplementary']:
            raise ValueError(f"Invalid update_type: {update_type}. Supported types are 'main' and 'supplementary'.")
        if key not in self.timestamp_dict.keys():
            self.timestamp_dict[key] = []
        if key not in self.timestamp_dict_supplementary.keys():
            self.timestamp_dict_supplementary[key] = []
        if update_type == 'main':
            self.timestamp_dict[key].extend(values)
        else:
            self.timestamp_dict_supplementary[key].extend(values)
    
    @staticmethod
    def flatten_to_tuples(data_dict):
        """
        Convert a dictionary {tile: timestamps} into a flat list of pairs. \n
        Input:
        - data_dict: dictionary {tile1: [timestamp1, timestamp2, ...], ...} \n
        Output:
        - list of tuples: [(tile1, timestamp1), (tile1, timestamp2), ...]
        """        
        out_list = []
        for tile in data_dict.keys():
            out_list.append(
                [(tile, timestamp) for timestamp in data_dict[tile]]
            )
        out_list = [i for element in out_list for i in element]  # flatten the list of lists
        return out_list
    
    @staticmethod
    def list_to_txt(data, output_path):
            """Save data to a text file. \n
            Input:
            - data: list of elements (each element will be saved in a separate line)
            - output_path: full path to the output file\n
            Output:
            - None \n
            Result of running the function:
            - Saves the data to a text file in the specified directory with the specified name.
            """
            with open(output_path, 'w') as file:
                for item in data:
                    file.write(f'{item[0]}, {item[1]}\n')
    
    def load_tuples_from_txt(self, load_type='main', reduce_samples=None):
        """
        Load list of tuples [(str, str), ...] from a text file. \n
        Input:
        - load_type: 'main' or 'supplementary'. 
            Main is list for training. Supplementary is list for eval during training or for separate 
            model validation on arbitrary timestamps.
        - reduce samples (int): if provided, reduce number of loaded samples to the specified number. \n
        Output:
        - data from txt as list (1 element = 1 row)
        """        
        def _read_lines_as_tuples(path):
            """Read lines with two elements separated by ',' from a text file."""
            data = []
            with open(path, 'r') as file:
                for line in file:
                    values = line.strip().split(', ')
                    data.append((values[0], values[1]))
            return data
        
        if load_type not in ['main', 'supplementary']:
            raise ValueError(f"Invalid load_type: {load_type}. Supported load_types are 'main' and 'supplementary'.")
        pattern_in = 'train' if load_type == 'main' else 'eval'
        tmp = list_filepaths(self.net_input_directory_txt,
            patterns_in=[pattern_in, '.txt'], patterns_out=['.aux', '.dat'])   
        if len(tmp) == 0:
            raise FileNotFoundError(f"No files found for load_type '{load_type}' in {self.net_input_directory_txt}. Please check the directory.")
        if len(tmp) > 1:
            # Sort by modification time (newest first)
            tmp.sort(key=lambda f: Path(f).stat().st_mtime, reverse=True)
            self.logger.warning(f"Multiple files found for load_type '{load_type}'. Using the most recent one: {os.path.basename(tmp[0])}")
        file_path = tmp[0]
        self.logger.info(f"{load_type.capitalize()} dataset: {file_path}")
        data = _read_lines_as_tuples(file_path)
        if reduce_samples and len(data) > reduce_samples:
            data = random.sample(data, reduce_samples)
        setattr(self, f'{load_type}_data', data)
        return data            
    
    # ======== ECOSTRESS-specific methods ========
    
    def find_ecostress_images(self, ecostress_processor, search_patterns=['_LSTE']):
        """
        Search ECOSTRESS images and generate list of timestamps. \n
        Input:
        - ecostress_processor: instance of EcostressManager. Provides search directory (masked_dir).
        - search_patterns: required patterns in the filenames to identify images.\n
        Output:
        - None \n
        Result of running the method:
        - list of matching timestamps is stored in self.eco_timestamps.
        """
        search_directory = ecostress_processor.download_dir
        self.logger.info(f"Searching for ECOSTRESS images in {search_directory}...")
        image_list = list_filepaths(search_directory, search_patterns, ['.aux'])
        self.logger.info(f"    ECOSTRESS images found: {len(image_list)}")
        
        timestamps = [ImageProcessor.get_timestamp_from_filename(i, data_source='ECOSTRESS') for i in image_list]
        # filter by year
        if self.year:
            self.logger.info(f"    Filtering images by year: {self.year}")
            timestamps = [i for i in timestamps if i.split('T')[0][0:4] == str(self.year)]
            self.logger.info(f"    Images in year {self.year}: {len(timestamps)}")
        # filter by month range
        timestamps = [
            t for t in timestamps 
            if self.month_range[0] <= int(t.split('T')[0][4:6]) <= self.month_range[1]
            ]
        self.logger.info(f"    Images in month range {self.month_range}: {len(timestamps)}")
        # filter by hour range
        timestamps = [
            t for t in timestamps 
            if self.hour_range[0] <= int(t.split('T')[1][:2]) <= self.hour_range[1]
            ]
        self.logger.info(f"    Images in hour range {self.hour_range}: {len(timestamps)}")
        self.eco_timestamps = timestamps
        
    def generate_timestamps_from_ecostress(self, ecostress_processor, tile):
        """
        Generate a list of tuples (timestamp, filepath, pct_nodata) for a given tile. \n
        Input:
        - ecostress_processor: instance of EcostressManager. Provides tile directory (reproj_dir).
        - tile: S2 tile\n
        Output:
        - list of tuples (timestamp, filepath, pct_nodata)
        """
        tile_dir = os.path.join(ecostress_processor.reproj_dir, tile)
        # self.logger.info(f"Generating timestamps for tile {tile}...")
        if getattr(self, 'eco_timestamps', None) is None:
            raise ValueError("ECOSTRESS timestamps are not defined. Please run 'find_ecostress_images' first.")
        if not os.path.exists(tile_dir):
            self.logger.warning(f"    Tile {tile} will be skipped: directory {tile_dir} does not exist. ")
            return []
        image_list = list_filepaths(
            tile_dir, 
            patterns_in=self.eco_timestamps, 
            patterns_out=['.aux'], 
            include_all_patterns=False, # non-inclusive search (match any pattern)
        )
        timestamps = [ImageProcessor.get_timestamp_from_filename(i, data_source='ECOSTRESS') for i in image_list]
        
        # check for duplicates
        if len(timestamps) != len(np.unique(timestamps)):
            self.logger.warning(f"    Duplicate timestamps found in tile {tile}. Please check the data.")
            
        # generate the final list of (tile, timestamp, filepath, pct_nodata) tuples 
        data = []
        for t, path in zip(timestamps, image_list):
            pct_nodata = ImageProcessor(path).compute_percent_nodata()
            data.append((t, path, pct_nodata))
        return data

    def train_test_split_ecostress(self, timestamps, 
        num_train=None, 
        num_test=None,
        seed=None
        ):
        """
        Split the timestamps into training and test sets. \n
        Input:
        - timestamps: list of Ecostress tuples (timestamp, filepath, pct_nodata)
        - num_train: number of timestamps for training. If None, all remaining images after test selection are used for training.
        - num_test: number of timestamps for testing. If None, no test set is created.
        - seed: random seed for reproducibility (default: None) \n
        Output:
        - tuple (train, test): lists of timestamp tuples for training/testing        
        """
        self.logger.debug("    Splitting data into training and test sets...")
        # separate the data by year
        data_by_year = {}
        for timestamp, path, pct in timestamps:
            year = timestamp[:4]
            data_by_year.setdefault(year, []).append((timestamp, pct))
        self.logger.debug(f"    Years appearing in the dataset: {list(data_by_year.keys())}")
        train = []
        test = []
        for year, entries in data_by_year.items():
            if len(entries)<=1:
                continue
            # select num_test images for test if specified
            if num_test:
                pct_nodata_threshold = 40  # filter good samples by % of no data
                good_samples = [e for e in entries if e[1] < pct_nodata_threshold]
                if len(good_samples) <= num_test:
                    best = min(entries, key=lambda x: x[1])
                    test.append(best[0])
                    self.logger.warning(f"    Not enough images with pct_nodata<{pct_nodata_threshold} for year {year}. Selecting only 1 best one for testing instead of requested {self.test_max_count}: {best}")
                else:
                    if seed is not None:
                        random.seed(seed)
                    selected = random.sample(good_samples, num_test)
                    test.extend([i[0] for i in selected])  # add selected timestamps to test
            # select images for training
            selected = [i for i in entries if i[0] not in test]
            if num_train:
                selected = sorted(selected, key=lambda x: x[1])[:min(num_train, len(selected))]  # sort by pct
            train.extend([i[0] for i in selected])  # add selected timestamps to training
        return train, test