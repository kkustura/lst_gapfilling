import sys
import os
import subprocess
import requests
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from progress.bar import ChargingBar
from typing import Optional
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.utils import list_filepaths, find_nearest_index
from lib.base import BaseSetup
from lib.data.data_config import BaseDataConfig
from lib.data.image import ImageProcessor
from lib.data.s2 import S2Manager

@dataclass
class IncaConfig(BaseDataConfig):
    """Dataclass for INCA configuration parameters."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    target_res: Optional[int] = None
    tiles: Optional[list] = None
    max_threads: Optional[int] = 1
    inca_cumulative_datasets: Optional[list] = None
    inca_cumulative_window: Optional[int] = None
    inca_range: Optional[int] = None

class IncaManager(BaseSetup):
    def __init__(self, cfg_path: IncaConfig):
        super().__init__(cfg_path)
        self.inca_subdirectory = 'INCA'
        self.inca_directory = os.path.join(self.data_directory, self.inca_subdirectory)
        self.download_dir_pattern = os.path.join(self.inca_directory, '<dataset>', 'output')  # where multiband tifs are stored
        self.nc_dir_pattern = os.path.join(self.inca_directory, '<dataset>')  # where .nc files are stored
        self.max_threads = min(self.max_threads, len(self.inca_datasets))

        # download parameters
        self.geosphere_url = 'https://dataset.api.hub.geosphere.at/v1/grid/historical/inca-v1-1h-1km'
        self.south, self.north, self.west, self.east = self.bounding_box
        self.params_dict = {
            'bbox': f'{self.south},{self.west},{self.north},{self.east}',
            'output_format': 'netcdf'
        }
    
    def prepare_directories(self):
        self.ensure_dirs_exist(self.inca_directory)
        self.ensure_dirs_exist(*[self.download_dir_pattern.replace('<dataset>', d) for d in self.inca_datasets])
        self.ensure_dirs_exist(*[self.nc_dir_pattern.replace('<dataset>', d) for d in self.inca_datasets])
        return self

    def download_images(self):
        self.logger.info('==== Downloading INCA datasets ====')
        self.logger.info(f'Download directory: {self.inca_directory}')
        
        # format dates
        self.date_format = '%Y-%m-%dT%H:%M:%S'
        self.start_date = datetime.strptime(self.start_date + 'T00:00:00', self.date_format)
        self.end_date = datetime.strptime(self.end_date + 'T00:00:00', self.date_format)
        self.logger.info(f'Start date: {self.start_date}, End date: {self.end_date}')
        
        for dataset in self.inca_datasets:
            try:
                self.download_dataset(dataset)
                self.extract_hourly_tifs(dataset)
            except Exception as e:
                self.logger.error(f'Error processing downloading {dataset}: {e}')
              
    def download_dataset(self, dataset):
        days_list = [self.start_date + timedelta(days=i) for i in range((self.end_date - self.start_date).days + 1)]
        
        for day in ChargingBar(f'{dataset}: downloading daily .nc files').iter(days_list):
            status_code = self.geosphere_daily_query(dataset, day)
            if status_code not in [200,429,'file_exists']:
                self.logger.error(f' Failed to retrieve the file. Status code: {status_code}')
            if status_code == 429:
                self.logger.error(f'Status code {status_code}: too many requests sent! Try downloading the rest of the data in ~30 min.')
                self.logger.error(f'Loop stopped at INCA dataset {dataset} and date {day}')
                break
    
    def geosphere_daily_query(self, dataset, day):
        output_dir = self.nc_dir_pattern.replace('<dataset>', dataset)
        download_start = day
        download_end = day + timedelta(hours=23)
        
        self.params_dict.update({
            'parameters': dataset,
            'start': str(download_start.strftime(self.date_format)),
            'end': str(download_end.strftime(self.date_format))
            })
        
        output_filename = f"{dataset}_{download_start.strftime('%Y%m%dT%H%M%S')}_{download_end.strftime('%Y%m%dT%H%M%S')}.nc"   
        output_path = os.path.join(output_dir, output_filename)
        if os.path.exists(output_path):
            return 'file_exists'
        
        response = requests.get(self.geosphere_url, params=self.params_dict)
        status_code = response.status_code

        if status_code == 200:  # (status code 200 = successful request)
            file_data = response.content
            with open(output_path, 'wb') as file:
                file.write(file_data)
        return status_code

    def extract_hourly_tifs(self, dataset):
        input_dir = self.nc_dir_pattern.replace('<dataset>', dataset)
        output_dir = self.download_dir_pattern.replace('<dataset>', dataset)
        available_files = list_filepaths(input_dir, ['.nc', dataset], ['.aux'], print_warning=False)
        if not available_files:
            return
        for file in ChargingBar(f'    {dataset}: georeferencing .tifs').iter(available_files):
            proc = IncaImage(file)
            proc.nc_to_tif(output_dir)
    
    def match_timestamp(self, target_timestamp):
        """Finds INCA image closest to target timestamp."""
        
        def find_nearest_hour_index(target_time):
            hour_array = np.array([time(hour, 0) for hour in np.arange(24)])
            hour_array = [datetime.combine(datetime.min, h) for h in hour_array]
            target_time = datetime.combine(datetime.min, target_time)
            
            tmp = np.zeros(len(hour_array))
            for i,h in enumerate(hour_array):
                tmp[i] = np.abs((target_time-h).total_seconds() )
            return int(tmp.argmin())
        
        template_dataset = os.listdir(self.inca_directory)[0]    
        input_dir = self.download_dir_pattern.replace('<dataset>', template_dataset)
        inca_files_list = list_filepaths(input_dir, ['.tif', template_dataset], ['.aux'], print_warning=False)
        
        if not inca_files_list:
            self.logger.warning(f'No INCA images found!')
            return None

        inca_timestamps = np.unique([(ImageProcessor.get_timestamp_from_filename(i, data_source='INCA')) for i in inca_files_list])
        inca_datetimes = [datetime.strptime(i, '%Y%m%dT%H%M%S') for i in inca_timestamps]
        inca_dates = [i.date() for i in inca_datetimes]
        
        target_datetime = datetime.strptime(target_timestamp, '%Y%m%dT%H%M%S') 
        target_date = target_datetime.date()
        target_time = target_datetime.time()
        
        if target_date not in inca_dates:
            self.logger.warning(f'No INCA data available for the date {target_date}!')
            return None
        
        idx_closest = find_nearest_index(inca_dates, target_date)
        closest_inca_date = inca_dates[idx_closest]
        closest_inca_timestamp = inca_timestamps[idx_closest]
        closest_band = find_nearest_hour_index(target_time) + 1

        return closest_inca_timestamp, closest_band

    @staticmethod
    def update_timestamp(timestamp, h):
        date_part, time_part = timestamp.split('T')  
        formatted_hour = '0'+str(h) if h < 10 else str(h)
        new_time_part = formatted_hour + time_part[2:]
        new_timestamp = date_part + 'T' + new_time_part
        return new_timestamp
        
    def reproject(self, dataset, tile, target_res, template_path, timestamp='', additional_arguments=''):
        
        input_dir = self.download_dir_pattern.replace('<dataset>', dataset)
        output_dir = os.path.join(
            self.nc_dir_pattern.replace('<dataset>', dataset), 
            f'reprojected_{str(target_res)}m',
            tile
            )
        os.makedirs(output_dir, exist_ok=True)
        
        pattern = timestamp if timestamp != '' else '.'
        images_to_reproject = list_filepaths(input_dir, ['.tif', pattern], ['.aux'], print_warning=False)
        if not images_to_reproject and timestamp != '':
            self.logger.warning(f"No INCA files to reproject for the date {datetime.strptime(timestamp, '%Y%m%dT%H%M%S').date()}!")
            return
        elif not images_to_reproject:
            self.logger.info('No INCA files to reproject!')
            return
        
        # define a band tag in case of single band resampling
        if '-b' in additional_arguments:
            args_parsed = additional_arguments.split(' ')
            band = int(args_parsed[args_parsed.index('-b') + 1])
            hour = band-1
        else:
            hour = None
            
        for f in images_to_reproject:
            if hour is None:
                output_filename = os.path.basename(f).split('.')[0] + f'_reprojected_{target_res}m.tif'
            else:
                timestamp = ImageProcessor.get_timestamp_from_filename(f, data_source='INCA')
                new_timestamp = IncaManager.update_timestamp(timestamp, hour)
                output_filename = '_'.join([os.path.basename(f).split('_')[0], new_timestamp, f'reprojected_{target_res}m.tif'])
            output_path = os.path.join(output_dir, output_filename)
            
            if os.path.exists(output_path):
                self.logger.debug(f'    File {output_path} already exists, skipping reprojection.')
                continue
            
            # self.logger.info(f'Reprojecting {f}...')
            proc = ImageProcessor(f)
            proc.reproject_by_template(template_path, output_path, 
                                        target_res=target_res,
                                        resampling_method='bilinear',
                                        target_no_data=self.inca_nodata, 
                                        additional_arguments=additional_arguments)
            self.logger.debug(f'    Reprojected {f} to {output_path}')
        return output_path
    
    def _locate_reprojection_directory(self, dataset, tile, target_res):
        """Returns the path to the directory where reprojected INCA images are stored."""
        output_dir = os.path.join(
            self.nc_dir_pattern.replace('<dataset>', dataset), 
            f'reprojected_{str(target_res)}m',
            tile
            )
        if not os.path.exists(output_dir):
            self.logger.warning(f'Reprojection directory {output_dir} does not exist and will be created.')
            self.ensure_dirs_exist(output_dir)
        return output_dir
            
    def locate_image(self, tile, dataset, resolution, template_path, timestamp, print_info=False):
        """
        Locate the INCA image for a given tile/dataset closest to timestamp.
        If dataset in self.inca_cumulative_datasets, it will locate images for the last 6 hours for cumulative sum.
        If dataset not in self.inca_cumulative_datasets, it will locate the desired number of previous hours defined in self.inca_range.
        """       
        closest_inca_timestamp, closest_inca_band = self.match_timestamp(timestamp)  # locate closest in INCA root dir
        hour = (closest_inca_band - 1)
        if closest_inca_timestamp is None:
            self.logger.error(f'Matching INCA observations not found for timestamp {timestamp}-{tile}')
            raise ValueError(f'Matching INCA observations not found for timestamp {timestamp}-{tile}')
        if print_info:
            self.logger.info(f'    The closest INCA observation to timestamp={timestamp} is: {closest_inca_timestamp} (hour={closest_inca_band-1})')
        reproj_dir = self._locate_reprojection_directory(dataset, tile, resolution)
        paths = []  # initialize paths list
        
        # define hour range based on inca_range and INCA_cumulative_datasets
        hour_range = range(hour-self.inca_cumulative_window, hour+1) if dataset in self.inca_cumulative_datasets else range(hour-self.inca_range, hour+1)  
        for h in hour_range:
            # update timestamp to the selected hour (YYYYMMDDT000000 --> YYYYMMDDThh0000)
            updated_timestamp = IncaManager.update_timestamp(closest_inca_timestamp, h)
            
            # search for image of the selected hour
            tmp = list_filepaths(reproj_dir, 
                patterns_in=[dataset, updated_timestamp], 
                patterns_out=['.aux']
            )
            if len(tmp) == 0:
                # generate if not found
                self.logger.debug(f'    No reprojected INCA images found in {reproj_dir} for timestamp {updated_timestamp}. Reprojecting...')
                path = self.reproject(dataset, tile, resolution, 
                    template_path=template_path,
                    timestamp=closest_inca_timestamp, 
                    additional_arguments=f'-b {str(h+1)}'
                )  
            elif len(tmp) == 1:
                path = tmp[0]
            else:
                raise ValueError(f"Multiple files found for dataset {dataset} and timestamp {closest_inca_timestamp} in tile {tile}. Please check the data.")        
            paths.append(path)
        return paths
            
     
class IncaImage(ImageProcessor):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.filepath = filepath
    
    def nc_to_tif(self, output_dir):
        dataset = os.path.basename(self.filepath).split('_')[0]
        output_path = os.path.join(output_dir, os.path.basename(self.filepath).replace('.nc', '.tif'))
        if not os.path.basename(self.filepath).endswith('.nc'):
            raise ValueError(f'File {self.filepath} is not a .nc file.')
        if not os.path.exists(output_path):
            cmd = ['gdalwarp', '-geoloc', f'NETCDF:{self.filepath}:{dataset}', output_path]
            ImageProcessor.run_gdal(cmd, capture_output=True)
              


