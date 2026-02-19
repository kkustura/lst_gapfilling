
import sys
import os
import subprocess
import rasterio
import csv
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from progress.bar import ChargingBar
from typing import Optional
from pathlib import Path
import configparser

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.utils import list_filepaths, find_nearest_index
from lib.base import BaseSetup
from lib.data.data_config import BaseDataConfig
from lib.data.image import ImageProcessor

@dataclass
class S2Config(BaseDataConfig):
    """Dataclass for Sentinel-2 specific configuration parameters."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    target_res: Optional[int] = None
    tiles: Optional[list] = None

class S2Manager(BaseSetup):
    def __init__(self, cfg: S2Config):
        super().__init__(cfg) 
        self.s2_subdirectory = 'S2'
        self.s2_directory = os.path.join(self.data_directory, self.s2_subdirectory)
        self.download_dir = os.path.join(self.s2_directory, 'output')
        self.resampled_dir = os.path.join(self.s2_directory, f'resampled_<res>m')

        # download parameters
        self.s3_url = 's3://eodata/Sentinel-2/MSI/L2A'

    def prepare_directories(self):
        self.ensure_dirs_exist(
            self.s2_directory,
            self.download_dir
            )
        return self
    
    def _generate_s3_cfg(self, secrets_path):
        """ 
        Generate a temporary .s3cfg from secrets.ini file for S3 download.
        Input:
            - secrets_path: path to secrets file. S3 data is assume dto be under [s3] heading

        """
        if not Path(secrets_path).exists():
            raise FileNotFoundError(f'Secrets file not found at {secrets_path}. Please provide a valid path to the secrets.ini file with S3 credentials.')
        secrets_cfg = configparser.ConfigParser()
        secrets_cfg.read(secrets_path)
        if 's3' not in secrets_cfg.sections():
            raise ValueError(f'S3 credentials not found in secrets file {secrets_path}. Please make sure the file contains [s3] section.')
        s3_data = dict(secrets_cfg.items('s3'))
        if not all(key in s3_data.keys() for key in ['access_key', 'secret_key', 'host_base', 'host_bucket', 'human_readable_sizes', 'use_https', 'check_ssl_certificate']):
            raise ValueError(f'Missing S3 credentials in secrets file {secrets_path}. Please make sure the file contains the following keys under [s3] section: access_key, secret_key, host_base, host_bucket, human_readable_sizes, use_https, check_ssl_certificate.')
        tmp_dir = Path('tmp')
        tmp_dir.mkdir(exist_ok=True)
        self.s3cfg_path = tmp_dir / '.s3cfg'
        with open(self.s3cfg_path, 'w') as f:
            f.write(f'[default]\n')
            f.write(f'access_key={s3_data["access_key"]}\n')
            f.write(f'secret_key={s3_data["secret_key"]}\n')
            f.write(f'host_base={s3_data["host_base"]}\n')
            f.write(f'host_bucket={s3_data["host_bucket"]}\n')
            f.write(f'human_readable_sizes={s3_data["human_readable_sizes"]}\n')
            f.write(f'use_https={s3_data["use_https"]}\n')
            f.write(f'check_ssl_certificate={s3_data["check_ssl_certificate"]}\n')
        self.logger.info(f'S3 config file generated at {self.s3cfg_path}')

    def search_s3(self, tile, start_date, end_date, secrets_path, reset=True):
        """Find S3 images for a given tile and time window. Returns a list of S3 paths."""
        if reset:
            self.download_links = []
        elif not hasattr(self, 'download_links'):
            self.download_links = []

        self._generate_s3_cfg(secrets_path)
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')

        current_dt = start_date_dt
        while current_dt <= end_date_dt:
            date_str = current_dt.strftime('%Y/%m/%d')
            current_dt += timedelta(days=1)
            current_dir = f'{self.s3_url}/{date_str}/'
            command = [
                's3cmd', '-c', str(self.s3cfg_path), 'ls', current_dir, '|', 'grep', tile
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            result_filtered = [line.split('DIR ')[1] for line in result.stdout.splitlines() if tile in line]
            if result_filtered:
                print(result_filtered[0] if len(result_filtered)==1 else result_filtered)
            self.download_links.extend(result_filtered)

    def download(self, bands=["B04", "B08", "SCL"]):
        """Download selected bands from the S3 paths in self.download_links."""
        if not hasattr(self, 'download_links'):
            raise ValueError('No download_links attribute found. Please run search_s3() method first.')
        if not self.download_links:
            self.logger.warning('No images found to download for the specific tile and time window.')
            return
        if not all(b in ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B11", "B12", "SCL"] for b in bands):
            raise ValueError(f'Invalid band name(s) in {bands}. Please use one of the following: B02, B03, B04, B05, B06, B07, B08, B08A, B11, B12, SCL')
        if not hasattr(self, 's3cfg_path') or not self.s3cfg_path.exists():
            raise ValueError('S3 config file not found. Please run search_s3() method first to generate the S3 config file.')
        for prod in self.download_links:
            prod = prod.strip()  # clean leading spaces
            img_data_path = prod + "GRANULE/"
            # list GRANULE subfolder
            list_granules = subprocess.run(
                ["s3cmd", "-c", str(self.s3cfg_path), "ls", img_data_path],
                capture_output=True, text=True, check=True)
            granule_dirs = [line.split()[-1] for line in list_granules.stdout.splitlines()]
            # TBD!!!! add filtering (skip granule if already downloaded)
            # download images for each band
            for granule in granule_dirs:
                for band in bands:
                    res_dir = 'R10m' if band in ["B02", "B03", "B04", "B08"] else 'R20m'
                    img_data = granule + f"IMG_DATA/{res_dir}/"
                    img = f"{img_data}*{band}*.jp2"
                    cmd = ["s3cmd", "-c", str(self.s3cfg_path), "get", img, str(self.download_dir)]
                    subprocess.run(cmd, check=False)
        # delete tmp s3cfg file
        self.s3cfg_path.unlink()

    def download_images(self, secrets_path):
        self.logger.info("==== Downloading S2 images ====")
        self.logger.info(f'download directory: {self.download_dir}')
        self.logger.info(f'Start date: {self.start_date}, end date: {self.end_date}')
                
        for tile in self.tiles:
            try:
                self.search_s3(tile, self.start_date, self.end_date, secrets_path)
                self.download(bands=self.s2_datasets)
            except Exception as e:
                self.logger.error(f'Error downloading/filtering tile {tile}: {e}')

    # TBD!!!! add filtering of duplicates
    # TBD!!!! add filtering by cloud cover

    def resample(self, tile, resolution, timestamp='', extension='.jp2'):
        """
        Wrapper for resampling S2 images.
        Input:
            - tile: Sentinel-2 tile / pattern to look for in an image
            - timestamp: if timestamp is provided, the resampling will only be carried out for this single
            timestamp. It must be provided in the format 'YYYYMMDDThhmmss'.
            In case of timestamp='', the resampling is done for all of the images in the input self.download_dir.
        """
        self.logger.info(f'Resampling tile {tile}...')
        output_dir = os.path.join(str(self.resampled_dir).replace('<res>', str(resolution)), tile)
        os.makedirs(output_dir, exist_ok=True)
        
        pattern = timestamp if timestamp != '' else '.'
        images_to_resample = list_filepaths(self.download_dir, [extension, tile, pattern], ['.aux'], print_warning=False)
        if not images_to_resample and timestamp != '':
            self.logger.warning(f'No S2 images found for tile {tile} at date {datetime.strptime(timestamp, "%Y%m%dT%H%M%S")}.')
            return
        elif not images_to_resample:
            self.logger.info(f'No S2 images found for tile {tile}.')
            return
        
        template_image = images_to_resample[0]
        for f in ChargingBar(f'Resampling {len(images_to_resample)} images...').iter(images_to_resample):
            output_filename = os.path.basename(f).split('.')[0][:-4]+f'_resampled_{resolution}m.tif'
            output_path =  os.path.join(output_dir,output_filename)
            if os.path.exists(output_path):
                # subprocess.run(['rm', raster_out])
                continue
            
            proc = ImageProcessor(f)
            proc.reproject_by_template(template_image, output_path, 
                                       target_res=resolution, 
                                       resampling_method = 'average', 
                                       use_src_nodata = True, 
                                       target_no_data=self.s2_nodata
                                       )
        self.logger.info(f'Resampled images saved to {output_dir}')
        return output_path
    
    def define_template_image(self,tile,res):
        """Defines a template filapath for a given tile (takes any from the list)"""
        s2_image_dir = os.path.join(self.s2_directory,f'resampled_{str(res)}m', tile)
        if not os.path.exists(s2_image_dir):
            self.logger.error(f'S2 image directory {s2_image_dir} does not exist. Please make sure the S2 images are downloaded and resampled to {str(res)} m.')
            raise FileNotFoundError(f'Directory not found: {s2_image_dir}')
        
        tmp = list_filepaths(s2_image_dir, ['.tif', tile], ['.aux'])
        if not tmp:
            self.logger.error(f'No S2 images found in {s2_image_dir}. Please make sure the S2 images are downloaded and resampled to {str(res)} m.')
            raise FileNotFoundError(f'No S2 images found in {s2_image_dir}')
        else:
            s2_path = tmp[0]
            self.logger.debug(f'    Template image for reprojection: {s2_path}')
        return s2_path
    
    def locate_image(self, tile, band, resolution, timestamp, print_info=False):
        """Locate the S2 image for a given tile/band closest to timestamp."""
        closest_s2_timestamp = self.match_timestamp(timestamp, tile)  # locate closest in S2 root dir
        if closest_s2_timestamp is None:
            self.logger.error(f'Matching S2 observations not found for timestamp {timestamp}-{tile}')
            raise ValueError(f'Matching S2 observations not found for timestamp {timestamp}-{tile}')
        
        # search resampled S2 images
        tmp = list_filepaths(
            os.path.join(self.resampled_dir.replace('<res>', str(resolution)), tile),
            patterns_in=[band, closest_s2_timestamp], patterns_out=['.aux']
        )
        if len(tmp) == 0:
            path = self.resample(tile,resolution,timestamp=closest_s2_timestamp)  # generate if not found
        elif len(tmp) == 1:
            path = tmp[0]
        else:
            self.logger.error(f"Multiple files found for band {band} and timestamp {closest_s2_timestamp} in tile {tile}. Please check the data.")
            raise ValueError("Multiple files found")
        if print_info:
            self.logger.info(f'    The closest S2 observation to timestamp={timestamp} at the tile {tile} is: {closest_s2_timestamp}')
        return path
    
    def match_timestamp(self, target_timestamp, tile):
        """Finds S2 scene closest to target timestamp"""
        target_date = datetime.strptime(target_timestamp.split('T')[0], '%Y%m%d') 
        
        s2_scenes = list_filepaths(self.download_dir, ['.jp2', tile], ['.aux'])
        if len(s2_scenes)==0:
            self.logger.warning(f'No S2 images found for tile {tile}!')
            return None
        s2_timestamps = np.unique([ImageProcessor.get_timestamp_from_filename(i, data_source='S2') for i in s2_scenes])
        s2_dates = [datetime.strptime(i, '%Y%m%dT%H%M%S') for i in s2_timestamps]
        idx_closest = find_nearest_index(s2_dates, target_date)
        return s2_timestamps[idx_closest]     










    # def find_images_in_time_window(self, tile):
    #     """Check if images for the tile already exist in the time window. Returns a list of images in self.download_dir."""
    #     out = []
    #     for i in list_filepaths(self.download_dir, ['.jp2', tile], ['.aux'], print_warning=False):
    #         timestamp = ImageProcessor.get_timestamp_from_filename(i, data_source='S2')
    #         date      = datetime.strptime(timestamp, '%Y%m%dT%H%M%S')
    #         if datetime.strptime(self.start_date, '%Y-%m-%d') <= date <= datetime.strptime(self.end_date, '%Y-%m-%d') + timedelta(days=1):
    #             out.append(i)   
    #     return out
        
    # def download_tile(self, tile):
    #     """Run the GC download docker"""
    #     # clear any files from the previous run
    #     self.logger.info(f"Downloading {tile}...")
    #     self.logger.info(f'Cleaning up temporary directory from old data: {self.tmp_dir}...')
    #     for file in os.listdir(self.tmp_dir):
    #         file_path = os.path.join(self.tmp_dir, file)
    #         if os.path.isfile(file_path):
    #             subprocess.run(['rm', file_path])
                                
    #     # check if time window already downloaded
    #     self.logger.info(f'Checking for already downloaded images in {self.download_dir}...')
    #     already_downloaded_images = self.find_images_in_time_window(tile)
        
    #     if already_downloaded_images:
    #         self.logger.warning(f'Skipping download for {tile}: {int(len(already_downloaded_images)/7)} scenes already downloaded for this time window.')
    #         self.skipping_download = True
    #         return self
    #     else:
    #         self.skipping_download = False
        
    #     # # download S2 images
    #     # self.logger.info(f"Running download docker...")
    #     # subprocess.run(['sudo', 'docker', 'run', '--network=host', '-v', '/mnt:/mnt', 'imagehub.geoville.com/gc_dl:development', '/bin/bash', 
    #     #                 '-c', f'julia -p 8 s2_download.jl -o {self.tmp_dir} -t {tile} -s {self.start_date} -e {self.end_date} --token LAWOtJ3n2N1vWjQ8mxw0kQVZyD2Ygm3njXjw6HEy96 --level L2A --bands B04 B08 SCL --cloud_max {int(self.cloud_cover_max)}']) # > /dev/null 2>&1'])
    #     return self
     
    # def filter_tile(self, tile):
    #     """Removes images with pct_valid < pct_valid_threshold for the current year"""
    #     self.logger.info(f'Removing images with < {self.pct_valid_threshold}% valid pixels...')
    #     year = self.start_date[:4]
    #     path_to_downloaded_images = os.path.join(self.tmp_dir, 'tmp')  # folder created by the docker
    #     file_list = list_filepaths(path_to_downloaded_images, ['.jp2', tile, year], ['.aux'], print_warning=False)
        
    #     for f in file_list:
    #         with rasterio.open(f, 'r') as src:
    #             data = src.read(1)
    #             data = np.nan_to_num(data)  # replace nan by 0
    #             pct_valid = 100 * (data != 0).sum() / (src.width * src.height)  # percent of valid px (!=0)
    #             if pct_valid < self.pct_valid_threshold:
    #                 subprocess.run(['rm', f])  # remove images with pct_valid < pct_valid_threshold
    #                 if 'B04' in f and tile in f:
    #                     self.logger.info(f'Removing {os.path.basename(f)}    ({int(pct_valid)} % valid pixels)')
    #             else: 
    #                 # move valid images to the final location
    #                 if self.tmp_directory != self.data_directory:
    #                     subprocess.run(['sudo', 'mv', f, self.download_dir])
    #                 if 'B04' in f and tile in f:
    #                     self.logger.info(f'Keeping {os.path.basename(f)}    ({int(pct_valid)} % valid pixels)')

    #     filtered_list = list_filepaths(self.download_dir, ['.jp2', tile, year], ['.aux'], print_warning=False)
    #     if len(filtered_list) == 0:
    #         self.logger.warning(f'No S2 images remaining for {tile} after filtering!')
    #     else:
    #         available_dates = [ImageProcessor.get_timestamp_from_filename(item, data_source='S2') for item in filtered_list]
    #         available_dates = np.unique(available_dates)
    #         for d in available_dates:
    #             file_list_day = [item for item in filtered_list if d in item]
    #             if len(file_list_day) % 7 != 0:
    #                 self.logger.warning(f'Removing {d}    (not all bands are downloaded). ')
    #                 for f_d in file_list_day:
    #                     subprocess.run(['rm', os.path.join(path_to_downloaded_images, f_d)])
     




    

    # def analyze_s2_from_csv(self, path_to_csv_dir, tile_id):
    #     """
    #     Analyze a csv file of a given S2 tile_id, and return a list of cloud_free observations for a given time window.
    #     Input: 
    #         - path_to_csv_dir: path to directory with csv files generated by the S2 downloading Docker
    #         - tile_id: id of the S2 tile
    #     Output: 
    #         - cloud_free_timestamps (cloud_free_dates): list of timestamps (dates) with cloud cover < self.cloud_cover_max
    #     """
    #     try:
    #         csv_path = [os.path.join(path_to_csv_dir,i) for i in os.listdir(path_to_csv_dir) if '.csv' in i and tile_id in i][0]
    #     except Exception:
    #         self.logger.warning(f'S2 data for tile {tile_id} is not available! Returning None.')
    #         return None, None
        
    #     cloud_free_timestamps = []
    #     cloud_free_dates = []
    #     with open(csv_path, 'r') as csv_file:
    #             reader = csv.DictReader(csv_file)
    #             for row in reader:
    #                 acq_timestamp = row['PRODUCT_ID'].split('_')[2]  # acquisition date is taken as the first date in PRODUCT_ID
    #                 acq_date = datetime.strptime(acq_timestamp,'%Y%m%dT%H%M%S')
    #                 cloud_cover = row['CLOUD_COVER']
    #                 if (self.start_date <= acq_date <= self.end_date) and (float(cloud_cover) < self.cloud_cover_max):
    #                     cloud_free_timestamps.append(acq_timestamp)
    #                     cloud_free_dates.append(acq_date)
    #     cloud_free_timestamps.sort()
    #     cloud_free_dates.sort()
    #     return cloud_free_timestamps, cloud_free_dates  

