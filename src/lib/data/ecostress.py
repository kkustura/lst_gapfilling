import sys
import os
import subprocess
import requests
import rasterio
import shlex
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from progress.bar import ChargingBar
from rasterio.windows import Window
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.utils import list_filepaths
from lib.base import BaseSetup
from lib.data.data_config import BaseDataConfig
from lib.data.image import ImageProcessor

@dataclass
class EcostressConfig(BaseDataConfig):
    """Dataclass for ECOSTRESS specific configuration parameters."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    target_res: Optional[int] = None
    tiles: Optional[list] = None
    max_threads: Optional[int] = 1  # TBD!!! define as parameter
    secret: Optional[dict] = None


class EcostressManager(BaseSetup):
    def __init__(self, cfg: EcostressConfig):
        super().__init__(cfg)
        self.ecostress_subdirectory = 'ECOSTRESS'
        self.ecostress_directory = Path(self.data_directory) / self.ecostress_subdirectory
        self.download_dir = self.ecostress_directory / 'output' # where downloaded, georeferenced images are saved
        self.reproj_dir = self.ecostress_directory / 'reprojected' # where reprojected images are saved
        
        # download parameters
        self.earthdata_url = 'https://cmr.earthdata.nasa.gov/search/granules.umm_json' # base URL for the website image search
        
        ## parameters for the Earthdata Search Tool command ##
        self.page_size = 50
        self.south, self.north, self.west, self.east = self.bounding_box
        self.params_dict = {
            'page_size': self.page_size, 
            'sort_key': 'start_date', 
            'bounding_box': f'{self.west},{self.south},{self.east},{self.north}',
            'temporal': f'{self.start_date},{self.end_date}'
            # 'temporal': f'{self.start_date}T00:00:00Z,{self.end_date}T23:59:59Z'
        }
        
    def prepare_directories(self):
        self.ensure_dirs_exist(
            self.ecostress_directory,
            self.download_dir
            )
        return self
        
    def filter_downloaded_links(self, download_links, tile):
        """Filter links corresponding to ECOSTRESS 2-LSTE images that are already found in the download directory."""
        # timestamps of the images already found in the download directory
        already_available_timestamps = np.unique([
            ImageProcessor.get_timestamp_from_filename(f, data_source='ECOSTRESS') for f in os.listdir(self.download_dir)
        ]) 
        for i in reversed(range(len(download_links))):
            link = download_links[i]
            base_name = os.path.basename(link).split('.')[0]
            timestamp = ImageProcessor.get_timestamp_from_filename(link, data_source='ECOSTRESS')
            if already_available_timestamps.size > 0 and timestamp in already_available_timestamps and tile in base_name:
                self.logger.warning(f'    -- Skipping: {base_name} (already found in {self.download_dir})')
                download_links.remove(link)   

    def query_download_links(self, short_name, tile):
        """
        Returns a list of ECOSTRESS download links (analogous to the download list generated via web interface).
        Input:
            - short_name: short name for the dataset to download. In our case:  ECO2LSTE and ECO1GEO
        Output:
            - self.download_links: list of download links for the images satisfying the spatial and temporal requirements.     
        """
        self.logger.info(f'==== Downloading {short_name} images... ====')
        self.logger.info(f'Querying Earthdata for period {self.start_date} to {self.end_date}...')
        self.params_dict.update({'short_name': short_name}) 
        if getattr(self, 'secret', None) is None:
            raise ValueError('No Earthdata credentials provided -- data download cannot proceed. Please use method update_from_dict({"secret": secret}) to add attribute to EcostressConfig (secret={"username": user, "password": pw}).')
        self.username = self.secret['username']  # username for the Earthdata account
        self.password = self.secret['password']  # password for the Earthdata account

        # loop over pages until all granules satisfying spatial and temporal requirements are retrieved
        all_granules = []
        page_number = 1  # initial page number
        while True:
            self.params_dict['page_num'] = page_number  
            response = requests.get(self.earthdata_url, params=self.params_dict)
            if response.status_code == 200:  # status code 200 = successful request
                data = response.json()  # parse the response
                granules = data.get('items', [])  # extract granules from the response
                all_granules.extend(granules) 
                if len(granules) < self.params_dict['page_size']:
                    break  # break when last page reached
                else:
                    page_number += 1    
            else:
                self.logger.error(f'Failed to retrieve data. Status code: {response.status_code}')
                break  

        # Extract download links from each granule's metadata
        download_links = []
        for granule in all_granules:
            related_urls = granule.get("umm", {}).get("RelatedUrls", [])
            for url_info in related_urls:
                if url_info.get("Type") == "GET DATA":
                    download_links.append(url_info.get("URL"))
        if len(download_links) == 0:
            self.logger.warning(f'No images found for download from Earthdata for the period {self.start_date} to {self.end_date}. No download will be performed.')
            self.download_links = []
            return self
        
        # TBD!!! move to a separate function
        # filter only this tile
        available_tiles = [Path(link).name.split('_')[5] for link in download_links]
        available_tiles = np.unique(available_tiles)
        if tile not in available_tiles:
            self.logger.warning(f'No images found for tile {tile} for the period {self.start_date} to {self.end_date}. Please check if tile within the bounding box.')
            self.download_links = []
            return self
        else:
            download_links = [link for link in download_links if f'_{tile}_' in link]
        
        # filter only LST files
        download_links = [link for link in download_links if link.endswith('_LST.tif')] # or link.endswith('_QC.tif')] 
        num_found = len(download_links)
        
        for f in download_links:
            self.logger.info(f'    {os.path.basename(f)}')

        # filter out data that is already downloaded
        self.filter_downloaded_links(download_links, tile)
        if len(download_links) == 0:
            self.logger.info(f'No new images will be downloaded: all images ({self.start_date} to {self.end_date}) already found in {self.download_dir}.')
        else:
            self.logger.info(f'    -- {len(download_links)}/{num_found} images are not found in {self.download_dir} -- added to download list.')
        self.download_links = download_links
        return self

    def download_images(self, additional_arguments=''):
        """Downloads ECOSTRESS images."""
        def _download_single_image(link):
            """Helper function to download a file from a single link using wget."""
            cmd = f'wget {additional_arguments} --user {self.username} --password {self.password} {link} --directory-prefix {self.download_dir}'
            subprocess.run(cmd.split(' '))

        if hasattr(self, 'download_links') == False:
            self.logger.error('No attribute download_links found. Please run obtain_download_links() first.')
            raise AttributeError('No attribute download_links found. Please run obtain_download_links() first.')
        
        if not self.download_links:
            return
        
        self.logger.info(f'Downloading {len(self.download_links)} ECOSTRESS images to {self.download_dir}...')
        self.logger.info(f"    -- Number of threads: {self.max_threads}")
        with ThreadPoolExecutor(self.max_threads) as executor:
            futures = {executor.submit(_download_single_image, link): link for link in self.download_links}
            for future in as_completed(futures):
                link = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f'Error downloading {link}: {e}')       
        self.logger.info('    -- Download completed.')
        


    def reproject(self, tile_id, images_to_reproject, template_path):
        """
        Wrapper for reprojecting ECOSTRESS images.
        Inputs:
            - tile_id: S2 tile ID (e.g., T32TQM)
            - images_to_reproject: list of ECOSTRESS images to reproject
            - template_path: path to the S2 template image for reprojection
        """
        output_dir = os.path.join(self.reproj_dir, tile_id)
        os.makedirs(output_dir, exist_ok=True)
        
        counter = 0  # initialize counter for removed images
        for f in ChargingBar(f'Reprojecting {len(images_to_reproject)} ECOSTRESS images to {tile_id} tile...').iter(images_to_reproject): 
            if 'QC' in os.path.basename(f): 
                output_filename = os.path.basename(f).split('.')[0][:-3] + '_reprojected.tif'
            else: 
                output_filename = os.path.basename(f).split('.')[0] + '_reprojected.tif'
            output_path = os.path.join(output_dir, output_filename)
        
            if os.path.exists(output_path):
                continue
            
            # reproject to S2 tile
            proc = ImageProcessor(f)
            proc.reproject_by_template(template_path, output_path, 
                                       resampling_method = 'bilinear',
                                       target_no_data=999 # TBD!!!! hardcoded nodata
                                       )  # TBD!!! resampling method?
                            
            # check if the output image is empty; if so, remove it
            with rasterio.open(output_path, 'r') as src:
                data = src.read(1)
                data = np.nan_to_num(data, src.nodata)
                pct_valid = 100 * (data != src.nodata).sum() / (src.width * src.height)
            
            # remove images with pct_valid < 1
            if pct_valid < 1:
                subprocess.run(['rm', output_path])
                counter += 1
        self.logger.info(f'{len(images_to_reproject)-counter}/{len(images_to_reproject)} images had valid ECOSTRESS pixels in {tile_id} and were saved to {output_dir}.')

    def locate_image(self, tile, timestamp):
        """Locate the target image for a given tile and timestamp."""
        path_to_target = list_filepaths(
            os.path.join(self.reproj_dir, tile),
            patterns_in=[timestamp], patterns_out=['.aux']
        )
        if len(path_to_target)==1:    
            return path_to_target[0]
        elif len(path_to_target)==0:
            return None  # in case of predictions outside ECOSTRESS coverage
           
