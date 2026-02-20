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
from lib.base import BaseSetup, BaseConfig
from lib.data.image import ImageProcessor

@dataclass
class EcostressConfig(BaseConfig):
    """Dataclass for ECOSTRESS specific configuration parameters."""
    data_directory: str
    ecostress_pct_valid_threshold: int
    ecostress_nodata: int
    bounding_box: list
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
            # 'temporal': f'{self.start_date},{self.end_date}'
            'temporal': f'{self.start_date}T00:00:00Z,{self.end_date}T23:59:59Z'
        }
        
    def prepare_directories(self):
        self.ensure_dirs_exist(
            self.ecostress_directory,
            self.download_dir
            )
        return self
        
    def filter_by_tile(self, download_links, tile):
        available_tiles = [Path(link).name.split('_')[5] for link in download_links]
        available_tiles = np.unique(available_tiles)
        if tile not in available_tiles:
            self.logger.warning(f'No images found for tile {tile} for the period {self.start_date} to {self.end_date}. Please check if tile within the bounding box.')
            download_links = []
        else:
            download_links = [link for link in download_links if f'_{tile}_' in link]
        return download_links

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
        # filter by tile
        download_links=self.filter_by_tile(download_links, tile)
        # filter only LST files
        download_links = [link for link in download_links if link.endswith('_LST.tif')] # or link.endswith('_QC.tif')] 
        for f in download_links:
            self.logger.info(f'    {os.path.basename(f)}')
        self.download_links = download_links
        return self

    def download_images(self, additional_arguments=''):
        """
        Downloads ECOSTRESS images.
        If pct_valid_threshold > 0 is provided, images with pct valid pixels less than will be removed.
        """
        def _download_single_image(link):
            """Helper function to download a file from a single link using wget."""
            cmd = f'wget {additional_arguments} --user {self.username} --password {self.password} {link} --directory-prefix {self.download_dir}'
            subprocess.run(cmd.split(' '))
            return self.download_dir / Path(link).name
        def _check_pct_valid(path):
            """Helper function to compute percentage of valid pixels in image."""
            with rasterio.open(path, 'r') as src:
                data = src.read(1)
                data = np.nan_to_num(data, 0)
                num_valid = (data != 0).sum()
                num_total = src.height * src.width
            pct_valid = 100 * (num_valid / num_total) if num_total > 0 else 0
            if pct_valid < self.ecostress_pct_valid_threshold:
                self.logger.warning(f'Image {path} has only {pct_valid:.0f}% valid pixels and will be removed.')
                subprocess.run(['rm', path])
            else:
                self.download_paths.append(output_path)

        if getattr(self, 'secret', None) is None:
            raise ValueError('No Earthdata credentials provided -- data download cannot proceed. Please use method update_from_dict({"secret": secret}) to add attribute to EcostressConfig (secret={"username": user, "password": pw}).')
        self.username = self.secret['username']  # username for the Earthdata account
        self.password = self.secret['password']  # password for the Earthdata account
        if hasattr(self, 'download_links') == False:
            self.logger.error('No attribute download_links found. Please run obtain_download_links() first.')
            raise AttributeError('No attribute download_links found. Please run obtain_download_links() first.')
        if not self.download_links:
            return
        self.logger.info(f'Downloading {len(self.download_links)} ECOSTRESS images to {self.download_dir}...')
        self.download_paths = []
        for link in self.download_links:
            try:
                output_path = _download_single_image(link)
                _check_pct_valid(output_path)
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
           
