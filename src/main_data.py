# ==================================================
# Main script for data acquisition and preprocessing
# ==================================================

import os
import argparse
import configparser
from pathlib import Path
from lib.data.s2 import S2Config, S2Manager
from lib.data.ecostress import EcostressConfig, EcostressManager
from lib.data.inca import IncaConfig, IncaManager

def parse_arguments():
    parser = argparse.ArgumentParser(description='Main script for data acquisition and preprocessing')
    parser.add_argument('--config', '-c',
                        help='Path to the configuration file.', 
                        type=str,
                        default='src/config/config.yml')
    parser.add_argument('--dataset', '-d', 
                        help='Dataset to process. Choices: ECOSTRESS, S2, INCA, all.',
                        choices=['ECOSTRESS', 'S2', 'INCA', 'all'],
                        type=str,
                        default='all')  # all datasets considered if not specified
    parser.add_argument('--action', '-a', 
                        help='Action to perform. Options: download (downloads data/locates on T); preprocess (resample, reprojects to tile).',  
                        choices=['download', 'preprocess', 'all'], # note: ECOSTRESS swath2grid performed if download=True, QC performed if preprocess=True
                        type=str,
                        default='all')  # all actions performed if not specified
    parser.add_argument('--start_date', '-s', 
                        help='Start date for processing (YYYY-MM-DD).', 
                        type=str)  # returns error if action=download and start_date not specified
    parser.add_argument('--end_date', '-e', 
                        help='End date for processing (YYYY-MM-DD).', 
                        type=str)  # returns error if action=download and end_date not specified
    parser.add_argument('--target_res', '-tr', 
                        help='Target resolution for resampling.', 
                        type=int)  # returns error if action=preprocess and target_res not specified
    parser.add_argument('--tiles', '-t',
                        help='List of tiles to process (e.g., --tiles T30TYN T30TXN).',
                        nargs='+') # returns error if action=preprocess and tile not specified
    parser.add_argument('--secrets', 
                        help='Path to the secrets file with credentials.', 
                        type=str, 
                        default='src/config/secrets.ini')
    return parser.parse_args()

def validate_arguments(args):
    """Raise errors for invalid argument combinations."""
    if args.action in ('download', 'all'):
        if not args.start_date or not args.end_date:
            raise ValueError(f'Download is requested (args.action={args.action}) -- start_date and end_date are required for download')
        if args.dataset in ('S2', 'ECOSTRESS', 'all') and args.tiles is None:
            raise ValueError(f'Download of S2/ECOSTRESS is requested (args.dataset={args.dataset}) -- tiles are required for S2/ECOSTRESS download')
    if args.action in ('preprocess', 'all'):
        if args.target_res is None:
            raise ValueError(f'Preprocess is requested (args.action={args.action}) -- target_res is required for preprocess')
        if args.tiles is None:
            raise ValueError(f'Preprocess is requested (args.action={args.action}) -- tiles are required for preprocess')
    if args.action == 'preprocess' and args.dataset == 'INCA':
        raise ValueError(f'Preprocess action is not defined for INCA data (args.action={args.action}, args.dataset={args.dataset})')
        
def process_s2(args):
    s2_cfg = S2Config.from_yaml(args.config).update_from_args(args)  # build S2 config from yaml and args
    s2_processor = S2Manager(s2_cfg).prepare_directories()
    # download S2
    if args.action in ('download', 'all'):
        s2_processor.download_images(args.secrets)
    # resample S2
    if args.action in ('preprocess', 'all'):
        s2_processor.logger.info(f'==== Resampling S2 images... ====')
        for tile in args.tiles:
            s2_processor.resample(tile, resolution = args.target_res)
            
def process_ecostress(args):
    # load earthdata credentials from .ini
    secrets_cfg = configparser.ConfigParser()
    secrets_cfg.read(args.secrets)
    # build ecostress objects
    eco_cfg = EcostressConfig.from_yaml(args.config)
    eco_cfg.update_from_args(args)
    eco_cfg.update_from_dict({'secret' : dict(secrets_cfg.items('earthdata'))})  
    eco_processor = EcostressManager(eco_cfg).prepare_directories() 
    if args.action in ('download', 'all'):
        for tile in args.tiles:
            eco_processor.query_download_links(short_name='ECO_L2T_LSTE', tile=tile)
            eco_processor.download_images(additional_arguments='--show-progress -nc -q -o src/logs/wget.log') 
    if args.action in ('preprocess', 'all'):
        eco_processor.logger.info(f'==== Reprojecting ECOSTRESS images... ====')
        downloaded_paths = [
            eco_processor.download_dir / Path(i).name 
            for i in os.listdir(eco_processor.download_dir)
            if i.endswith('.tif')
            ]
        for tile in args.tiles:
            # S2 template image for reprojection
            s2_path = S2Manager(eco_cfg).define_template_image(tile,args.target_res)
            eco_processor.reproject(tile, downloaded_paths, template_path=s2_path)

def process_inca(args):
    inca_cfg = IncaConfig.from_yaml(args.config).update_from_args(args)
    inca_processor = IncaManager(inca_cfg).prepare_directories()
    # download INCA datasets
    if args.action in ('download', 'all'):
        inca_processor.download_images()
        
def log_from_main():
    """Simple logger to log from main."""
    from lib.base import BaseConfig, BaseSetup
    return BaseSetup(BaseConfig).logger
            
def main():
    args = parse_arguments()
    validate_arguments(args)
    if args.dataset in ('S2', 'all'):
        process_s2(args)      
    if args.dataset in ('ECOSTRESS', 'all'):
        process_ecostress(args)
    if args.dataset in ('INCA', 'all'):        
        process_inca(args)
    log_from_main().info('DONE.')

if __name__ == '__main__':
    main()