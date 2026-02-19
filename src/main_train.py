# =========================================================
# Main script for input data preparation and model training
# =========================================================

from attrs import fields
import yaml
import argparse
from datetime import datetime
from pathlib import Path
# from tensorflow.keras import backend as K

from lib.train.train_config import BaseTrainConfig
from lib.train.timestamps import TimestampGenerator
from lib.train.arrays import NetPreprocessorConfig, NetPreprocessor
from lib.data.ecostress import EcostressConfig, EcostressManager
from lib.data.s2 import S2Config, S2Manager
from lib.data.inca import IncaConfig, IncaManager
from lib.train.cnn import CNNModel


# from lib.train.cnn import CNNModel
# from lib.validate.analysis import PredictionAnalyzer
# from lib.validate.plots import PredictionPlotter


def parse_arguments():
    parser = argparse.ArgumentParser(description='Main script for input data preparation and model training.')
    parser.add_argument('--config', '-c',
                        help='Path to the configuration file.', 
                        type=str,
                        default='src/config/config.yml')
    parser.add_argument('--action', '-a',
                        help='Actions to perform. Options: timestamps, arrays, train.',
                        choices=['timestamps', 'arrays', 'train'],
                        nargs='+',
                        default=['timestamps', 'arrays', 'train'])  # all actions performed if not specified
    # ============================================
    # ==== arguments for timestamp generation ====
    # ============================================
    parser.add_argument('--tiles', '-t', 
                        help='Tile identifier for processing (single tile or a list).', 
                        nargs='+',
                        default = None)  # returns error if action=timestamps and tile not specified
    parser.add_argument('--num_timestamps_train', '-n_train',
                        help='Number of timestamps per tile for training.', 
                        type=int,
                        default = None)
    parser.add_argument('--num_timestamps_test', '-n_test',
                        help='Number of timestamps per tile for testing.', 
                        type=int,
                        default = None)
    parser.add_argument('--additional_start_end', '-add_se',
                        help='Start and end dates for generation of additional timestamps (YYYY-MM-DD YYYY-MM-DD).', 
                        nargs=2,
                        type=str,
                        default=None)
    parser.add_argument('--additional_freq', '-add_freq',
                        help='Frequency for generation of additional timestamps. Options: 3-day, daily, hourly.', 
                        choices=['3-day', 'daily', 'hourly'],
                        type=str,
                        default=None)
    parser.add_argument('--tag',
                        help='Tag for the output txt files. If none, tag is "latest".', 
                        type=str,
                        default='latest')
    # =====================================================
    # ==== optional arguments for array preparation ====
    # =====================================================
    parser.add_argument('--target_dataset', '-td',
                        help='Target dataset for model training. Options: ECOSTRESS.',
                        choices=['ECOSTRESS'],
                        type=str,
                        default='ECOSTRESS')
    parser.add_argument('--save_npy',
                        help='Save prepared numpy arrays to .dat files.', 
                        action='store_true', 
                        default=False)
    # =====================================================
    # ==== optional arguments for model training =====
    # =====================================================
    # parser.add_argument('--tag',
    #                     help='Tag for the output files. If none, tag is "latest".', 
    #                     type=str,
    #                     default='latest')
    parser.add_argument('--train_res', '-tr',
                        help='Resolution for training data preparation. The corresponding data must be prepeared by the main_data.py part of the pipeline.',
                        type=int,
                        default=70)
    return parser.parse_args()


def validate_arguments(args):
    """Raise errors for invalid argument combinations."""
    if 'timestamps' in args.action and args.tiles is None:
        raise ValueError(f'Timestamp generation is requested (args.action={args.action}) -- tiles are required.')
    if 'timestamps' in args.action and 'train' in args.action and 'arrays' not in args.action:
        raise ValueError(f"Meaningless combination of actions (args.action={args.action}). Skipping 'arrays' loads presaved arrays for training --> timestamp generation has no effect.")
    if args.additional_start_end:
        try:
            start_str, end_str = args.additional_start_end
        except ValueError:
            raise ValueError(
                f"Start and end dates for additional timestamp generation must be provided as two values 'YYYY-MM-DD YYYY-MM-DD' (args.additional_start_end={args.additional_start_end}).")
        date_format = "%Y-%m-%d"
        try:
            start = datetime.strptime(start_str, date_format).date()
            end = datetime.strptime(end_str, date_format).date()
        except ValueError:
            raise ValueError("Dates must be in format YYYY-MM-DD")
        if start >= end:
            raise ValueError("Start date must be before end date")
    if args.additional_freq is not None and args.additional_start_end is None:
        raise ValueError(f"Frequency for additional timestamp generation can only be specified if start and end dates are also specified (args.additional_freq={args.additional_freq}, args.additional_start_end={args.additional_start_end}).")

        
def run_timestamp_generation(args):
    # initiate timestamp object
    tstp_cfg = BaseTrainConfig.from_yaml(args.config).update_from_args(args)
    ts_generator = TimestampGenerator(cfg=tstp_cfg).prepare_directories()
    ts_generator.logger.info("==== Generating timestamp list... ====")
    # initiate ECOSTRESS object
    eco_cfg = EcostressConfig.from_yaml(args.config)
    eco_manager = EcostressManager(eco_cfg)
    # generate timstamps
    ts_generator.find_ecostress_images(eco_manager)
    for tile in args.tiles:
        # generate timestamps for a given tile from ECOSTRESS
        ts_data = ts_generator.generate_timestamps_from_ecostress(eco_manager, tile)
        main_list, suppl_list = ts_generator.train_test_split_ecostress(ts_data, 
            num_train=args.num_timestamps_train, num_test=args.num_timestamps_test, seed=None
        )
        # update timestamp dictionary with tile-specific lists
        ts_generator.update_timestamps(key=tile, values=main_list, update_type='main')
        ts_generator.update_timestamps(key=tile, values=suppl_list, update_type='supplementary')
        # update supplementary dict with additinonal timestamps if requested
        if args.additional_start_end:
            additional_start, additional_end = args.additional_start_end
            freq = args.additional_freq
            if freq is None:
                additional_timestamps = ts_generator.generate_timestamps_from_dates(additional_start, additional_end)
            else:
                additional_timestamps = ts_generator.generate_timestamps_from_dates(additional_start, additional_end, frequency=freq)
            # update supplementary timestamps dict
            ts_generator.update_timestamps(key=tile, values=additional_timestamps, update_type='supplementary')
    #save lists to txt files
    for d, list_type in zip([ts_generator.timestamp_dict, ts_generator.timestamp_dict_supplementary], ['train', 'eval']):
        # flatten dict {tile: timestamps} --> list [(tile, t1), (tile, t2), ...]
        flattened_list = ts_generator.flatten_to_tuples(d)
        # prepare output path
        tile_str = args.tiles[0] if len(args.tiles) == 1 else 'multiple_tiles'
        output_name = f'{tile_str}_timestamps_{list_type}_{args.tag}.txt'
        output_path = Path(ts_generator.net_input_directory_txt) / output_name
        # save to txt
        TimestampGenerator.list_to_txt(flattened_list, output_path)
        ts_generator.logger.info(f"{list_type.capitalize() if list_type == 'train' else 'Supl.'} timestamp list saved to {output_path}")


def run_array_preparation(args):
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    # initiate all image objects
    eco_cfg = EcostressConfig.from_yaml(args.config)
    eco_manager = EcostressManager(eco_cfg)
    s2_cfg = S2Config.from_yaml(args.config)
    s2_manager = S2Manager(s2_cfg)
    inca_cfg = IncaConfig.from_yaml(args.config)
    #
    # TBD!!! improve reading fromv various sources
    # some keys not defineed by from_yaml since it is based on BaseConfig, need to load config_dict manually
    #
    inca_cfg.update_from_dict(config_dict)
    inca_manager = IncaManager(inca_cfg)
    
    # load timestamps
    tstp_cfg = BaseTrainConfig.from_yaml(args.config).update_from_args(args)
    ts_generator = TimestampGenerator(cfg=tstp_cfg).prepare_directories()
    samples = ts_generator.load_tuples_from_txt()
    
    # initiate array preparator
    arr_cfg = NetPreprocessorConfig.from_yaml(args.config).update_from_args(args)
    arr_prep = NetPreprocessor(cfg=arr_cfg).prepare_directories()
    
    for s in samples:
        tile, timestamp = s
        ts_generator.logger.info(f"Preparing input arrays for tile {tile} at timestamp {timestamp}...")
        # locate all input images
        input_datasets = [
            arr_prep.target_dataset,
            *arr_prep.numerical_datasets
        ]
        arr_prep.locate_input_data(tile, timestamp, input_datasets,
            target_manager=eco_manager,
            s2_manager=s2_manager,
            inca_manager=inca_manager
        )
    
    # prepare X and Y arrays for CNN training
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    X, Y = arr_prep.prepare_input_arrays(
        samples=samples, 
        resolution=args.train_res,
        target_manager=eco_manager,
        s2_manager=s2_manager,
        inca_manager=inca_manager,
        window=None, 
        patch_size=config_dict['patch_size'], 
        overlap_pixels=False, 
        data_type='train',  # 'train' or 'test',
        save_npy=args.save_npy  # save the prepared arrays to .dat
    )
    return arr_prep, X, Y


def run_training(args, arr_prep=None, X=None, Y=None):
    if arr_prep is None or X is None or Y is None:
        # TBD!!!! load arrays from .dat if not provided as arguments
        raise ValueError("Code not working yet with presaved arrays. Please run --action timestams arrays train for now.")
    

    def filter_config_dict(config_dict, dataclass_type):
        from dataclasses import fields
        valid_keys = {f.name for f in fields(dataclass_type)}
        return {k: v for k, v in config_dict.items() if k in valid_keys}

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    filtered_config = filter_config_dict(config_dict, BaseTrainConfig)
    cfg = BaseTrainConfig(**filtered_config)
    model = CNNModel(cfg, cnn_preparator=arr_prep)
    # model = CNNModel(args.config, cnn_preparator = arr_prep)
    model.train(X, Y)


if __name__ == '__main__':    
    args = parse_arguments()
    validate_arguments(args)
    print(f"Running processes: {args.action} for tiles: {args.tiles}")
    
    # generate timestamps for training
    if 'timestamps' in args.action:
        run_timestamp_generation(args)
    if 'arrays' in args.action:
        arr_prep, X, Y = run_array_preparation(args)        
    if 'train' in args.action:
        run_training(args, arr_prep=arr_prep, X=X, Y=Y)