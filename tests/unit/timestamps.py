import pytest
import sys
import os
import numpy as np
from pathlib import Path
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lib.train.timestamps import TimestampConfig, TimestampGenerator
from lib.data.ecostress import EcostressConfig, EcostressManager, EcostressImage

test_cfg = TimestampConfig(
    net_input_directory="",
    net_output_directory="",
    year=2022,
    month_range=[6, 7], 
    hour_range=[10, 12],
    target_dataset="LST",
    S2_datasets=[],
    inca_datasets=[],
    other_datasets=[],
    categorical_datasets=[],
    inca_range=0,
    inca_cumulative_datasets=[],
    inca_cumulative_window=0,
    filter_by_SCL=False,
    patch_size=5,
    max_epochs=10,
    dropout_rate=0.5,
    train_valid_split_shuffle=True,
    validation_split=0.2,
    num_timestamps_train=10,
    num_timestamps_test=1
)

def test_generate_timestamps_from_dates():
    ts_generator = TimestampGenerator(test_cfg)
    
    timestamps= ts_generator.generate_timestamps_from_dates('2022-06-01', '2022-06-07', frequency='3-day')
    expected_timestamps = ['20220601T150000', '20220604T150000', '20220607T150000']
    assert timestamps == expected_timestamps, "Timestamps do not match expected values"
    
    timestamps = ts_generator.generate_timestamps_from_dates('2022-06-01', '2022-06-07', frequency='daily')
    expected_timestamps = [
        '20220601T150000', '20220602T150000', '20220603T150000',
        '20220604T150000', '20220605T150000', '20220606T150000',
        '20220607T150000'
    ]
    assert timestamps == expected_timestamps, "frequency=daily: timestamps do not match expected values"
    
    timestamps = ts_generator.generate_timestamps_from_dates('2022-06-01', '2022-06-02', frequency='hourly')
    expected_length = 25
    assert len(timestamps) == expected_length, "frequency=hourly: number of timestamps do not match expected value"
    
    
def test_update_timestamps():
    ts_generator = TimestampGenerator(test_cfg)
    
    key = '32TPT'
    values = ['20220601T150000', '20220604T150000']
    ts_generator.update_timestamps(key, values, update_type='main')
    assert '32TPT' in ts_generator.timestamp_dict, "Tile '32TPT' not found in main timestamp_dict"
    assert ts_generator.timestamp_dict[key] == values, "Timestamps for '32TPT' do not match expected values"
    assert ts_generator.timestamp_dict_supplementary[key] == [], "Supplementary timestamps for '32TPT' should be empty"
    
    key = '33UWP'
    values = ['20220602T150000']
    ts_generator.update_timestamps(key, values, update_type='main')
    assert '33UWP' in ts_generator.timestamp_dict, "Tile '33UWP' not found in main timestamp_dict"
    assert ts_generator.timestamp_dict[key] == values, "Timestamps for '33UWP' do not match expected values"
    assert ts_generator.timestamp_dict_supplementary[key] == [], "Supplementary timestamps for '33UWP' should be empty"
    assert ts_generator.timestamp_dict.keys() == {'32TPT', '33UWP'}, "Main timestamp_dict keys do not match expected values"
    
    key = '33TWN'
    values = ['20220603T150000', '20220606T150000']
    ts_generator.update_timestamps(key, values, update_type='supplementary')
    assert '33TWN' in ts_generator.timestamp_dict_supplementary, "Tile '33TWN' not found in supplementary timestamp_dict"
    assert ts_generator.timestamp_dict_supplementary[key] == values, "Supplementary timestamps for '33TWN' do not match expected values"
    
    
def test_flatten_to_tuples():    
    data_dict = {
        '32TPT': ['20220601T150000', '20220604T150000'],
        '33UWP': ['20220602T150000']
    }
    expected_output = [('32TPT', '20220601T150000'), ('32TPT', '20220604T150000'), ('33UWP', '20220602T150000')]
    output = TimestampGenerator.flatten_to_tuples(data_dict)
    assert output == expected_output, "Flattened tuples do not match expected output"
    
    
def test_list_to_txt(tmp_path):
    tmp_path = tmp_path / "timestamps_output.txt"
    data = [('32TPT', '20220601T150000'), ('32TPT', '20220604T150000'), ('33UWP', '20220602T150000')]
    TimestampGenerator.list_to_txt(data, tmp_path.parent, tmp_path.name)
    assert tmp_path.exists(), "Output text file was not created"
    with open(tmp_path, 'r') as file:
        lines = file.read().splitlines()
    expected_lines = ['32TPT, 20220601T150000', '32TPT, 20220604T150000', '33UWP, 20220602T150000']
    assert lines == expected_lines, "Content of the output text file does not match expected lines"
    
    
def test_load_from_txt(tmp_path):
    test_cfg.net_input_directory = tmp_path
    ts_generator = TimestampGenerator(test_cfg).prepare_directories()
    ts_generator.net_input_directory_txt = tmp_path
    
    # generate sample text files
    lines = ['32TPT, 20220601T150000', '32TPT, 20220604T150000', '33UWP, 20220602T150000']
    filepath = ts_generator.net_input_directory_txt / "main_file_1.txt"
    with open(filepath, 'w') as file:
        for line in lines:
            file.write(line + '\n')
    filepath = ts_generator.net_input_directory_txt / "supplementary_file_1.txt"
    with open(filepath, 'w') as file:
        for line in lines:
            file.write(line + '\n')
            
    # test load method
    data = ts_generator.load_tuples_from_txt(type='main', reduce_samples=None)
    assert type(data) == list, "Loaded data is not a list"
    assert type(data[0]) == tuple, "Loaded data items are not tuples"
    expected_data = [('32TPT', '20220601T150000'), ('32TPT', '20220604T150000'), ('33UWP', '20220602T150000')]
    assert data == expected_data, "Loaded data from text file does not match expected data"
    
    data = ts_generator.load_tuples_from_txt(type='main', reduce_samples=1)
    assert len(data) == 1, "Loaded data length does not match reduce_samples value"
    
    data = ts_generator.load_tuples_from_txt(type='supplementary', reduce_samples=None)
    assert type(data) == list, "Loaded data is not a list"
    assert type(data[0]) == tuple, "Loaded data items are not tuples"
    expected_data = [('32TPT', '20220601T150000'), ('32TPT', '20220604T150000'), ('33UWP', '20220602T150000')]
    assert data == expected_data, "Loaded data from text file does not match expected data"
    
    # generate another sample text file, test duplicate handling
    filepath = ts_generator.net_input_directory_txt / "main_file_2.txt"
    lines = ['32TPT_newer, 20220605T150000', '33UWP_newer, 20220602T150000']
    with open(filepath, 'w') as file:
        for line in lines:
            file.write(line + '\n')
    data = ts_generator.load_tuples_from_txt(type='main', reduce_samples=None)
    expected_key = '32TPT_newer'
    assert any(item[0] == expected_key for item in data), "Newer file not prioritized correctly"
    
    
    
def test_find_ecostress_images(tmp_path):
    eco_cfg = EcostressConfig(
        data_directory=tmp_path,
        tmp_directory=tmp_path,
        ancillary_directory=tmp_path,
        ecostress_qc_directory=tmp_path,
        ecostress_nodata=255,
        s2_nodata=255,
        pct_valid_threshold=0,
        cloud_cover_max=100,
        inca_nodata=255,
        inca_datasets=[],
        bounding_box=[0, 0, 100, 100],
        ancillary_layers=None
    )
    eco_processor = EcostressManager(eco_cfg).prepare_directories()
    
    # generate a list of dummy Ecostress images in masked_dir
    timestamps = [
        '20220601T110000', '20220604T120000', '20220607T130000', '20220610T140000', '20220801T150000',
        '20230601T110000', '20230604T120000', '20230607T130000', '20230610T140000', '20230801T150000',
        ]
    for ts in timestamps:
        filename = f"ECOSTRESS_LST_UTM_001_masked_{ts}_v1.tif"
        filepath = Path(eco_processor.masked_dir) / filename
        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=10,
            width=10,
            count=1,
            dtype=rasterio.uint8,
            crs='EPSG:32632',
            transform=rasterio.transform.from_origin(0, 10, 1, 1)
        ) as dst:
            dst.write(np.random.randint(0, 255, (10, 10)).astype(rasterio.uint8), 1)
            
    ts_generator = TimestampGenerator(test_cfg)
    print(os.listdir(eco_processor.masked_dir))
    ts_generator.find_ecostress_images(eco_processor, search_patterns=['LST_UTM'])
    expected_output = ['20220601T110000', '20220604T120000']
    assert set(ts_generator.eco_timestamps) == set(expected_output), "Found ECOSTRESS timestamps do not match expected values"
    
    ts_generator.year = None
    ts_generator.month_range = [6,8]
    ts_generator.hour_range = [14, 15]
    ts_generator.find_ecostress_images(eco_processor, search_patterns=['LST_UTM'])
    expected_output = ['20220610T140000', '20220801T150000', '20230610T140000', '20230801T150000']
    assert set(ts_generator.eco_timestamps) == set(expected_output), "Found ECOSTRESS timestamps do not match expected values"
    # reset ts_generator parameters
    ts_generator.year = 2022
    ts_generator.month_range = [6,7]
    ts_generator.hour_range = [10, 12]
    
def test_generate_timestamps_from_ecostress(tmp_path):
    eco_cfg = EcostressConfig(
        data_directory=tmp_path,
        tmp_directory=tmp_path,
        ancillary_directory=tmp_path,
        ecostress_qc_directory=tmp_path,
        ecostress_nodata=255,
        s2_nodata=255,
        pct_valid_threshold=0,
        cloud_cover_max=100,
        inca_nodata=255,
        inca_datasets=[],
        bounding_box=[0, 0, 100, 100],
        ancillary_layers=None
    )
    eco_processor = EcostressManager(eco_cfg).prepare_directories()
    
    # generate a list of dummy tiled Ecostress images in reproj_dir
    tile = '32TPT'
    timestamps = ['20220601T110000', '20230801T150000']
    for ts in timestamps:
        filename = f"ECOSTRESS_LST_UTM_001_masked_{ts}_v1.tif"
        filepath = Path(eco_processor.masked_dir) / filename
        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=10,
            width=10,
            count=1,
            dtype=rasterio.uint8,
            crs='EPSG:32632',
            transform=rasterio.transform.from_origin(0, 10, 1, 1)
        ) as dst:
            dst.write(np.random.randint(0, 255, (10, 10)).astype(rasterio.uint8), 1)
    for ts in timestamps:
        filename = f"ECOSTRESS_LST_UTM_001_reprojected_{ts}_v1_32TPT.tif"
        tile_dir = Path(eco_processor.reproj_dir) / tile
        tile_dir.mkdir(parents=True, exist_ok=True)
        filepath = tile_dir / filename
        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=10,
            width=10,
            count=1,
            dtype=rasterio.uint8,
            crs='EPSG:32632',
            transform=rasterio.transform.from_origin(0, 10, 1, 1)
        ) as dst:
            dst.write(np.random.randint(0, 255, (10, 10)).astype(rasterio.uint8), 1)
            
    ts_generator = TimestampGenerator(test_cfg)
    ts_generator.find_ecostress_images(eco_processor, search_patterns=['LST_UTM'])
    data = ts_generator.generate_timestamps_from_ecostress(eco_processor, tile)
    assert len(data) == 1, "Number of generated timestamps does not match expected value"
    assert len(data[0]) == 3, "Generated data tuple does not have expected length"
    assert data[0][0] == '20220601T110000', "Tile in generated data does not match expected value"
    
    
def test_train_test_split_ecostress():
    ts_generator = TimestampGenerator(test_cfg)
    timestamps = [
        ('20220601T110000', 'path1', 10),
        ('20220604T120000', 'path2', 20),
        ('20220607T130000', 'path3', 30),
        ('20220610T140000', 'path4', 15),
        ('20220613T150000', 'path5', 25),
        ('20220616T150000', 'path6', 35)
    ]
    train, test = ts_generator.train_test_split_ecostress(timestamps)
    assert len(train) == 5, "Number of training timestamps does not match expected value"
    assert len(test) == 1, "Number of testing timestamps does not match expected value"
    
    train, test = ts_generator.train_test_split_ecostress(timestamps, seed=42)    
    expected_out = ['20220601T110000', '20220610T140000', '20220604T120000', '20220613T150000', '20220607T130000']
    assert set(train) == set(expected_out), "Training timestamps do not match expected values"