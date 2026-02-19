import pytest
import sys
import os
from pathlib import Path
import rasterio
import configparser

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lib.data.s2 import S2Config, S2Manager

secrets_path = Path("src/config/secrets.ini")
if not secrets_path.exists():
    print("Warning: secrets.ini file does not exist. Tests that require credentials will be skipped!")
        

class TestS2Manager:
    def test_s2_config(self):
        s2_cfg = S2Config.from_yaml("tests/data/config_test.yml")
        assert hasattr(s2_cfg, 'data_directory')
        if not secrets_path.exists():
            return
        secrets_cfg = configparser.ConfigParser()
        secrets_cfg.read(secrets_path)
        s2_cfg.update_from_dict({'secret' : dict(secrets_cfg.items('earthdata'))}) 
        assert hasattr(s2_cfg, 'secret')

    def test_prepare_directories(self, tmp_path):
        s2_cfg = S2Config.from_yaml("tests/data/config_test.yml")
        # change to tmp_path for testing
        s2_cfg.data_directory = tmp_path
        s2_processor = S2Manager(s2_cfg)
        s2_processor.prepare_directories() 
        assert (tmp_path / 'S2').exists(), "S2 directory was not created"

    def test_generate_s3_cfg(self, tmp_path):
        s2_cfg = S2Config.from_yaml("tests/data/config_test.yml")
        s2_cfg.data_directory = tmp_path
        s2_processor = S2Manager(s2_cfg).prepare_directories() 
        s2_processor._generate_s3_cfg(secrets_path)
        assert s2_processor.s3cfg_path.exists(), "S3 config file was not created"
        with s2_processor.s3cfg_path.open() as f:
            content = f.read()
        assert 'access_key' in content, "Key access_key not found in .s3cfg"

    def test_search_s3(self, tmp_path):
        s2_cfg = S2Config.from_yaml("tests/data/config_test.yml")
        s2_cfg.data_directory = tmp_path
        s2_processor = S2Manager(s2_cfg).prepare_directories() 
        s2_processor.search_s3(
            tile='32TPT',
            start_date='2025-08-01',
            end_date='2025-08-02',
            secrets_path = secrets_path
        )
        assert s2_processor.download_links, "No download links were founf"
        assert len(s2_processor.download_links) ==1, f"Unexpected number of download links: {len(s2_processor.download_links)}, expected 1"

    def test_download_images(self, tmp_path):
        s2_cfg = S2Config.from_yaml("tests/data/config_test.yml")
        s2_cfg.data_directory = tmp_path
        s2_processor = S2Manager(s2_cfg).prepare_directories() 
        s2_processor._generate_s3_cfg(secrets_path)
        s3cfg_path = s2_processor.s3cfg_path
        s2_processor.download_links = [' s3://eodata/Sentinel-2/MSI/L2A/2025/08/02/S2A_MSIL2A_20250802T101041_N0511_R022_T32TPT_20250802T140417.SAFE/']
        s2_processor.download()
        downloaded_files = list(Path(s2_processor.download_dir).glob('*.jp2'))
        assert downloaded_files, "No files were downloaded"
        assert len(downloaded_files) == 3, f"Expected 3 files, but found {len(downloaded_files)}"
        assert not s3cfg_path.exists(), "tmp config file .s3cfg was not deleted after download"



def test_s2_resample(tmp_path):
    """
    Test the resample method of S2Manager with a small test TIF.
    """
    cfg = S2Config(
        data_directory=tmp_path,
        ecostress_nodata=None,
        s2_nodata=None,
        pct_valid_threshold=None,
        cloud_cover_max=None,
        inca_nodata=None,
        inca_datasets=None,
        bounding_box=None
        )
    # Path to test TIF
    test_tif = "tests/data/s2_test_10m.tif"
    assert Path(test_tif).exists(), "Test file does not exist"
    
    # Build a minimal S2Config for testing
    cfg = S2Config.from_yaml("tests/data/config_test.yml") #.update_from_dict({"tiles": ["32TPT"], "target_res": 70})
    s2_manager = S2Manager(cfg)
    
    # Override directories for testing
    s2_manager.download_dir = "tests/data/"
    s2_manager.resampled_dir = tmp_path / "resampled"
    s2_manager.resampled_dir.mkdir(parents=True, exist_ok=True)

    # Call the method under test
    tile = "s2"
    resolution = 70
    s2_manager.resample(tile=tile, resolution=resolution, timestamp='', extension='.tif')
    
    # Check that output file exists
    output_file = s2_manager.resampled_dir / tile / os.listdir(s2_manager.resampled_dir / tile)[0]
    assert output_file.exists(), f"Resampled file was not created: {output_file}"
    
    # Check basic properties
    with rasterio.open(output_file) as src:
        assert src.width > 0 and src.height > 0, "Resampled file has invalid dimensions"
        assert src.transform.a == resolution, "Pixel width does not match resolution"
        assert src.transform.e == -resolution, "Pixel height does not match resolution"
        
    # Check stats
        data = src.read(1)
        assert data.min() != data.max(), "Resampled data has no variation"
        assert data.mean() > 0, "Resampled data mean is not greater than zero"
        assert data.min() == 1262, f"Unexpected min value: {data.min()}, expected 1262"
        assert data.max() == 4199, f"Unexpected max value: {data.max()}, expected 4199"
        assert 2018<data.mean()<2019, f"Unexpected mean value: {data.mean()}, expected around 2018.92"
        assert 448<data.std()<449, f"Unexpected std value: {data.std()}, expected around 448.10"
        assert (data != src.nodata).sum() / data.size > 0.99, f"Unexpected valid data percentage: {(data != src.nodata).sum() / data.size}, expected 1"