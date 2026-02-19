import pytest
import sys
import os
import rasterio
import numpy as np
from pathlib import Path
import configparser

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lib.data.ecostress import EcostressConfig, EcostressManager

secrets_path = Path("src/config/secrets.ini")
if not secrets_path.exists():
    print("Warning: secrets.ini file does not exist. Tests that require credentials will be skipped!")
        

class TestEcostressManager:
    def test_ecostress_config(self):
        eco_cfg = EcostressConfig.from_yaml("tests/data/config_test.yml")
        assert hasattr(eco_cfg, 'data_directory')

        if not secrets_path.exists():
            return
        secrets_cfg = configparser.ConfigParser()
        secrets_cfg.read(secrets_path)
        eco_cfg.update_from_dict({'secret' : dict(secrets_cfg.items('earthdata'))}) 
        assert hasattr(eco_cfg, 'secret')

    def test_prepare_directories(self, tmp_path):
        eco_cfg = EcostressConfig.from_yaml("tests/data/config_test.yml")
        # change to tmp_path for testing
        eco_cfg.data_directory = tmp_path
        eco_processor = EcostressManager(eco_cfg)
        eco_processor.prepare_directories() 
        assert (tmp_path / 'ECOSTRESS').exists(), "ECOSTRESS directory was not created"

  

def test_ecostress_reproject(tmp_path):
    """
    Test the reproject method of EcostressManager with a small test TIF.
    """
    cfg = EcostressConfig(
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
    test_tif = "tests/data/ecostress_test.tif"
    assert Path(test_tif).exists(), "Test file does not exist"
    
    # Build a minimal EcostressConfig for testing
    cfg = EcostressConfig.from_yaml("tests/data/config_test.yml")
    ecostress_manager = EcostressManager(cfg)
    
    # Override directories for testing
    ecostress_manager.download_dir = "tests/data/"
    ecostress_manager.reproj_dir = tmp_path / "resampled"
    ecostress_manager.reproj_dir.mkdir(parents=True, exist_ok=True)

    # Call the method under test
    tile = "eco"
    template_path = "tests/data/s2_test_10m.tif"
    ecostress_manager.reproject(tile, images_to_reproject=[test_tif], template_path=template_path)
    
    # Check that output file exists
    generated_files = os.listdir(ecostress_manager.reproj_dir / tile)
    assert len(generated_files) > 0, "No files were created in the reprojection directory"
    filename = [f for f in generated_files if f.endswith('.tif') and '.aux' not in f][0]
    output_file = ecostress_manager.reproj_dir / tile / filename
    assert output_file.exists(), f"Resampled file was not created: {output_file}"
    
    # Check basic properties
    with rasterio.open(output_file) as src, rasterio.open(template_path) as template_src:
        assert src.width > 0 and src.height > 0, "Resampled file has invalid dimensions"
        assert src.transform.a == template_src.transform.a, "Pixel width does not match resolution"
        assert src.transform.e == template_src.transform.e, "Pixel height does not match resolution"
        
    # Check stats
        data = src.read(1)
        data[np.where(data == src.nodata)] = np.nan
        assert data.min() != data.max(), "Resampled data has no variation"
        assert 25<np.nanmin(data)<26, f"Unexpected min value: {data.min()}, expected around 25.0"
        assert 33<np.nanmax(data)<34, f"Unexpected max value: {data.max()}, expected around 33.59"
        assert 29<np.nanmean(data)<30, f"Unexpected mean value: {data.mean()}, expected around 29.41"
        assert 1.7<np.nanstd(data)<1.8, f"Unexpected std value: {data.std()}, expected around 1.72"
        assert (data != src.nodata).sum() / data.size > 0.98, f"Unexpected valid data percentage: {(data != src.nodata).sum() / data.size}, expected 0.99"