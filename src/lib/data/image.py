import os
import rasterio
import subprocess
import numpy as np
from pathlib import Path
from osgeo import ogr
from rasterio.windows import Window


class ImageProcessor:
    """Class for processing images."""
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

    @staticmethod
    def run_gdal(cmd: list, capture_output: bool = False, logger: object = None):
        """Run a GDAL command using subprocess."""
        result = subprocess.run(cmd, capture_output=capture_output)
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8') if capture_output else 'Unknown error'
            if logger:
                logger.error(f"GDAL command failed: {error_msg}")
            raise RuntimeError(f"GDAL command failed: {error_msg}")

    def compress_image(self):
        """Compress image and save it to the same location (deletes original)."""
        filename = self.filepath.name
        directory = self.filepath.parent
        tmp = directory / f'temp_{filename}'
        
        with rasterio.open(self.filepath) as src:
            meta = src.meta.copy()
            meta.update(compress='lzw')  
            with rasterio.open(tmp, 'w', **meta) as dst:
                for i in range(1, src.count + 1):
                    dst.write(src.read(i), i)
        os.remove(self.filepath)
        os.rename(tmp, self.filepath)
        
    def rasterize_shapefile(self, output_path, target_res: float, capture_output: bool = True):
        """
        Rasterize input_image (.shp file) to the specified resolution target_res.
        The output raster has two values: 1=inside shape, 0=outside shape
        capture_output: if set to True, the output od the subprocess comand will NOT be printed
        """
        
        if not str(self.filepath).endswith('.shp'):
            raise ValueError(f"Input must be a shapefile (.shp): {self.filepath}")
        
        source_ds = ogr.Open(str(self.filepath))
        source_layer = source_ds.GetLayer()
        x_min, x_max, y_min, y_max = source_layer.GetExtent()
        source_srs = source_layer.GetSpatialRef()
        source_ds = None
        
        x_res = int((x_max - x_min) / target_res)
        y_res = int((y_max - y_min) / target_res)      
        
        command = [
            'gdal_rasterize',
            '-te', str(x_min), str(y_min), str(x_max), str(y_max),
            '-ts', str(x_res), str(y_res),
            '-ot', 'Byte',
            '-of', 'GTiff',
            '-co', 'COMPRESS=LZW',
            '-init', '0',
            '-burn', '1',
            self.filepath, output_path
        ]
        ImageProcessor.run_gdal(command, capture_output=capture_output)
        
    def reproject_by_template(self, template_file, output_path, 
                              target_res=None,
                              resampling_method='nearest',
                              use_src_nodata=False, 
                              target_no_data=None, 
                              additional_arguments = '', 
                              capture_output=True
                              ):
        """
        Reproject and resample image to the template image.
        - Input: 
            template_file: image based on which to do the reprojection
            output_path: path to the output image
            target_res: target resolution of the output image. If None, the resolution of the template image is used
            use_src_nodata: if True, -srcnodata src.nodata is passed to gdalwarp
            target_no_data: if not none, -dstnodata target_no_data is passed to gdalwarp
            additional_arguments: more arguments to pass to GDAL. It is expected as a single string
            capture_output: if set to True, the output od the subprocess comand will NOT be printed
        """
        if not (str(self.filepath).endswith('.tif') or str(self.filepath).endswith('.jp2')):
            raise ValueError(f"Input must be a GeoTIFF file or a JPEG2000 file: {self.filepath}")
        
        # obtain bounds from the template_file
        with rasterio.open(template_file) as src:
            target_projection = str(src.crs)
            bounds = src.bounds # left bottom right top
            x_res, y_res = src.res
            src_no_data = src.nodata
            
        if target_res is None:
            target_res = min(x_res, y_res)
        
        ulx = min(bounds[0], bounds[2])  # ulx / xmin
        lrx = max(bounds[0], bounds[2])  # lrx / xmax
        uly = max(bounds[1], bounds[3])  # uly / ymax
        lry = min(bounds[1], bounds[3])  # lry / ymin    
        
        # parse additional_arguments string
        if len(additional_arguments) > 0:
            additional_arguments_parsed = additional_arguments.split(' ')
        else:
            additional_arguments_parsed = []
            
        if use_src_nodata == True:
            additional_arguments_parsed += ['-srcnodata', str(src_no_data)]
            
        if target_no_data is not None:
            additional_arguments_parsed += ['-dstnodata', str(target_no_data)]
        
        # GDAL reprojection command
        cmd = ['gdalwarp'] + additional_arguments_parsed + [
                            # '-srcnodata', str(src_no_data), '-dstnodata', str(target_no_data),
                            '-tr', str(target_res), str(target_res), 
                            '-te', str(ulx), str(lry), str(lrx), str(uly),
                            '-t_srs', target_projection, 
                            '-r', resampling_method, 
                            '-of','GTiff', 
                            '-co', 'compress=LZW', 
                            self.filepath, output_path
                            ]    
        ImageProcessor.run_gdal(cmd, capture_output=capture_output)
        
    @staticmethod
    def set_nan_values(arr: np.ndarray, nv: float) -> np.ndarray:
        try:
            arr = arr.astype(float)
            return np.nan_to_num(arr, nan=nv)
        except:
            return np.nan_to_num(arr, nan=0.0)  # TBD!!!! quick fix for aspc/eudem nodata  

    def path_to_array(self, window=None):
        """Read a raster file into a numpy array."""
        if not str(self.filepath).endswith('.tif'):
            raise ValueError(f"Input must be a GeoTIFF file: {self.filepath}")
        with rasterio.open(self.filepath) as raster:
            if window is None:
                # read the whole raster
                arr = raster.read(1)
            else:
                # read a specific window of the raster
                col_off, row_off, width, height = window
                arr = raster.read(1, window=Window(col_off, row_off, width, height))
            nodata = raster.nodata
            resolution = raster.res[0]
        # arr[np.where(arr == nodata)] = np.nan
        # replace nan by nodata value
        arr = ImageProcessor.set_nan_values(arr, nodata)
        metadata = {'nodata':nodata, 'resolution':resolution}
        return arr, metadata

    def compute_percent_nodata(self):
        """Compute the percentage of no-data pixels in the image."""
        arr, metadata = self.path_to_array(window=None)
        arr = ImageProcessor.set_nan_values(arr, metadata['nodata'])
        msk = np.where(arr == metadata['nodata'])
        num_invalid_px = len(msk[0])
        num_total_px = arr.size
        return int(round(100*num_invalid_px/num_total_px,0))
    
    @staticmethod
    def get_timestamp_from_filename(filepath: str, data_source: str) -> str:
        """
        Returns a timestamp string YYYYMMDDThhmmss from a provided path.
        Supported data sources: 'S2', 'ECOSTRESS', 'INCA'.
        """
        basename = os.path.basename(filepath)
        if data_source == 'S2':
            timestamp = basename.split('_')[1]
        if data_source == 'ECOSTRESS':
            timestamp = basename.split('_')[6]
        if data_source == 'INCA':
            timestamp = basename.split('_')[1]
        return timestamp