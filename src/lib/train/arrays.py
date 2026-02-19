import sys 
import os
import yaml
import numpy as np
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from keras.utils import to_categorical

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.utils import list_filepaths
from lib.train.patches import cut_into_patches
from lib.base import BaseSetup
from lib.train.train_config import BaseTrainConfig
from lib.data.ecostress import EcostressManager
from lib.data.s2 import S2Manager
from lib.data.inca import IncaManager
from lib.data.image import ImageProcessor


# from lib.train.train_config import BaseTrainConfig
# from lib.data.image import ImageProcessor

# from lib.train.patches import unblockshaped, unblockshaped_all_pixels

@dataclass
class NetPreprocessorConfig(BaseTrainConfig):
    """Dataclass for net preprocessor configuration parameters."""
    save_npy: Optional[bool] = True
    target_dataset: Optional[str] = 'ECOSTRESS'  # name of the ground truth dataset
    


class NetPreprocessor(BaseSetup):
    """Class for preparing images for CNN training/validation and reading them to numpy arrays."""
    def __init__(self, cfg: NetPreprocessorConfig):
        super().__init__(cfg)
        # define self.numericaL_datasets attibute
        self.define_numerical_datasets()
        # define a 'master key' (will be used for image shape)
        if self.numerical_datasets:
            self.master_key = self.numerical_datasets[0]
        elif self.categorical_datasets:
            self.master_key = self.categorical_datasets[0]
        else:
            raise ValueError("No input datasets defined in numerical_datasets or categorical_datasets. Please check the input.")
        # dircetory for saving numpy arrays
        self.net_input_dir_arr = os.path.join(self.net_input_directory, 'input_arrays')
        
    def prepare_directories(self):
        self.ensure_dirs_exist(self.net_input_directory, self.net_input_dir_arr)
        return self
           
    def define_numerical_datasets(self):
        self.numerical_datasets = self.s2_datasets + self.inca_datasets

    def locate_input_data(self, 
        tile, 
        timestamp, 
        datasets,
        target_manager=None,
        s2_manager=None,
        inca_manager=None,
        ancillary_manager=None
    ):
        """
        Obtain file paths of all input layers for a given tile and timestamp.
        Managers are passed as parameters to avoid internal instantiation.
        Output:
        - dictionary `{dataset_name: file_path}`
        - generates a modified self.numerical_datasets_modified if self.inca_range > 0. \n
            `('T2M' --> 'T2M_-n', 'T2M_-n+1', ..., 'T2M' )`
        """
        def _add_to_output(d, path):
            """Add dataset path to output dictionary, handling INCA history if needed."""
            updated_datasets = []
            if d not in self.inca_datasets:      
                output[d] = path  
            elif isinstance(path, list) and len(path) == 1:
                output[d] = path[0]
            else:
                hr_range = len(path)  # // 2
                for i, p in enumerate(path):
                    updated_d = f"{d}_{-hr_range+1+i}"  # definew new dataset name
                    output[updated_d] = p  
                    updated_datasets.append(updated_d)
                    
            # return dictionary with new dataset names in case of INCA history range (else just return the original dataset name)
            return {d: updated_datasets} if updated_datasets else d
        
        resolution = self.resolution  # target resolution for all datasets
        output = {}
        for d in datasets:            
            if d==self.target_dataset:
                path = target_manager.locate_image(tile, timestamp)
                _add_to_output(d, path)  
            s2_datasets = self.s2_datasets # + self.scl_dataset
            if d in s2_datasets:
                print_info = (d == s2_datasets[0])  # print info only for the first S2 dataset
                path = s2_manager.locate_image(tile, d, resolution, timestamp, print_info=print_info)
                _add_to_output(d, path) 
            if d in self.inca_datasets:
                print_info = (d == self.inca_datasets[0])  # print info only for the first INCA dataset
                s2_path = s2_manager.define_template_image(tile, resolution)
                path = inca_manager.locate_image(tile, d, resolution, s2_path, timestamp, print_info=print_info)
                d_dict = _add_to_output(d, path)  # outputs a dictionary of new dataset names in case INCA history is used, d otherwise
                # generate a modified numerical dataset if inca_range > 0 for non-cumulative datasets
                if self.inca_range > 0 and d not in self.inca_cumulative_datasets and isinstance(d_dict, dict):
                    tmp = self.numerical_datasets.copy()
                    idx = tmp.index(d)
                    self.numerical_datasets_modified = tmp[:idx] + list(d_dict.values())[0] + tmp[idx+1:]                
        self.logger.debug(f"Located input data for tile {tile} and timestamp {timestamp}:")
        for key, value in output.items():
            self.logger.debug(f"    {key} : {value}")
        self.input_data = output
                
                
    def read_images(self, window=None): 
        """
        Read images into numpy arrays. Images are provided as a dictionary {key: path} in self.input_data.
        Size of target array is adjusted to the size of feature arrays if needed.
        INCA datasets are summed if they are in self.inca_cumulative_datasets list and have the same key.
        """  
        def _read_target(target, window, feature_res, feature_shape):
            path = target[self.target_dataset]
            if path is not None:
                arr, metadata = ImageProcessor(path).path_to_array(window=None)
                res = metadata['resolution']
                nv = metadata['nodata']
                if res != feature_res:
                    arr, res = _adjust_array_size(arr, res, feature_res, window=window) 
            else:
                arr = np.array([])  # if target is not available, return empty array
                nv = None
            if arr.size != 0 and arr.shape != feature_shape:
                raise ValueError(f"Target array shape {arr.shape} does not match feature array shape {feature_shape}. Please check the data.")
            return arr, nv
        
        def _adjust_array_size(arr, res, requested_res, window):
            """Adjust array size to requested resolution by repeating values."""
            if requested_res>res:
                self.logger.error(f"    Requested resolution ({int(requested_res)} m) is larger than resolution {res} m (not supported)")
                raise ValueError(f"    Requested resolution ({int(requested_res)} m) is larger than resolution")
            if res%requested_res != 0:
                self.logger.error(f"    Requested resolution ({int(requested_res)} m) is not a multiple of resolution {res} m (not supported)")
                raise ValueError(f"    Requested resolution ({int(requested_res)} m) is not a multiple of resolution {res}")
            else:
                factor = int(res/requested_res)  
                
            if factor > 1:               
                self.logger.warning(f"    Requested resolution ({int(requested_res)} m) is {factor} times smaller than resolution. Array will be repeated.")
                arr = np.repeat(np.repeat(arr, factor, axis=1), factor, axis=0)  # repeat values to adjust size
            
                # if larger, reduce arr to size of S2 (not needed)
                # arr = arr[:10980,:10980] 
                
                # if window provided, keep only corresponding part of the array
                if window is not None:
                    arr = arr[window[1]:window[1]+window[3],window[0]:window[0]+window[2]] 
            return arr, requested_res
        
        if getattr(self, 'input_data', None) is None:
            raise ValueError("Input data is not defined. Please run 'locate_input_data' first.")
        
        self.arrays = {}
        self.nodata = {}
        resolution = {}
        shape = {}
        
        input_images = self.input_data.copy()
        
        # separate target dataset from input features
        if self.target_dataset in input_images.keys():
            target = {self.target_dataset: input_images.pop(self.target_dataset)}
        else:
            target = {}
        
        previous_key = None  # store previous key for later use
        for key, path in input_images.items():
            key = key.split('_')[0] if key.split('_')[0] in self.inca_cumulative_datasets else key  # remove history suffix if present            
            arr, metadata = ImageProcessor(path).path_to_array(window=window) # hardcoded
            if key in self.inca_cumulative_datasets and key == previous_key:
                # self.logger.warning(f"    Same dataset detected: {key}, {previous_key}. Summing arrays.")
                self.arrays[previous_key] += arr  # sum INCA datasets into a cumulative array
            else:
                self.arrays[key] = arr  # save array separately for each key
                self.nodata[key] = metadata['nodata']
                resolution[key] = metadata['resolution']
                shape[key] = arr.shape
                previous_key = key
            
        res_list = list(set(resolution.values()))
        if len(res_list) > 1:
            raise ValueError(f"Multiple resolutions found in input datasets: {resolution}. Please check the data.") 
        else:
            feature_res = res_list[0]   
            
        shape_list = list(set(shape.values()))
        if len(shape_list) > 1:
            raise ValueError(f"Multiple shapes found in input datasets: {shape}. Please check the data.") 
        else:
            self.image_shape = shape_list[0]
            self.num_datasets = list(self.arrays.keys())
               
        # handle target dataset     
        if target:
            arr, nv = _read_target(target, window, feature_res, self.image_shape)
            self.arrays[self.target_dataset] = arr
            self.nodata[self.target_dataset] = nv
            self.original_target_array = arr  # keep original target array for later plotting
            

    def generate_patches(self, patch_size=5, overlap_pixels=False):
        """Generate patches of size patch_size x patch_size from input arrays and filter them based on no_data and SCL values."""
        self.patched_arrays = {}
        for key, arr in self.arrays.items():
            # skip target array if empty
            if key==self.target_dataset and len(arr)==0:
                self.patched_arrays[key] = np.array([])  
                continue
            arr_patch, first_pixel, arr_shape = cut_into_patches(
                arr, 
                patch_size=patch_size, 
                padd_value=self.nodata[key], 
                overlap_pixels=overlap_pixels
            )
            self.patched_arrays[key] = arr_patch 
            
        # obtain arr size and patch count from master key; same for all arrays
        self.n_patches_raw = self.patched_arrays[self.master_key].shape[0]
        self.logger.debug(f'    Number of patches: {self.n_patches_raw}')
        self.patched_shape = self.patched_arrays[self.master_key].shape
        self.logger.debug(f'    Patch shape: {self.patched_shape}')
        self.first_pixel = first_pixel  # first pixel coordinates in the original image (top-left corner of the first patch)
        self.shape_before_patching = arr_shape  # shape of the original image before patching 
    
    
    def filter_patches(self, features_to_clean):
        """
        Filter patches based on a custom filter criterion function (_valid_data_patch/_clean_scl_patch)
        If remove_after is True, remove the feature from patched_arrays after filtering.
        """
        
        self.logger.debug(f"    Cleaning patches for features: {features_to_clean}")
        def _find_valid_indices(filter_by_feature, criterion, remove_after):
            valid_indices = []
            arr = self.patched_arrays[filter_by_feature]
            for i in range(arr.shape[0]):
                patch = arr[i, :, :]
                if criterion(patch):
                    valid_indices.append(i)
            if remove_after:
                self.patched_arrays.pop(filter_by_feature)
            return valid_indices
    
        def _valid_data_patch(patch, no_data):
            """Check if patch contains no_data value."""
            return no_data not in patch

        def _clean_scl_patch(patch):
            """
            Check if patch contains only SCL values 4,5,6.
            (4=vegetation, 5 =non-vegetated, 6=water) 
            https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/
            """
            return np.all(np.isin(patch, [4, 5, 6]))
            
        # identify indices of clean patches for each feature (no no_data/snow/clouds)
        valid_idx_per_feature = []
        for feature in features_to_clean:
            if feature == self.target_dataset and self.patched_arrays[feature].shape[0] == 0:
                self.logger.warning(f"    Target dataset {feature} has no patches. Skipping filtering for this feature.")
                continue
            valid_idx_per_feature.append(
                _find_valid_indices(
                    filter_by_feature=feature, 
                    criterion=_clean_scl_patch if feature=='SCL' else lambda patch: _valid_data_patch(patch, self.nodata[feature]), 
                    remove_after=True if feature=='SCL' else False
                )
            )
            
        # obtain a list of valid indices (intersection of all lists in tmp_list)
        valid_idx_set = [set(sublist) for sublist in valid_idx_per_feature] # transform into sets
        common_elements = set.intersection(*valid_idx_set)                  # compute intersection
        self.valid_indices = sorted(common_elements)                             # transform to sorted list

        # update all arrays
        for key, values in self.patched_arrays.items():
            if len(self.patched_arrays[key]) == 0:
                continue
            self.patched_arrays[key] = (self.patched_arrays[key])[self.valid_indices,:,:]
            
        self.logger.debug(f'    Number of valid patches after filtering: {len(self.valid_indices)}')
                        
            
    def _one_hot_encoding(self):
        """
        N/A
        """
        # if not self.categorical_datasets:
        # self.logger.debug("    No categorical datasets to process. Returning empty array for X_cat.")
        return np.array([])
    
    
    def generate_numpy_arrays(self):
        """Generate numpy arrays from patched_arrays."""
        if getattr(self, 'patched_arrays', None) is None:
            raise ValueError("Patched arrays are not defined. Please run 'generate_patches' first.")

        # define target array
        self.Y_sample = self.patched_arrays[self.target_dataset]

        # define numerical numpy array (dimensions: patches, rows, cols, features)
        num_tmp = [self.patched_arrays[key] for key in self.numerical_datasets]
        X_num = np.stack(num_tmp, axis=-1)  # stack into a numpy array

        # define categorical numpy array if categorical datasets are provided (else, empty array)
        X_cat = self._one_hot_encoding()  # one-hot encode categorical features

        self.N_num = X_num.shape[3]  # number of numerical features
        self.N_cat = X_cat.shape[3] if X_cat.size > 0 else 0  # number of categorical features

        self.X_sample = np.concatenate([X_num, X_cat], axis=3) if self.N_cat > 0 else X_num  # concatenate numerical and categorical arrays
        self.n_patches = self.X_sample.shape[0]  # number of patches after filtering

        # Gracefully skip if either X_sample or Y_sample is empty
        if self.Y_sample.size == 0 or self.X_sample.size == 0:
            self.n_patches = 0
            return

        if self.Y.size != 0 and self.Y_sample.shape[0] != self.X_sample.shape[0]:
            raise ValueError(f"Target array Y shape {self.Y_sample.shape} does not match input array X shape {self.X_sample.shape}. Please check the data.")

        
    def update_input_datasets(self):
        """Prepare a list of input features based on the feature lists and the INCA history.
        Example: if self.inca_range=2 and self.inca_cumulative_datasets=['T2M'], ['B02', 'B03', 'T2M', 'RH2M'] will be 
        updated to ['B02', 'B03', 'T2M'(cummulative-doesn't expand), 'RH2M-2', 'RH2M-1', 'RH2M'(expands)].
        """
        features = self.config.s2_datasets + self.config.inca_datasets
        expanded = []
        for f in features:
            if f in self.inca_datasets and f not in self.inca_cumulative_datasets and self.inca_range > 0:
                for t in range(-self.inca_range, 1):
                    expanded.append(f"{f}_{t}")
            else:
                expanded.append(f)
        return expanded
    
      
    def prepare_sample(self, tile, timestamp, resolution, 
        target_manager,
        s2_manager,
        inca_manager,
        data_type='train',
        window=None, 
        patch_size=5, 
        overlap_pixels=False,
        skip_filter_for = ()
        ): 
        """
        Prepare self.X_sample and self.Y_sample for a single sample (tile, timestamp) with given resolution.
        X_sample: numerical and categorical features, shape [N_patches x patch_size x patch_size x N_features].
        Y_sample: target array, shape [N_patches x patch_size x patch_size]        
        """
        if data_type not in ['train', 'test']:
            raise ValueError(f"Invalid data_type: {data_type}. Supported types are 'train' (for training data) and 'test' (for unseen data).")
        if data_type == 'train' and skip_filter_for:
            raise ValueError("Skipping filtering for specific features is not supported for training data! Please remove 'skip_filter_for' argument.")
        
        # input datasets as provided in the config file
        input_datasets = [self.target_dataset,*self.numerical_datasets]
        if self.filter_by_SCL:
            input_datasets.append('SCL')
        
        self.locate_input_data(
            tile=tile,
            timestamp=timestamp,
            datasets=input_datasets,
            target_manager=target_manager,
            s2_manager=s2_manager,
            inca_manager=inca_manager,
        )  # modifies self.numerical_datasets if inca_range > 0
        self.logger.debug('    READING IMAGES')
        self.read_images(window=window)
        self.generate_patches(patch_size, overlap_pixels)
        self.logger.debug('    DONE')
        
        # modified list of input datasets based on INCA range and sum information
        input_datasets = [self.target_dataset,*self.update_input_datasets()] if data_type=='train' else self.update_input_datasets()
        if data_type=='test':  # different filtering for test set
            input_datasets.remove('aspc')  # TBD!!!! weird single pixel patches due to aspc -- check why this is suddenly here 
            for f in skip_filter_for:
                if f in input_datasets:
                    input_datasets.remove(f)
        
        self.filter_patches(features_to_clean=input_datasets)
        self.generate_numpy_arrays()
        
        
    def append_data_to_array(self, num_layers, data_type='train'):
        """
        Check if all layers were read for a sample and append to the output arrays self.X and self.Y.
        Return the number of layers read for this sample.
        """
        
        if data_type not in ['train', 'test']:
            raise ValueError(f"Invalid data_type: {data_type}. Supported types are 'train' (for training data) and 'test' (for unseen data).")
        if data_type == 'train' and self.Y_sample.shape[0] == 0:
            raise ValueError("Empty target array Y is not allowed for training data")
        
        if num_layers is None:
            num_layers = len(self.arrays)  # initialize num_layers with the number of layers read
        elif len(self.arrays) != num_layers:
            self.logger.error(f"    Sample {self.sample} does not have the same number of features as the rest ({len(self.arrays)} != {num_layers}) -- skipping sample.")         
        
        # append sample arrays to the output
        if self.Y.shape[0] == 0 and self.X.shape[0] == 0:  # if arrays are empty, initialize them
            self.Y = self.Y_sample
            self.X = self.X_sample
        else:
            try:
                self.Y = np.concatenate([self.Y,self.Y_sample])
                self.X = np.concatenate([self.X,self.X_sample])
            except Exception as e:
                self.logger.error(f"    Error appending sample to output arrays: {e} -- skipping")   
        return num_layers
        
   
    def prepare_input_arrays(self, samples, resolution,
        target_manager, s2_manager, inca_manager,
        window=None, 
        patch_size=5, 
        overlap_pixels=False,
        data_type='train',
        save_npy=False,
        skip_filter_for=()
    ):
        """
        Prepare numpy arrays from samples and concatenate them into single arrays X and Y.
        samples can be a single sample = (tile, timestamp) or a list of samples = [(tile, timestamp), ...] 
        If data_type is 'train', arrays and metadata will be saved to net input directory.
        """
        def _save_data(arr, output_path, batch_size=1024):
            """ Save numpy array to .dat file in batches. If the file already exists, append to it."""
            if not output_path.endswith('.dat'):
                raise ValueError("Output path must be a .dat file.")
            mode = 'ab' if  os.path.exists(output_path) else 'wb'
            with open(output_path, mode) as f:
                i = 0
                while i < arr.shape[0]:
                    batch = arr[i:i+batch_size]
                    batch.tofile(f)
                    i += batch_size
        
        if data_type not in ['train', 'test']:
            raise ValueError(f"Invalid data_type: {data_type}. Supported types are 'train' (for training data) and 'test' (for unseen data).")
        if data_type=='test' and isinstance(samples, list):
            raise TypeError("Test data can only be prepared for a single sample, not a list. Please select one sample from the list.")
        if data_type=='test' and save_npy:
            self.logger.warning("Saving .npy files is not supported for test data. Arrays will not be saved.")
        
        now = datetime.now().strftime('%Y%m%dT%H%M%S')
        self.Y = np.array([])  # target array
        self.X = np.array([])  # numerical features array
        samples = [samples] if isinstance(samples, tuple) else samples  
        
        self.overlap_pixels = overlap_pixels

        # initialize counters
        monitor_layer_num = None  # number of layers read (to be checked later)
        self.total_n_patches_raw = 0  # number of patches before filtering
        self.total_n_patches = 0
        
        for i,sample in enumerate(samples):
            self.logger.info(f'Processing sample {i+1}/{len(samples)}: {sample}')
            tile, timestamp = sample
            self.prepare_sample(tile, timestamp, resolution, target_manager, s2_manager, inca_manager, data_type, window, patch_size, overlap_pixels, skip_filter_for)
            # if i != len(samples)-1:  # TBD!!! remove inca_cummulative mess!
                # self.reset_numerical_datasets()  # reset numerical datasets for the next sample
            self.total_n_patches_raw += self.n_patches_raw
            self.total_n_patches += self.n_patches
            if self.n_patches == 0:
                self.logger.warning('    No valid patches remain after filtering -- skipping sample')
                continue
                        
            # load all samples to numpy arrays  TBD!!!! figure out how to handle array loading in case of large datasets
            monitor_layer_num = self.append_data_to_array(monitor_layer_num, data_type)
            
            # save training data to .dat files
            if data_type == 'train' and save_npy:
                self.logger.debug('    Saving data...')
                _save_data(self.X, output_path=os.path.join(self.net_input_dir_arr, f"{now}_X.dat"))
                _save_data(self.Y, output_path=os.path.join(self.net_input_dir_arr, f"{now}_Y.dat"))
    
        if data_type == 'test':
            self.logger.info(f'    Number of patches before and after cleaning: {self.total_n_patches_raw} --> {self.total_n_patches}')
            return self.X, self.Y
        
        # # skip saving metadata if save_npy==False
        # if not save_npy:
        #     return self.X, self.Y  
            
        # prepare metadata
        self.metadata = {
            'time_created': datetime.strptime(now,'%Y%m%dT%H%M%S').strftime('%Y-%m-%d %H:%M:%S'),     
            'type': data_type,
            'target_dataset': self.target_dataset,
            'target_nodata': self.nodata[self.target_dataset],
            'input_datasets': self.numerical_datasets,
            'nodata': {key: self.nodata[key] for key in self.numerical_datasets},
            'n_patches': self.total_n_patches,
            'N_num': self.N_num,
            'N_cat': self.N_cat,
            'resolution': resolution,
            'patch_size': patch_size,
            'overlap_pixels': overlap_pixels,
            'window': window if window is not None else None,
            'INCA_range': self.inca_range,
            'INCA_cumulative_datasets': self.inca_cumulative_datasets,
            'n_samples': len(samples),
            'samples': [', '.join(s) for s in samples],  # list of samples as strings                                        
        }
        self.logger.info(f"Numerical features: {self.numerical_datasets}")
        # self.logger.info(f"Categorical features: {self.categorical_datasets}")
        self.logger.info(f'Number of patches before cleaning: {self.total_n_patches_raw}')
        self.logger.info(f'Number of patches after  cleaning: {self.total_n_patches}')
        self.logger.info('Total array shape after cleaning [N_samples x patch_size x patch_size x N_features]: ')
        self.logger.info(f'mask:                                   {self.Y.shape}')
        self.logger.info(f'numerical features:                     {self.X.shape[:3]+(self.N_num,)}') 
        if self.N_cat > 0:       
            self.logger.info(f'categorical features (after OHE):       {self.X.shape[:3]+(self.N_cat,)}')
            
        # save metadata to a .yml file
        with open(os.path.join(self.net_input_dir_arr, f'{now}_metadata.yml'), 'w') as f:
            yaml.dump(self.metadata, f, default_flow_style=False, sort_keys=False)
        self.logger.info(f'Metadata saved to {self.net_input_dir_arr}')
        return self.X, self.Y  
    
    

    
    
    def load_input_arrays(self):
        """ Load input arrays from .dat files in the net_input_dir_arr directory."""
        
        def _load_data(input_path, arr_shape):
            """Load .np array from a .dat file."""
            with open(input_path, mode='rb') as f:
                arr = np.fromfile(f, dtype=np.float64)
                self.logger.error(f'array size: {arr.size}')
                self.logger.error(f'arr_shape to reshape into : {arr.shape}')
            # out = np.reshape(arr, (-1, *arr_shape[1:]))
            # return out
            return np.reshape(arr, arr_shape)
        
        tmp = list_filepaths(self.net_input_dir_arr, patterns_in=['_X.dat'], patterns_out=[])
        tmp = sorted(tmp, key =lambda x: os.path.basename(x), reverse=True)
        if not tmp:
            raise ValueError(f"No input arrays found in {self.net_input_dir_arr}. Please prepare input arrays first.")
        elif len(tmp) > 1:
            self.logger.warning(f"Multiple input arrays found in {self.net_input_dir_arr}. Using the most recent one: {os.path.basename(tmp[0])}")
        
        path_to_X = tmp[0]
        path_to_Y = path_to_X.replace('_X.dat', '_Y.dat')
        path_to_mtd = path_to_X.replace('_X.dat', '_metadata.yml')
            
        self.logger.info(f"Loading input arrays from {os.path.dirname(path_to_X)}...")
        with open(path_to_mtd, 'r') as file:
            mtd = yaml.safe_load(file)
        
        if mtd['patch_size'] != self.config['patch_size']:
            raise ValueError(f"Metadata patch_size ({mtd['patch_size']}) does not match the current configuration ({self.config['patch_size']}).")
        if mtd['target_nodata'] != self.target_nodata:
            raise ValueError(f"Metadata target_nodata ({mtd['target_nodata']}) does not match the current configuration ({self.target_nodata}).")
        
        input_datasets = self._update_input_datasets()
        
        if set(input_datasets) != set(mtd['input_datasets']):
            raise ValueError(f"Input datasets in saved array metadata do not match the current configuration \n - saved metadata:  {mtd['input_datasets']} \n - current config:  {input_datasets} (INCA history range={self.config['INCA_history_range']}).")
        
        input_shape_X = (mtd['n_patches'], mtd['patch_size'], mtd['patch_size'], mtd['N_num'] + mtd['N_cat'])
        self.X = _load_data(path_to_X, input_shape_X)
        
        input_shape_Y = (mtd['n_patches'], mtd['patch_size'], mtd['patch_size'])
        self.Y = _load_data(path_to_Y, input_shape_Y)
        
        self.metadata = mtd
        self.logger.info(f"Numerical features: {mtd['input_datasets'][:mtd['N_num']]}")
        self.logger.info(f"Categorical features: {mtd['input_datasets'][mtd['N_num']:]}")
        self.logger.info(f'Number of patches: {mtd["n_patches"]}')
        self.logger.info('Total array shape [N_samples x patch_size x patch_size x N_features]: ')
        self.logger.info(f'mask:                                   {self.Y.shape}')
        self.logger.info(f'numerical features:                     {self.X.shape[:3]+(mtd["N_num"],)}') 
        if mtd["N_cat"] > 0:       
            self.logger.info(f'categorical features (after OHE):       {self.X.shape[:3]+(mtd["N_cat"],)}') 
        return self.X, self.Y

   
if __name__ == "__main__":
    cnn_preparator = NetPreprocessor(config_path='src/config/config.yml')  
    samples = [
        ('32TPT', '20240602T122300'),
        ('32TPT', '20240606T104243')
        ]
    # train
    cnn_preparator.prepare_input_arrays(
        samples=samples, 
        resolution=70, 
        target_manager=cnn_preparator.target_manager,
        s2_manager=cnn_preparator.s2_manager,
        inca_manager=cnn_preparator.inca_manager,
        window=None, 
        patch_size=5, 
        overlap_pixels=False, 
        data_type='train' # 'train' or 'test'
    )
    # # predict
    # cnn_preparator.prepare_input_arrays(
    #     samples=samples[0], 
    #     resolution=70,
    #     window=None, 
    #     patch_size=5, 
    #     overlap_pixels=False, 
    #     data_type='test'
    # )