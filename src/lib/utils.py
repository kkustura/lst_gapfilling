import os
import numpy as np
import random
from datetime import datetime


def list_filepaths(dir, patterns_in, patterns_out, include_all_patterns=True, print_warning=False):
    
    '''
    List of filepaths in a dir that contain all patterns in patterns_in, 
    and do not contain any of the patterns in patterns_out.
    include_all_patterns: if True, all patterns in patterns_in are required in a single filename.
    If False, any of the patterns in patterns_in is sufficeint. 
    (Note: include_all_patterns does not apply to pattern out - ALL patterns in patterns_out are always excluded)
    '''
    
    if include_all_patterns:
        def patterns_in_bool(i):
            return all(pattern in i for pattern in patterns_in)
    else:
        def patterns_in_bool(i):
            return any(pattern in i for pattern in patterns_in)
        
    def patterns_out_bool(i):
        return all(pattern not in i for pattern in patterns_out)

    out = [os.path.join(dir,i) for i in os.listdir(dir) if patterns_in_bool(i) and patterns_out_bool(i)]
    
    if not out and print_warning:
        print(f'Warning! No paths found for the specified patterns ({patterns_in}) in {dir} (returning an empty list). ')
        
    return out
    

def locate_single_file(dir, patterns_in, patterns_out=[], include_all_patterns=True, print_warning=False):
    file_paths = list_filepaths(dir, patterns_in, patterns_out, include_all_patterns, print_warning)
    if len(file_paths) > 1:
        raise ValueError(f"Multiple files found in {dir} matching patterns {patterns_in}. "
                         f"Please refine your search or use a different method to select a file.")
    elif len(file_paths) == 0:
        raise FileNotFoundError(f"No files found in {dir} matching patterns {patterns_in}. "
                                f"Please check the directory and patterns.")
    else:
        return file_paths[0]




###############################################################################
# TBD!!!! array processing
def find_nearest_index(array, value):
    """Find index of element in array nearest to given value"""
    is_datetime = isinstance(value, datetime)
    if is_datetime:
        diffs = [abs((value - d).total_seconds()) for d in array]
    else:
        diffs = [abs(value - d) for d in array]
    return int(np.argmin(diffs))






##############################################################################
#TBD!!!! where is this used?

def list_filepaths_in_subdirs(dir, patterns_in, patterns_out):
    
    '''
    List of filepaths in a dir and its subdirs that contain all patterns 
    in patterns_in, and do not contain any of the patterns in patterns_out.
    '''
    
    file_paths = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            if any(pattern in f for pattern in patterns_in) and all(pattern not in f for pattern in patterns_out):  
                file_paths.append(os.path.join(root, f))
    return file_paths
    
    
    
def write_paths_to_txt(file_paths, output_file):
    
    
    '''
    Write a list of files in file_paths to output txt file.
    '''
    
    with open(output_file, 'w') as file:
        for path in file_paths:
            file.write(path + '\n')

         