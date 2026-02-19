#### Functions for image patching. Used for dividing image into 5x5 super-pixels in pixelwise CNN model ####

import numpy as np

def pad_to_multiple(val, patch_size):
    """Pad a value to the next multiple of patch_size."""
    return ((val + patch_size - 1) // patch_size) * patch_size


def cut_into_patches(arr, patch_size=5, padd_value=None, overlap_pixels=False):
        """
        Cut an array into patches of size patch_size x patch_size.
        If padd_value is provided, pad the array with this value before cutting.
        If overlap_pixels is True, generate patch for every pixel. If false, perform grid cutting
        (each pixel is in exactly one patch).
        """
        if not overlap_pixels:
            return blockshaped(arr, patch_size, patch_size, padd_value) 
        else: 
            return blockshaped_all_pixels(arr, patch_size, patch_size, padd_value) 
            

def blockshaped(arr, nrows, ncols, nv, tag=None, inca_diff=(1,1)):
    """
    Reshape array into patches.
    2D array of size (h,w) is reshaped into 3D array of size (h//nrows, nrows, w//ncols, ncols).
    If h or w is not divisible by nrows or ncols, the array is padded to the nearest multiple 
    of nrows or ncols before padding (relevant for reconstructing the original image).
    Returns the reshapped array; index of first pixel in the patch (0,0); and the shape 
    of the padded array (h_padded, w_padded).
    """
    if tag=='INCA':  #padd INCA array (assumed has less px than other images!)  !!! this is hardcoded and dependend on the shape of all the input files!!!!
        arr_zero = np.full((inca_diff[0], arr.shape[1]), nv)
        arr = np.concatenate((arr, arr_zero), axis=0)
        arr_zero = np.full((arr.shape[0],inca_diff[1]), nv)
        arr = np.concatenate((arr, arr_zero), axis=1)
    h, w = arr.shape
    
    # padd shape to the nearest multiple of nrows and ncols
    h_padded = pad_to_multiple(h, nrows)
    w_padded = pad_to_multiple(w, ncols)
    
    while h != h_padded:  # padd image at the bottom
        arr_zero = np.full((1, arr.shape[1]), nv)
        arr = np.concatenate((arr, arr_zero), axis=0)
        h += 1
    while w != w_padded:  # padd image at the right
        arr_zero = np.full((arr.shape[0],1), nv)
        arr = np.concatenate((arr, arr_zero), axis=1)
        w += 1
    arr_reshaped = (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols)) 
    return arr_reshaped, (0,0), (h_padded, w_padded)


def unblockshaped(input_array, output_shape, window_size):
    '''
    Function to reconstruct the intial image from the 5x5 patches.,
    Works both for the array of patches, as well as the array of middle values of each patch.
    '''
    if len(input_array.shape) != 3:  # it's an array of patches
        patches = np.zeros((input_array.shape[0],window_size,window_size))
        for i,p in enumerate(input_array):
            tmp=np.ones((window_size,window_size))
            patches[i] = p*tmp
    else:  # it's an array of single pixels
        patches = input_array
    rows, cols = output_shape
    patches = patches.reshape(rows//window_size, cols//window_size, window_size, window_size)  # Reshape patches into a 2D array
    reconstructed_image = patches.transpose(0, 2, 1, 3).reshape(rows, cols)  # Stack the patches to reconstruct the original image
    return reconstructed_image


def blockshaped_all_pixels(arr, nrows, ncols, nv, tag=None, inca_diff=(1,1)):   #!!!! doesn't work for new clipping anymore!!!!
    
    if tag=='INCA':  #padd INCA array (assumed has less px than other images!)  !!! this is hardcoded and dependend on the shape of all the input files!!!!
        arr_zero = np.full((inca_diff[0], arr.shape[1]), nv)
        arr = np.concatenate((arr, arr_zero), axis=0)
        arr_zero = np.full((arr.shape[0],inca_diff[1]), nv)
        arr = np.concatenate((arr, arr_zero), axis=1)
    
    margin_rows = nrows//2
    margin_cols = ncols//2
    
    arr_unpadd = arr[margin_rows:-1-margin_rows, margin_cols:-1-margin_cols]  # remove border pixels
    n_tot = arr_unpadd.shape[0]*arr_unpadd.shape[1]  # total number of 5x5 patches to be made
    
    patches = np.zeros((n_tot,nrows,ncols))
    
    counter = 0
    # msk = []
    for i in range(arr_unpadd.shape[0]):
        for j in range(arr_unpadd.shape[1]):
            tmp = arr[i:i+nrows,j:j+ncols]
            patches[counter] = tmp
            
            # test = len(tmp[np.where(tmp==large_no_data_value)])  # how many values are missing in the patch
            # # if test <= int(0.1*nrows*ncols):  # only consider patches with # of missing values below some threshold      
            # if test <= 2:
            #     msk.append(counter)  # add index to mask
            counter+=1
            
    # counter = 0
    # for i in range(arr_unpadd.shape[0]):
    #     for j in range(arr_unpadd.shape[1]):
    #         tmp = arr[i:i+nrows,j:j+ncols]
            
    #         test = len(tmp[np.where(tmp==large_no_data_value)])  # how many values are missing in the patch
    #         # if test <= int(0.1*nrows*ncols):  # only consider patches with # of missing values below some threshold      
    #         if test <= 1:
    #             patches[counter] = tmp
    #             counter+=1

    return patches, (margin_rows, margin_cols), arr_unpadd.shape  


def unblockshaped_all_pixels(input_array, input_shape, first_pixel):
    '''!!!! TBD Refine this !!!!'''
    
    output_array = np.full((input_shape[0]+(2*first_pixel[0]+1), input_shape[1]+(2*first_pixel[1]+1)),np.nan)  # to reconstruct the original shape (before margin padding),
    # print(output_array.shape)

    pred_arr = input_array.reshape(input_shape)
    # print(pred_arr.shape)
    
    output_array[first_pixel[0]:-1-first_pixel[0], first_pixel[1]:-1-first_pixel[1]] = pred_arr  # fill in unpadded regions with predicted values

    return output_array
