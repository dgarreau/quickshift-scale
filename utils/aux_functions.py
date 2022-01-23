#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Auxilliary functions are collected here.

"""

import numpy as np
import sys
import os

from errno import EEXIST
from os import makedirs
from os import path

DBL_MAX = sys.float_info.max

def image_shape_helper(input_shape,ratio=1.0):
    """
    Modify this function to change the way h,w evolve with respect to 
    input_shape (here we are considering squares).
    """
    height = int(input_shape)
    width = int(np.round(ratio*input_shape,decimals=0))
    return height,width

def get_local_max(my_seg,my_tree):
    """
    Get the local maxima ids from the quickshift tree.
    Inspired from the scikit learn implementation.
    """
    
    my_tree_flat = my_tree.ravel()
    
    # get the number of superpixels
    n_seg = len(np.unique(my_seg))

    # for each superpixels, get the local tree
    local_max = np.zeros((n_seg,),dtype=int)
    for i_seg in range(n_seg):
        
        # find the local max by starting at an arbitrary point
        local_tree = my_tree[my_seg == i_seg]
        current = local_tree[0]
        
        # we follow the path, and stop when not moving anymore or moving 
        # outside the local structure
        cont = True
        while cont:
            succ = my_tree_flat[current]
            if succ == current or not np.isin(succ,local_tree):
                cont = False
            else:
                current = succ
        local_max[i_seg] = current

    return local_max

def convert_to_2D(indices,height,width):
    """
    Converting flat indices to 2D indices.
    """
    n_ind = indices.shape[0]
    new_indices = np.zeros((n_ind,2),dtype=int)
    for i in range(n_ind):
        current = indices[i]
        line = int(np.round(current/width,decimals=0))
        column = int(np.mod(current,width))
        new_indices[i] = (line,column)
    return new_indices

def count_local_max(local_maxima,h_min,h_max,w_min,w_max):
    """
    Couting the number of local maxima in a rectangle, assuming that the local 
    max are already converted to 2D.
    """
    n_tot = local_maxima.shape[0]
    n_max = 0
    for i_max in range(n_tot):
        i,j = local_maxima[i_max]
        if h_min <= i and i < h_max and w_min <= j and j < w_max:
            n_max += 1
    return n_max

def format_helper(kernel_size,max_dist):
    """
    Standardized naming for the hyperparameters.
    """
    s_kernel_size = str(kernel_size)
    if max_dist >= DBL_MAX/2:
        s_max_dist = "inf"
    else:
        s_max_dist = str(int(np.round(max_dist,decimals=0)))
    
    return s_kernel_size,s_max_dist

def get_result_path(result_folder,kernel_size,max_dist,ratio=None):
    """
    Standardized naming.
    """
    s_kernel_size,s_max_dist = format_helper(kernel_size,max_dist)
    if ratio is None:
        aux_res_path = "ks_" + s_kernel_size + "_dm_" + s_max_dist + "/"
    else:
        aux_res_path = "ks_" + s_kernel_size + "_dm_" + s_max_dist + "_ratio_" + str(ratio) + "/"
    result_path = os.path.join(result_folder, aux_res_path)
    return result_path

def mkdir(mypath):
    """
    Creates a directory (credits to https://stackoverflow.com/questions/11373610/save-matplotlib-file-to-a-directory)
    
    INPUT:
        - mypath: str with path to the directory    
    """
    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: 
            raise
