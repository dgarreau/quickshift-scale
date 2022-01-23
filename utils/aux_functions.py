#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Auxilliary functions are collected here.

"""

import numpy as np
import sys
import os
import pickle

from errno import EEXIST
from os import makedirs
from os import path

from skimage.io import imread
from skimage.color import gray2rgb
from skimage.transform import resize
from skimage.segmentation import quickshift

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

def name_helper(id_image,dataset='ILSVRC'):
    """
    Formatting the image name with dataset appropriate convention.
    """
    if dataset == 'ILSVRC':
        image_name = 'ILSVRC2017_test_' + str(id_image+1).zfill(8)  
    elif dataset == 'cityscapes':
        image_name = "berlin_" + str(id_image).zfill(6) + "_000019_leftImg8bit"
    elif dataset == 'pascal':
        with open('utils/pascal_index.txt','r') as f:
            for i,line in enumerate(f):
                if i == id_image:
                    image_name = line[:-1]
    return image_name

def get_segmentation(result_folder,data_path,id_image,ratio=None,kernel_size=5,max_dist=10,verbose=False,dataset='ILSVRC'):
    """
    Getting the segmentation of an image. We check if precomputed, compute 
    otherwise.
    """
    image_name = name_helper(id_image,dataset=dataset)
    image_path = path_helper(data_path,image_name,dataset)
    result_path = get_result_path(result_folder,kernel_size,max_dist,ratio=ratio)
    seg_path = os.path.join(result_path,image_name + '.pkl')
    if verbose:
        print("trying to open " + seg_path + "...")
    try:
        with open(seg_path,'rb') as f_seg:
            if verbose:
                print("the result file seems to exist, fetching data...")
            comp_seg = pickle.load(f_seg)
            if verbose:
                print("decompressing...")
            seg = decompress(comp_seg)

    except IOError:
            if verbose:
                print("The result file does not seem to exist, computing the segmentation...")
            
            image_rgb = load_image(image_path,ratio=ratio,verbose=verbose)
            seg = quickshift(image_rgb,kernel_size=kernel_size,max_dist=max_dist)
    
    return seg

def compress_helper(array):
    """
    array contains a sequence of ints with a lot of equalities. we count them 
    and return a list of (int,number of occurences)
    """
    length = array.shape[0]
    compressed_array = []
    current_value = array[0]
    current_count = 0
    for i in range(length):
        if array[i] == current_value:
            current_count += 1
        else:
            compressed_array += (current_value,current_count)
            current_value = array[i]
            current_count = 1
    if current_count != 0:
        compressed_array += (current_value,current_count)
    return compressed_array

def decompress_helper(comp_list,width):
    """
    Decompressing the compressed version.
    """
    length = len(comp_list)
    res = np.zeros((width,),dtype=int)
    
    # filling the array
    current_index = 0
    for i in range(int(length/2)):
        current_value = comp_list[2*i]
        n_rep = comp_list[2*i+1]
        res[current_index:(current_index+n_rep)] = current_value
        current_index += n_rep
        
    return res

def compress(segmentation):
    """
    compressing a segmentation as a list of lists. 
    """
    height,width = segmentation.shape
    compressed = []
    for i in range(height):
        compressed += [compress_helper(segmentation[i])]
    return compressed

def get_width(comp_list):
    """
    Get the height from a compressed list.
    """
    length = len(comp_list)
    width = 0
    for i in range(int(length/2)):
        width += comp_list[2*i+1]
    return width

def decompress(comp_seg):
    """
    Decompress a compressed segmentation.
    """
    height = len(comp_seg)
    width = get_width(comp_seg[0])
    segmentation = np.zeros((height,width),dtype=int)
    for i in range(height):
        segmentation[i] = decompress_helper(comp_seg[i],width)
    return segmentation

def load_image(image_path,ratio=None,verbose=False):
    """
    Loading a RGB image from path.
    """
    if verbose:
        print("retrieving " + image_path)
    image = imread(image_path)
    n_channels = len(image.shape)
    if n_channels == 3:
        image_rgb = image
    elif n_channels == 2:
        image_rgb = gray2rgb(image)
    if ratio is not None:
        height = int(np.round(image.shape[0]/ratio,decimals=0))
        width = int(np.round(image.shape[1]/ratio,decimals=0))
        image_rgb = resize(image_rgb,(height,width))
    return image_rgb

def ILSVRC_index_reader(index_path,verbose=False):
    """
    Reading the index file for the ILSVRC subset.
    """
    
    n_images = 748
    ids = np.zeros((n_images,),dtype=int)
    rect_store = np.zeros((n_images,4),dtype=int)

    # reading the index file
    count = 0
    with open(index_path,'r') as f_index:
        while count < n_images:
            line = f_index.readline()
            splitted_line = line.split(" ")
            # # means comment
            if splitted_line[0] != '#':
                ids[count] = int(splitted_line[0])
                #print(splitted_line)
                for j in range(4):
                    rect_store[count,j] = int(splitted_line[j+1])            
                count += 1
    return ids,rect_store

def path_helper(data_path,image_name,dataset='ILSVRC'):
    """
    Getting image path depending on dataset.
    """
    if dataset == 'ILSVRC':
        image_path = os.path.join(data_path,image_name + '.JPEG')
    elif dataset == 'cityscapes':
        image_path = os.path.join(data_path,image_name + '.png')
    elif dataset == 'pascal':
        image_path = os.path.join(data_path,image_name + '.jpg')
    return image_path

def dataset_helper(dataset):
    """
    Number of images and path to data depending on the dataset.
    """
    if dataset == 'ILSVRC':

        n_images = 748
        data_path = "../data/ILSVRC/Data/DET/test/"

    elif dataset == 'cityscapes':
        
        n_images = 544
        data_path = "../data/cityscapes/leftImg8bit/test/berlin/"
    
    elif dataset == 'pascal':
        
        n_images = 500
        data_path = "../data/pascal/VOCdevkit/VOC2012/JPEGImages/"
    return n_images,data_path