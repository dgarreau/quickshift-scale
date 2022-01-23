# -*- coding: utf-8 -*-
"""

Computing the raw results for Figure 6, 14, 15, and 16. We look at the number 
of superpixels and local maxima in a rectangle at distance 2*kernel_width from 
the border to totally ignore the border effects. Results are saved in the 
folder result_folder

"""

import numpy as np
import sys
import os
import pickle
import time

from skimage.color import rgb2lab
from skimage.segmentation import quickshift

from utils.aux_functions import get_local_max
from utils.aux_functions import convert_to_2D
from utils.aux_functions import count_local_max
from utils.aux_functions import get_result_path
from utils.aux_functions import mkdir
from utils.aux_functions import image_shape_helper

from utils.theory import compute_approx_expec

DBL_MAX = sys.float_info.max

# default values: kernel_size = 5, kernel_width = 15, max_dist = 10
kernel_size = 10
kernel_width = 3*kernel_size
max_dist = 15

# shape of the image and noise level
sigma = 5.0
ratio = 1.0

# number of repetitions to get error bars
# NOTE: already a lot of randomness in the process, not a lot of reps are needed
n_rep = 10

# background color
# NOTE: does not change anything except if we want to plot the original image
background_color = (100,100,100)

# size of the images
# NOTE: - 28 is the MNIST standard
# - 299 is the imagenet standard
# - should start at 4*kernel_width minimum to avoid any problem
min_shape = 150
max_shape = 160
n_exp = max_shape - min_shape
n_seg_store = np.zeros((n_exp,n_rep),dtype=int)
n_max_store = np.zeros((n_exp,n_rep),dtype=int)

t_start = time.time()

for i_exp in range(n_exp):
    
    # current input shape
    input_shape = min_shape + i_exp
    height,width = image_shape_helper(input_shape,ratio=ratio)
    print("size {},{}, kernel_size = {}, max_dist = {}, sigma = {}".format(height,width,kernel_size,max_dist,sigma))
    
    h_min = int(2*kernel_width)
    h_max = int(height - 2*kernel_width)
    w_min = int(2*kernel_width)
    w_max = int(width - 2*kernel_width)
    
    for i_rep in range(n_rep):
        
        # create the image
        xi_array = np.ones((height,width,3),dtype=float) * background_color
    
        # converting to lab space
        xi_rgb   = np.uint8(xi_array)
        xi_lab = rgb2lab(xi_rgb)

        # adding some noise to the pixel values in the lab space
        xi_lab += np.random.normal(0,sigma,(height,width,3))
        
        # computing the segmentation
        out = quickshift(xi_lab,
                                 max_dist=max_dist,
                                 kernel_size=kernel_size,
                                 random_seed=i_rep,
                                 convert2lab=False,
                                 return_tree=True)
        my_seg  = out[0]
        my_tree = np.asarray(out[1])
        
        n_seg = len(np.unique(my_seg[h_min:h_max,w_min:w_max]))
        print("     -> {} superpixels".format(n_seg))
        n_seg_store[i_exp,i_rep] = n_seg
        
        local_max = get_local_max(my_seg,my_tree)
        local_max_2D = convert_to_2D(local_max,height,width)
        n_max = count_local_max(local_max_2D,h_min,h_max,w_min,w_max)
        print("     -> {} local max".format(n_max))
        n_max_store[i_exp,i_rep] = n_max
    
# ncomputing the theory
approx_store = np.zeros((n_exp,))
for i_exp in range(n_exp):

    input_shape = min_shape + i_exp
    height,width = image_shape_helper(input_shape,ratio=ratio)
    h_min = 2*kernel_width
    h_max = height - 2*kernel_width
    w_min = 2*kernel_width
    w_max = width - 2*kernel_width
    approx_theo = compute_approx_expec(height,
                                     width,
                                     h_min,h_max,w_min,w_max,
                                     kernel_size=kernel_size,max_dist=max_dist)
    approx_store[i_exp] = approx_theo


t_end = time.time()
print("elapsed = {}".format(t_end-t_start))
print()

# saving the results
result_folder = "../results/evolution/"
result_path =  get_result_path(result_folder,kernel_size,max_dist)
mkdir(result_path)
aux_str = "sigma_" + str(sigma) + "_ratio_" + str(ratio) + "_min_" + str(min_shape) + "_max_" + str(max_shape)
pickle_name = os.path.join(result_path,aux_str + ".pkl")
print("saving results...")
with open(pickle_name,'wb') as f:
    pickle.dump([kernel_size,max_dist,sigma,ratio,min_shape,max_shape,n_seg_store,n_max_store,approx_store],f)
print("done!")
print()

