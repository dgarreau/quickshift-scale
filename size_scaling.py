#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Scaling with respect to the image size. First run get_segmentations.py to 
compute all segmentations. Then run this script to obtain the results from 
Table 1.

"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import time

from utils.aux_functions import format_helper
from utils.aux_functions import get_segmentation
from utils.aux_functions import dataset_helper
from utils.aux_functions import ILSVRC_index_reader

DBL_MAX = sys.float_info.max

# ratio for image rescaling
ratio = 2

# quickshift hyperparameters
kernel_size = 5
#kernel_width = 3*kernel_size
max_dist = 18

# for print purposes
s_kernel_size,s_max_dist = format_helper(kernel_size,max_dist)

# which dataset are we looking at?
dataset = "ILSVRC"

if dataset == 'ILSVRC':
    
    # path to the index file
    index_path = "utils/images_id.txt"
    ids,rect_store = ILSVRC_index_reader(index_path,verbose=False)

n_images,data_folder = dataset_helper(dataset)

# path to the precomputed segmentations
result_folder = "../results/segmentations/" + dataset 

n_large_store  = np.zeros((n_images,))
n_small_store  = np.zeros((n_images,))
for i_image in range(n_images):
    
    t_start = time.time()
    
    #id_image = i_image
    if dataset == "ILSVRC":
        id_image = ids[i_image]
    else:
        id_image = i_image
    
    print("looking at image {}".format(id_image))

    # get the segmentation for the large image
    large_seg = get_segmentation(result_folder,
                                 data_folder,
                                 id_image,
                                 ratio=None,
                                 kernel_size=kernel_size,
                                 max_dist=max_dist,
                                 verbose=True,
                                 dataset=dataset)

    # number of superpixels in the segmentation of the original image
    n_large_store[i_image] = len(np.unique(large_seg))

    # get the segmentation for the smaller image
    small_seg = get_segmentation(result_folder,
                                 data_folder,
                                 id_image,
                                 ratio=ratio,
                                 kernel_size=kernel_size,
                                 max_dist=max_dist,
                                 verbose=True,
                                 dataset=dataset)
    
    # number of superpixels in the segmentation of the resized image
    n_small_store[i_image] = len(np.unique(small_seg))
        
    print("large seg: {}".format(n_large_store[i_image]))
    print("small seg: {}".format(n_small_store[i_image]))
    

    # timing
    t_end = time.time()
    print("elapsed: {}s".format(np.round(t_end-t_start,2)))
    print()

# looking at the ratio
comp = n_large_store/n_small_store

print("average ratio: {}".format(np.mean(comp)))
print("std: {}".format(np.std(comp)))

##############################################################################

# some plotting to visualize the distribution

big_lw = 3
small_lw = 1.5
my_alpha = 0.2
big_fs = 25
small_fs = 15


fig,ax = plt.subplots(1,1,figsize=(15,10))
ax.scatter(np.arange(1,n_images+1),comp)
ax.hlines(ratio**2,0,n_images,color='r',linewidth=big_lw)



