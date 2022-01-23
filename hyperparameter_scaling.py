#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Scaling with respect to the hyperparameters. First run get_segmentations.py to 
compute all segmentations. Then run this script to obtain the results from 
Table 2.

"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import time

from utils.aux_functions import get_segmentation
from utils.aux_functions import ILSVRC_index_reader
from utils.aux_functions import dataset_helper

DBL_MAX = sys.float_info.max

# rescaling parameter
kappa = 0.5

# parameters
small_ks = 5
small_dm = 10
large_ks = kappa*small_ks
large_dm = kappa*small_dm

# which dataset are we looking at?
dataset = 'ILSVRC'

if dataset == 'ILSVRC':
    
    # path to the index file
    index_path = "utils/images_id.txt"

    # get the data
    ids,rect_store = ILSVRC_index_reader(index_path,verbose=False)

n_images,data_folder = dataset_helper(dataset)

# path to the precomputed segmentations
result_folder = "../results/segmentations/" + dataset + '/'
    
# main loop
n_large_store  = np.zeros((n_images,))
n_small_store  = np.zeros((n_images,))
for i_image in range(n_images):
    
    t_start = time.time()
    
    if dataset == 'ILSVRC':
        id_image = ids[i_image]
    else:
        id_image = i_image
        
    print("looking at image {}".format(id_image))
    
    # get the segmentation for the large parameters
    large_seg = get_segmentation(result_folder,
                                 data_folder,
                                 id_image,
                                 ratio=None,
                                 kernel_size=large_ks,
                                 max_dist=large_dm,
                                 verbose=True,
                                 dataset=dataset)
    
    n_large_store[i_image] = len(np.unique(large_seg))

    # get the segmentation for the smaller parameters
    small_seg = get_segmentation(result_folder,
                                 data_folder,
                                 id_image,
                                 ratio=None,
                                 kernel_size=small_ks,
                                 max_dist=small_dm,
                                 verbose=True,
                                 dataset=dataset)

    n_small_store[i_image] = len(np.unique(small_seg))


    print("large seg: {}".format(n_large_store[i_image]))
    print("small seg: {}".format(n_small_store[i_image] ))

    # timing
    t_end = time.time()
    print("elapsed: {}s".format(np.round(t_end-t_start,2)))
    print()

# looking at the ratio
comp = n_large_store/n_small_store

print("average ratio: {}".format(np.mean(comp)))
print("std: {}".format(np.std(comp)))

##############################################################################

# for visualization purposes

big_lw = 3
small_lw = 1.5
my_alpha = 0.2
big_fs = 25
small_fs = 15


fig,ax = plt.subplots(1,1,figsize=(15,10))
ax.scatter(np.arange(1,n_images+1),comp)
ax.hlines(1/kappa**2,0,n_images,color='r',linewidth=big_lw)










