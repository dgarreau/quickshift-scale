#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Run this script to get segmentations on a set of images and save the results in 
a compressed format. 

IMPORTANT:
    - first download the data:
        - ILSVRC can be found at http://image-net.org/image/ILSVRC2017/ILSVRC2017_DET_test_new.tar.gz
        - CityScapes at https://www.cityscapes-dataset.com/downloads/ (after login)
        - Pascal VOC at http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    - modify the dataset_helper function in utils/aux_functions to specify 
    where the datasets are located. 

"""

import numpy as np
import pickle
import os
import sys
import time

from skimage.segmentation import quickshift

from utils.aux_functions import mkdir
from utils.aux_functions import compress
from utils.aux_functions import get_result_path
from utils.aux_functions import load_image
from utils.aux_functions import format_helper
from utils.aux_functions import get_local_max
from utils.aux_functions import name_helper
from utils.aux_functions import path_helper
from utils.aux_functions import dataset_helper
from utils.aux_functions import ILSVRC_index_reader

DBL_MAX = sys.float_info.max

# if ratio is None, then the default size of images is taken into account
# if not, then the image size gets divided by ratio
ratio = None

# quickshift hyperparameters
# default values: kernel_size = 5, kernel_width = 15, max_dist = 10
kernel_size = 2.5
#kernel_width = 3*kernel_size
max_dist = 5
s_kernel_size,s_max_dist = format_helper(kernel_size,max_dist)

# specify the dataset name here
dataset = 'ILSVRC'

if dataset == 'ILSVRC':
    # path to the index file
    index_path = "utils/images_id.txt"
    ids,rect_store = ILSVRC_index_reader(index_path,verbose=False)

n_images,data_path = dataset_helper(dataset)

# results from the experiment will be saved here
result_folder = "../results/segmentations/" + dataset + '/'
result_path = get_result_path(result_folder,kernel_size,max_dist,ratio)
mkdir(result_path)

# main loop
for i_image in range(n_images):
    
    if dataset == "ILSVRC":
        id_image = ids[i_image]
    else:
        id_image = i_image
    
    print("computing segmentation for image {}".format(id_image))
    print("kernel size = " + s_kernel_size + ", max_dist = " + s_max_dist + ", ratio = " + str(ratio))
    
    t_start = time.time()

    # get the filename
    image_name = name_helper(id_image,dataset)
    image_path = path_helper(data_path,image_name,dataset)
    pickle_name = os.path.join(result_path,image_name + '.pkl')

    # check if the file already exists
    if os.path.exists(pickle_name):
        print("already computed, nothing to do here!")
    else:
    
        # get the image    
        image_rgb = load_image(image_path,ratio=ratio,verbose=False)

        if len(image_rgb.shape) == 3:
            height,width,_ = image_rgb.shape
        else:
            height,width = image_rgb.shape
        
        print(height,width)

        # call quickshift and get the tree
        out = quickshift(image_rgb,
                         kernel_size=kernel_size,
                         max_dist=max_dist,
                         return_tree=True)

        # get the segmentation
        # BEWARE: int16 max value is 32767
        #seg_int = out[0].astype(np.int16)
        my_seg  = out[0]
        my_tree = np.asarray(out[1])

        # compress the segmentation
        comp_seg = compress(my_seg)

        # get the local maxima
        local_max = get_local_max(my_seg,my_tree)

        # save the result
        print("saving results...")
        with open(pickle_name,'wb') as f:
            pickle.dump(comp_seg,f)

        # sanity check
        #decomp = decompress(comp_seg)
        #plt.imshow(segmentation-decomp)

    t_end = time.time()
    
    print("elapsed: {}s".format(np.round(t_end-t_start,2)))
    print()
    
    

    









