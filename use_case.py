#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Use-case experiment. Run this script to obtain Figure 9, 14-16 of the paper. 

TODO: other datasets!
 - nonsense with the comma

"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import time

from skimage.segmentation import mark_boundaries

from utils.aux_functions import format_helper
from utils.aux_functions import load_image
from utils.aux_functions import get_segmentation
from utils.aux_functions import name_helper

from utils.aux_functions import ILSVRC_index_reader

t_start = time.time()

DBL_MAX = sys.float_info.max

# the ratio for image rescaling
ratio = 2.0

# quickshift parameters
kernel_size = 5
#kernel_width = 3*kernel_size
max_dist = DBL_MAX

# for print purposes
s_kernel_size,s_max_dist = format_helper(kernel_size,max_dist)

# path to the data
data_folder = "../data/ILSVRC/Data/DET/test/"

# path to the precomputed segmentations
result_folder = "../results/segmentations/ILSVRC/"

# path to the index file
index_path = "utils/images_id.txt"

ids,rect_store = ILSVRC_index_reader(index_path,verbose=False)
n_images = ids.shape[0]

#i_image = 30
#id_image = ids[i_image]
id_image = 7
print("looking at image {}".format(id_image))

# segmentating the small image
image_path = data_folder + name_helper(id_image) + '.JPEG'
small_image_rgb = load_image(image_path,ratio=ratio,verbose=True)
small_seg = get_segmentation(result_folder,data_folder,id_image,ratio=ratio,kernel_size=kernel_size,max_dist=max_dist)
n_small = len(np.unique(small_seg))
print("ratio = {}, ks = {}, dm = {}:".format(ratio,kernel_size,max_dist))
print("   {} superpixels".format(n_small))

# segmentating the large image with the same parameters
large_image_rgb = load_image(image_path,verbose=True)
large_seg = get_segmentation(result_folder,data_folder,id_image,ratio=None,kernel_size=kernel_size,max_dist=max_dist)
n_large = len(np.unique(large_seg))
print("ratio = None, ks = {}, dm = {}".format(kernel_size,max_dist))
print("   {} superpixels".format(n_large))

# rescaling the parameters
new_kernel_size = ratio*kernel_size
if max_dist == DBL_MAX:
    new_max_dist = DBL_MAX
else:
    new_max_dist = ratio*max_dist
s_ks_new,s_dm_new = format_helper(new_kernel_size,new_max_dist)
new_seg = get_segmentation(result_folder,data_folder,id_image,ratio=None,kernel_size=new_kernel_size,max_dist=new_max_dist)
n_rescaled = len(np.unique(new_seg))
print("ratio = None, ks = {}, dm = {}".format(new_kernel_size,new_max_dist))
print("   {} superpixels".format(n_rescaled))

t_end = time.time()

print("{} elapsed".format(t_end-t_start))

##############################################################################

s_title_0 = r"$k_s=" + s_kernel_size + ", d_m = " + s_max_dist + "\\rightarrow N=" + str(n_small) + "$" 
s_title_1 = r"$k_s=" + s_kernel_size + ", d_m = " + s_max_dist + "\\rightarrow N=" + str(n_large) + "$"
s_title_2 = r"$k_s=" + s_ks_new + ", d_m = " + s_dm_new + "\\rightarrow N=" + str(n_rescaled) + "$"

# plot parameters
big_lw = 3
small_lw = 1.5
my_alpha = 0.2
big_fs = 20
small_fs = 15

fig,ax = plt.subplots(1,3,figsize=(15,10))

ax[0].imshow(mark_boundaries(small_image_rgb,small_seg))
ax[0].set_title(s_title_0,fontsize=big_fs)
ax[0].axis('off')

ax[1].imshow(mark_boundaries(large_image_rgb,large_seg))
ax[1].set_title(s_title_1,fontsize=big_fs)
ax[1].axis('off')

ax[2].imshow(mark_boundaries(large_image_rgb,new_seg))
ax[2].set_title(s_title_2,fontsize=big_fs)
ax[2].axis('off')

fig_name = "../figures/final_rescaling_id_" + str(id_image) + "_ks_" + s_kernel_size + "_dm_" + s_max_dist + "_ratio_" + str(ratio) + ".pdf"
fig.savefig(fig_name,format='pdf',bbox_inches = 'tight',pad_inches = 0)

