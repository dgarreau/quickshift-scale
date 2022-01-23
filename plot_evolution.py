#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script produces the plots from Figure 6, 14, 15, and 16 of the paper. 

IMPORTANT: 
    - first compute the raw results by running compute_evolution.py
    - the parameters should correspond to something already computed

"""

import sys
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

from utils.aux_functions import format_helper
from utils.aux_functions import get_result_path

DBL_MAX = sys.float_info.max

# quickshift hyperparameters
kernel_size = 5
ratio = 1.0

# image shape and noise level: tweak here if you want to plot something different
min_shape = 150
max_shape = 160
n_exp = max_shape-min_shape
shape_store = np.arange(min_shape,max_shape)
sigmas = np.array([1,10])
n_sigmas = sigmas.shape[0]

dms = np.array([DBL_MAX,18,10])
n_dm = dms.shape[0]
n_rep = 10

n_seg_store    = np.zeros((n_exp,n_rep,n_dm,n_sigmas))
n_max_store    = np.zeros((n_exp,n_rep,n_dm))
n_approx_store = np.zeros((n_exp,n_dm))

# where to look for the results
result_folder = "../results/evolution/"

# loading results
for i_dist in range(n_dm):
    max_dist = dms[i_dist]
    
    result_path =  get_result_path(result_folder,kernel_size,max_dist)
    
    for i_sig in range(n_sigmas):
        
        sigma = sigmas[i_sig]
        aux_str = "sigma_" + str(sigma) + "_ratio_" + str(ratio) + "_min_" + str(min_shape) + "_max_" + str(max_shape)
        pickle_name = os.path.join(result_path,aux_str + ".pkl")
        with open(pickle_name,'rb') as f:
            ks,md,sig,rat,min_s,max_s,aux_n_seg,aux_n_max,aux_approx = pickle.load(f)


        # checking that everything corresponds
        if kernel_size != ks or sig != sigma or rat != ratio:
            print("something is wrong, input parameters do not correspond to what is stored")
            sys.exit()

        # store everything
        n_seg_store[:,:,i_dist,i_sig] = aux_n_seg
        if i_sig == 0:
            n_max_store[:,:,i_dist] = aux_n_max
            n_approx_store[:,i_dist] = aux_approx

##############################################################################

# force vector fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

big_lw = 3
small_lw = 1.5
my_alpha = 0.2
big_fs = 25
small_fs = 15

fig,ax = plt.subplots(1,n_dm,figsize=(10*n_dm,7))    

for i_dist in range(n_dm):

    # plotting the number of local maxima
    mean_local_max = np.mean(n_max_store[:,:,i_dist],1)
    loc_line, = ax[i_dist].plot(shape_store,mean_local_max,color='b')
    std_local_max = np.std(n_max_store[:,:,i_dist],1)
    ax[i_dist].fill_between(shape_store,mean_local_max-std_local_max,mean_local_max+std_local_max,facecolor='b',alpha=my_alpha)
    loc_line.set_label(r"average number of local maxima ($\sigma= {}$)".format(sigmas[0]))
    
    # plotting the number of superpixels, small sigma
    mean_n_seg = np.mean(n_seg_store[:,:,i_dist,0],1)
    seg_line_small, = ax[i_dist].plot(shape_store,mean_n_seg,color='k',linestyle='-')
    std_n_seg= np.std(n_seg_store[:,:,i_dist,0],1)
    ax[i_dist].fill_between(shape_store,mean_n_seg-std_n_seg,mean_n_seg+std_n_seg,facecolor='k',alpha=my_alpha)
    seg_line_small.set_label(r"average number of superpixels ($\sigma={}$)".format(sigmas[0]))
    
    # plotting the number of superpixels, large sigma
    mean_n_seg = np.mean(n_seg_store[:,:,i_dist,1],1)
    seg_line_large, = ax[i_dist].plot(shape_store,mean_n_seg,color='k',linestyle='--')
    std_n_seg= np.std(n_seg_store[:,:,i_dist,1],1)
    ax[i_dist].fill_between(shape_store,mean_n_seg-std_n_seg,mean_n_seg+std_n_seg,facecolor='k',alpha=my_alpha)
    seg_line_large.set_label(r"average number of superpixels ($\sigma={}$)".format(sigmas[1]))
 
    
    # plotting the theory
    th_line, = ax[i_dist].plot(range(min_shape,max_shape),n_approx_store[:,i_dist],color='r',linewidth=big_lw)
    th_line.set_label("approximation")
    
    # bigger fonts
    ax[i_dist].tick_params(axis='x',labelsize=small_fs)
    ax[i_dist].tick_params(axis='y',labelsize=small_fs)

    # legend
    ax[i_dist].legend(fontsize=small_fs,loc=2)

    # title
    s_kernel_size,s_max_dist = format_helper(kernel_size,dms[i_dist])
    s_title = r"kernel_size $=$ " + s_kernel_size + ", max_dist $=$ " + s_max_dist 
    ax[i_dist].set_title(s_title,fontsize=big_fs)

    # xlabel
    ax[i_dist].set_xlabel("rectangle width",fontsize=small_fs)
    
    # set a common scale
    #ax[i_dist].set_ylim(0,np.max())

# saving the figure
#fig_name = "../figures/final_evolution_ks_" + s_kernel_size + ".pdf"
#fig.savefig(fig_name,format='pdf',bbox_inches = 'tight',pad_inches = 0)

    