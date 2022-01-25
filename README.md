# How to scale hyperparameters for quickshift image segmentation

Python code for the paper [How to scale hyperparameters for quickshift image segmentation](https://arxiv.org/abs/2201.09286)

There is nothing to install, one simply has to run the scripts. Some experiments require to download data (additional details are given in the scripts).


## Disclaimer 


The code was tested with version 0.18.2 of skimage, using another version may lead to unexplained behaviors. 


## General Organization 


The main scripts producing the results and plotting are in the root directory:

 * compute_evolution.py: raw results for the evolution of the number of local maxima and number of superpixels as a function of the image size.
 * plot_evolution.py: plotting the evolution of the number of superpixels as a function of the image size
 * get_segmentations.py: computs segmentations with quickshift and storing the results
 * size_scaling.py: scaling with respect to the size of the images on real data
 * hyperparameter_scaling.py: scaling with respect to the hyperparameters on real data

Auxilliary functions are collected in the utils folder.

