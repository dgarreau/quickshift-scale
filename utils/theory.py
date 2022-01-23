#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Auxilliary functions for the theory are collected here.

"""

import numpy as np

def compute_gamma(s_data,d_data):
    """
    Computing the gamma function (area of a circle segment of parameters (s,d), 
    where d is the radius of the circle and s the half of the square size).
    """
    d_sq = np.square(d_data)
    s_sq = np.square(s_data)
    return d_sq*np.arctan(np.sqrt(d_sq-s_sq)/s_data) - s_data * np.sqrt(d_sq-s_sq)

def compute_rounded_square_area(s_data,d_data):
    """
    Computing the area of a rounded square of parameters s,d.
    """
    return np.pi*np.square(d_data) - 4*compute_gamma(s_data,d_data)

def compute_approx_expec(height,width,h_min,h_max,w_min,w_max,kernel_size=5,max_dist=10):
    """
    Approx expectation from Th. 3.2.    
    """
    # kernel width is specified here
    kernel_width = 3*kernel_size
    
    # looking at the intersection with the image center
    if h_min <= kernel_width:
        new_h_min = kernel_width
    else:
        new_h_min = h_min
    if h_max >= height - kernel_width:
        new_h_max = height - kernel_width
    else:
        new_h_max = h_max
    if w_min <= kernel_width:
        new_w_min = kernel_width
    else:
        new_w_min = w_min
    if w_max >= width - kernel_width:
        new_w_max = width - kernel_width
    else:
        new_w_max = w_max
    
    # dimensions of the rectangle to consider
    rect_height = new_h_max - new_h_min
    rect_width  = new_w_max - new_w_min
    rect_area = rect_height*rect_width
    
    # disjunction depending on the relative size of ks and dm
    if max_dist < kernel_width:
        return rect_area/(np.pi*max_dist**2)
    elif kernel_width <= max_dist < np.sqrt(2)*kernel_width:
        return rect_area/compute_rounded_square_area(kernel_width,max_dist)
    else:
        return rect_area/(4*kernel_width**2)



