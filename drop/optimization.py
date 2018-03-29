#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:16:27 2018

@author: fr
"""
import numpy as np

from scipy.interpolate import interp1d
from drop.utils import split_profile
from drop.theory import rotate_lines, young_laplace

def squared_distance(R, Z, R_edges, Z_edges):
    """
    Calculate the squared distance for half profile.

    Theoretical points are interpolated on experimental ones.


    """
    # TODO: use of extrapolate due to the rotation.
    # We can be smarter and integrate over a longer distance...
    R_theo_interpolator = interp1d(Z, R, kind='linear', fill_value='extrapolate')
    R_theo_interpolated = R_theo_interpolator(Z_edges)
    return (R_theo_interpolated - R_edges)**2


def error_f(variables, image_shape, radius, R_edges, Z_edges, tip, guess_tipx, center_x):
    """
    Return the RMS for a profile given by set of parameters to the experimental profile.
    """
    print("variables:",  variables)


    R, Z = young_laplace(variables, image_shape, radius, R_edges, Z_edges, tip, guess_tipx, center_x)


    R_left, Z_left, R_right, Z_right = split_profile(R, Z)
    R_edges_left, Z_edges_left, R_edges_right, Z_edges_right = split_profile(R_edges, Z_edges)

    # Error on both sides.
    e_left = squared_distance(R_left, Z_left, R_edges_left, Z_edges_left)
    e_right = squared_distance(R_right, Z_right, R_edges_right, Z_edges_right)

    e_all = np.concatenate((e_left, e_right))
    chi_squared = np.sum(e_all)
    RMS = np.sqrt(chi_squared) / len(e_all)
    return RMS
