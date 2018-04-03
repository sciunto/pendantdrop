#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:16:27 2018

@author: fr
"""
import numpy as np

from scipy.interpolate import interp1d
from drop.utils import split_profile
from drop.theory import rotate_lines, theoretical_contour


def young_laplace(variables, image_shape, radius, R_edges, Z_edges,
                  tip, guess_tipx, center_x,
                  calib, rho=1000, gravity=9.81):
    """
    Returns the Young Laplace solution resized and oriented to the image.

    Parameters
    ----------

    calib : scalar
        Calibration in mm per px.
    rho : scalar, optional
        Fluid density.
    gravity : scalar, optional
        Gravitational acceleration.

    Returns
    -------
    """
    gamma = variables[0]
    theta = variables[1]
    center_y = variables[2]

    rho_g = rho * gravity
    capillary_length = np.sqrt(gamma / rho_g)
    r0 = radius * calib
    bond_number = (r0 / capillary_length)**2

    R, Z = theoretical_contour(image_shape, bond_number, tip, calib)

    # Rescale
    R = R * r0
    Z = Z * r0

    # Symetrize the contour
    R = np.concatenate((-R, R))
    Z = np.concatenate((Z, Z))

    # Rotate
    base_center = np.array((0, (center_x - guess_tipx + 1) * calib))
    R, Z = rotate_lines(R, Z, base_center, theta)
    R = np.array(R)
    Z = np.array(Z)

    # Cut
    Z0 = image_shape[0] - tip[1]
    Zmax = Z0 * calib  # maximum possible values of Z to be upgraded
    R = R[Z < Zmax]
    Z = Z[Z < Zmax]

    # Rescales contour to the image axes
    R = R / calib + center_y
    Z = Z / calib + guess_tipx - 1

    aa = np.where(Z > np.max(Z_edges))
    R = np.delete(R, aa[0])
    Z = np.delete(Z, aa[0])

    return R, Z


def squared_distance(R, Z, R_edges, Z_edges):
    """
    Calculate the squared distance for half profile.

    Theoretical points are interpolated on experimental ones.

    Parameters
    ----------

    Returns
    -------

    """
    R_theo_interpolator = interp1d(Z, R, kind='linear', fill_value='extrapolate')
    R_theo_interpolated = R_theo_interpolator(Z_edges)
    return (R_theo_interpolated - R_edges)**2


def deviation_edge_model(variables, image_shape, radius, R_edges, Z_edges, tip, guess_tipx, center_x, calib):
    """
    Return the RMS for a profile given by set of parameters to the experimental profile.

    Parameters
    ----------

    Returns
    -------
    """
    R, Z = young_laplace(variables, image_shape, radius, R_edges, Z_edges, tip, guess_tipx, center_x, calib)

    # Split profiles to compute errors on each side
    R_left, Z_left, R_right, Z_right = split_profile(R, Z)
    R_edges_left, Z_edges_left, R_edges_right, Z_edges_right = split_profile(R_edges, Z_edges)

    # Error on both sides.
    e_left = squared_distance(R_left, Z_left, R_edges_left, Z_edges_left)
    e_right = squared_distance(R_right, Z_right, R_edges_right, Z_edges_right)

    # Merge errrors
    e_all = np.concatenate((e_left, e_right))
    chi_squared = np.sum(e_all)
    RMS = np.sqrt(chi_squared) / len(e_all)
    return RMS
