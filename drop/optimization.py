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


def young_laplace(gamma, angle, center_R, center_Z, radius, R_edges, Z_edges,
                  calib, rho=1000, gravity=9.81, num_points=1e3):
    """
    Returns the Young Laplace solution resized and oriented to the image.

    Parameters
    ----------
    gamma : scalar
        Surface tension
    angle : scalar

    center_R : scalar

    center_Z : scalar

    radius : scalar

    R_edges : array
        Radial coordinates of the edge.
    Z_edges : array
        Vertical coordinates of the edge.
    calib : scalar
        Calibration in mm per px.
    rho : scalar, optional
        Fluid density.
    gravity : scalar, optional
        Gravitational acceleration.
    num_points : scalar, optional
        Number of points used in `theoretical_contour`

    Returns
    -------
    coordinates : tuple
        (R, Z)
    """
    rho_g = rho * gravity
    capillary_length = np.sqrt(gamma / rho_g)
    r0 = radius * calib
    bond_number = (r0 / capillary_length)**2

    # For the theoretical contour, the position (0, 0) corresponds to
    # the tip of the drop.
    R, Z = theoretical_contour(bond_number, calib, num_points=num_points)

    # Rescale
    R = R * r0
    Z = Z * r0

    # Symetrize the contour
    # Eliminate the first point as it is R=0, Z=0
    # To do not count it twice.
    # Then, flip the arrays to get points nicely ordered.
    R = np.concatenate((-R[1:][::-1], R))
    Z = np.concatenate((Z[1:][::-1], Z))

    # Rotate
    base_center = np.array((0, r0))
    R, Z = rotate_lines(R, Z, base_center, angle)
    R = np.array(R)
    Z = np.array(Z)

    # Rescales contour to the image axes
    R = R / calib + center_R
    Z = Z / calib + (center_Z - radius)

    # Drop the theoretical points that go beyond the latest detected pixel
    # ie outside the image
    aa = np.where(Z > np.max(Z_edges))
    R = np.delete(R, aa)
    Z = np.delete(Z, aa)

    return R, Z


def radial_squared_distance(R, Z, R_edges, Z_edges):
    """
    Calculate the radial squared distance for half profile.

    Theoretical points are interpolated on experimental ones.

    Parameters
    ----------
    R : array
        Radial coordinates of the theoretical contour.
    Z : array
        Vertical coordinates of the theoretical contour.
    R_edges : array
        Radial coordinates of the edge.
    Z_edges : array
        Vertical coordinates of the edge.

    Returns
    -------
    distance

    """
    R_theo_interpolator = interp1d(Z, R,
                                   kind='linear', fill_value='extrapolate')
    R_theo_interpolated = R_theo_interpolator(Z_edges)
    return (R_theo_interpolated - R_edges)**2


def deviation_edge_model_simple(variables, angle, center_R, center_Z, radius, R_edges, Z_edges, calib):
    """
    Return the RMS for a profile given by set of parameters to the experimental profile.

    Parameters
    ----------
    variables : tuple
        (surface tension, angle, center_R)
    radius : scalar

    R_edges : array
        Radial coordinates of the edge.
    Z_edges : array
        Vertical coordinates of the edge.
    center_Z :

    calib : scalar
        Calibration in mm per px.

    Returns
    -------
    RMS
    """
    R, Z = young_laplace(*variables, angle, center_R, center_Z, radius, R_edges, Z_edges, calib)

    # Split profiles to compute errors on each side
    R_left, Z_left, R_right, Z_right = split_profile(R, Z)
    R_edges_left, Z_edges_left, R_edges_right, Z_edges_right = split_profile(R_edges, Z_edges)

    # Error on both sides.
    e_left = radial_squared_distance(R_left, Z_left, R_edges_left, Z_edges_left)
    e_right = radial_squared_distance(R_right, Z_right, R_edges_right, Z_edges_right)

    # Merge errrors
    e_all = np.concatenate((e_left, e_right))
    chi_squared = np.sum(e_all)
    RMS = np.sqrt(chi_squared) / len(e_all)
    return RMS


def deviation_edge_model_full(variables, R_edges, Z_edges, calib):
    """
    Return the RMS for a profile given by set of parameters to the experimental profile.

    Parameters
    ----------
    variables : tuple
        (surface tension, angle, center_R)
    radius : scalar

    R_edges : array
        Radial coordinates of the edge.
    Z_edges : array
        Vertical coordinates of the edge.
    center_Z :

    calib : scalar
        Calibration in mm per px.

    Returns
    -------
    RMS
    """
    R, Z = young_laplace(*variables, R_edges, Z_edges, calib)

    # Split profiles to compute errors on each side
    R_left, Z_left, R_right, Z_right = split_profile(R, Z)
    R_edges_left, Z_edges_left, R_edges_right, Z_edges_right = split_profile(R_edges, Z_edges)

    # Error on both sides.
    e_left = radial_squared_distance(R_left, Z_left, R_edges_left, Z_edges_left)
    e_right = radial_squared_distance(R_right, Z_right, R_edges_right, Z_edges_right)

    # Merge errrors
    e_all = np.concatenate((e_left, e_right))
    chi_squared = np.sum(e_all)
    RMS = np.sqrt(chi_squared) / len(e_all)
    return RMS
