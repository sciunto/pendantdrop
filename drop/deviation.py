#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from drop.utils import split_profile


def _radial_squared_distance(R, Z, R_edges, Z_edges):
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
    squared_distance

    """
    R_theo_interpolator = interp1d(Z, R, kind='linear',
                                   fill_value='extrapolate')
    R_theo_interpolated = R_theo_interpolator(Z_edges)
    return (R_theo_interpolated - R_edges)**2


def radial_RMS(R, Z, R_edges, Z_edges):
    """
    Calculate the radial RMS distance.

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
    RMS

    """
    # Split profiles to compute errors on each side
    R_left, Z_left, R_right, Z_right = split_profile(R, Z)
    R_edges_left, Z_edges_left, R_edges_right, Z_edges_right = split_profile(R_edges, Z_edges)

    # Error on both sides.
    e_left = _radial_squared_distance(R_left, Z_left, R_edges_left, Z_edges_left)
    e_right = _radial_squared_distance(R_right, Z_right, R_edges_right, Z_edges_right)

    # Merge errrors
    e_all = np.concatenate((e_left, e_right))
    chi_squared = np.sum(e_all)
    return np.sqrt(chi_squared) / len(e_all)


def _orthogonal_squared_distance(R, Z, R_edges, Z_edges):
    """
    Calculate the orthogonal squared distance for half profile.

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
    squared_distance
    """
    def distance(z, R_edge, Z_edge):
        return (R_theo_interpolator(z) - R_edge)**2 + (Z_edge - z)**2

    def get_distance(el):
        r, z = el
        res = minimize(distance, z, args=(r, z), method=None)
        return distance(res.x[0], r, z)

    R_theo_interpolator = interp1d(Z, R, kind='linear',
                                   fill_value='extrapolate')
    all_dist = [get_distance(el) for el in np.column_stack((R_edges, Z_edges))]

    return np.array(all_dist)


def orthogonal_RMS(R, Z, R_edges, Z_edges):
    """
    Calculate the orthogonal RMS distance.

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
    RMS

    """
    # Split profiles to compute errors on each side
    R_left, Z_left, R_right, Z_right = split_profile(R, Z)
    R_edges_left, Z_edges_left, R_edges_right, Z_edges_right = split_profile(R_edges, Z_edges)

    # Error on both sides.
    e_left = _orthogonal_squared_distance(R_left, Z_left, R_edges_left, Z_edges_left)
    e_right = _orthogonal_squared_distance(R_right, Z_right, R_edges_right, Z_edges_right)

    # Merge errrors
    e_all = np.concatenate((e_left, e_right))
    chi_squared = np.sum(e_all)
    return np.sqrt(chi_squared) / len(e_all)