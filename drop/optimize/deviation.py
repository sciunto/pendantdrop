#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize, root_scalar
from drop.utils import split_profile


def _vertical_squared_distance(R, Z, R_edges, Z_edges):
    """
    Calculate the vertical squared distance for half profile.

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
    Z_theo_interpolator = interp1d(R, Z, kind='linear',
                                   fill_value='extrapolate')
    Z_theo_interpolated = Z_theo_interpolator(R_edges)
    return (Z_theo_interpolated - Z_edges)**2


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


def radial_RMS(RZ_model, RZ_edges):
    """
    Calculate the radial RMS distance.

    Theoretical points are interpolated on experimental ones.

    Parameters
    ----------
    RZ_model : tuple of array
        (Radial Vertical) coordinates of the theoretical contour.
    RZ_edges : tuple of array
        (Radial, Vertical) coordinates of the edge.

    Returns
    -------
    RMS

    """
    R_edges, Z_edges = RZ_edges
    R, Z = RZ_model
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


def _shortest_squared_distance(R, Z, R_edges, Z_edges):
    """
    Calculate the shortest squared distance for half profile.

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
        :1
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


def shortest_RMS(RZ_model, RZ_edges):
    """
    Calculate the shortest RMS distance.

    Theoretical points are interpolated on experimental ones.

    Parameters
    ----------
    RZ_model : tuple of array
        (Radial Vertical) coordinates of the theoretical contour.
    RZ_edges : tuple of array
        (Radial, Vertical) coordinates of the edge.

    Returns
    -------
    RMS

    """
    R, Z = RZ_model
    R_edges, Z_edges = RZ_edges

    # Split profiles to compute errors on each side
    R_left, Z_left, R_right, Z_right = split_profile(R, Z)
    R_edges_left, Z_edges_left, R_edges_right, Z_edges_right = split_profile(R_edges, Z_edges)

    # Error on both sides.
    e_left = _shortest_squared_distance(R_left, Z_left, R_edges_left, Z_edges_left)
    e_right = _shortest_squared_distance(R_right, Z_right, R_edges_right, Z_edges_right)

    # Merge errrors
    e_all = np.concatenate((e_left, e_right))
    chi_squared = np.sum(e_all)
    return np.sqrt(chi_squared) / len(e_all)


def _orthogonal_squared_distance(R, Z, R_edges, Z_edges):
    """
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
    """
    # Slope
    # ( R(i+1) - R(i-1) ) / (Z(i+1) - Z(i-1))
    num = np.diff(R_edges[:-1]) + np.diff(R_edges[1:])
    denom = np.diff(Z_edges[:-1]) + np.diff(Z_edges[1:])

    R_theo_interpolator = interp1d(Z, R, kind='linear',
                                   fill_value='extrapolate')

    def f_to_minimize(z, R_edge, Z_edge, slope):
        return R_theo_interpolator(z) + slope * (z - Z_edge) - R_edge

    def get_distance(el, max_distance):
        r, z, num, denom = el
        if denom == 0:
            # Return zero distance, we cant evaluate
            return 0
        slope = num / denom
        print('test')
        try:
            res = root_scalar(f_to_minimize,
                          bracket=(0, 30000),
                          method='bisect',
                          args=(r, z, slope))
        except ValueError:
            print(r, z, num, denom)
            print('oops')
            return 0
        Z_theo = res.root
        R_theo = R_theo_interpolator(Z_theo)
        dist2 = (Z_theo - z)**2 + (R_theo - r)**2
        return dist2


	# TODO provide a guess for max_distance
    # Do not consider first and last elements of the data
    # because we don't have the slope.
    max_distance = 2000
    all_dist = [get_distance(el, max_distance)  for el in zip(R_edges[1:-1], Z_edges[1:-1], num, denom)]
    return np.array(all_dist)


def orthogonal_RMS(RZ_model, RZ_edges):
    """
    Calculate the orthogonal RMS distance.

    Parameters
    ----------
    RZ_model : tuple of array
        (Radial Vertical) coordinates of the theoretical contour.
    RZ_edges : tuple of array
        (Radial, Vertical) coordinates of the edge.

    Returns
    -------
    RMS

    """
    R, Z = RZ_model
    R_edges, Z_edges = RZ_edges

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


def o_RMS(RZ_model, RZ_edges):
    """
    Calculate the orthogonal RMS distance.

    Parameters
    ----------
    RZ_model : tuple of array
        (Radial Vertical) coordinates of the theoretical contour.
    RZ_edges : tuple of array
        (Radial, Vertical) coordinates of the edge.

    Returns
    -------
    RMS

    """
    R, Z = RZ_model
    R_edges, Z_edges = RZ_edges

    # Split profiles to compute errors on each side
    R_left, Z_left, R_right, Z_right = split_profile(R, Z)
    R_edges_left, Z_edges_left, R_edges_right, Z_edges_right = split_profile(R_edges, Z_edges)

    # Error on both sides.
    re_left = _radial_squared_distance(R_left, Z_left, R_edges_left, Z_edges_left)
    re_right = _radial_squared_distance(R_right, Z_right, R_edges_right, Z_edges_right)
    oe_left = _vertical_squared_distance(R_left, Z_left, R_edges_left, Z_edges_left)
    oe_right = _vertical_squared_distance(R_right, Z_right, R_edges_right, Z_edges_right)

    e_left = 0.5 * (re_left + oe_left)
    e_right = 0.5 * (re_right + oe_right)

    # Merge errrors
    e_all = np.concatenate((e_left, e_right))
    chi_squared = np.sum(e_all)
    return np.sqrt(chi_squared) / len(e_all)

