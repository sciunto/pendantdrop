#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:11:22 2018

@author: fr
"""
import numpy as np


def orthogonal_norm(data, curve):
    """
    Calculate the orthogonal norm between data points and a curve.

    The assumption of a dense representation of the curve is made and is
    crucial for the accuracy of the norm.

    Parameters
    ----------
    data : array
        Data points.
    curve : array
        Points representing the curve.

    Returns
    -------
    norm : scalar
        L2 orthogonal norm.
    """
    norm = 0
    for dat in data:
        norm_fist_point = [np.linalg.norm(el - dat) for el in curve]
        idx = np.argmin(norm_fist_point)
        norm += norm_fist_point[idx]
    return norm


def split_profile(R, Z):
    """
    Split a profile in two parts to get a single value for each Z.

    Parameters
    ----------
    R : array
        Radial coordinates.
    Z : array
        Vertical coordinates.
    """
    # Assumption on bubble upward orientation
    mask_left = R < R[Z.argmin()]
    R_left = R[mask_left]
    Z_left = Z[mask_left]
    R_right = R[~mask_left]
    Z_right = Z[~mask_left]
    return R_left, Z_left, R_right, Z_right


def rotate(x, y, cx, cy, angle):
    """
    Rotations of points.

    Parameters
    ----------
    x : array
        x-coordinates.
    y : array
        y-coordinates.
    cx : scalar
        x-center coordinate.
    cy : scalar
        y-center coordinate.
    angle : scalar
        Rotation angle in radian.

    Returns
    -------
    result : tuple
        (x_new, y_new)
    """
    vector = np.array((x - cx, y - cy))
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.matrix([[c, s],
                                 [-s, c]])
    m = np.dot(vector.T, rotation_matrix)

    return m.T[0] + cx, m.T[1] + cy
