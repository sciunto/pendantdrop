#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:11:22 2018

@author: fr
"""
import numpy as np


def split_profile(R, Z):
    """
    Split a profile in two parts to get a single value for each Z.
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

    return  m.T[0] + cx, m.T[1] + cy
