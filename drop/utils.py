# -*- coding: utf-8 -*-
import numpy as np


def print_parameters(title, surface_tension, tilt_angle,
                     center_R, center_Z, radius,
                     RMS=None):
    print('')
    print('#' * 20)
    print(f'{title}')
    print('#' * 20)
    print(f'Surface tension: {surface_tension:.4} N/m')
    print(f'Tilt angle: {tilt_angle:.2}')
    print(f'Osculating circle center: {center_R:.4}, {center_Z:.4}')
    print(f'Osculating circle radius: {radius:.4}')
    if RMS is not None:
        print(f'RMS: {RMS:.3}')
    print('')


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
