#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 10:53:58 2018

@author: fr
"""

import numpy as np
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage import measure


def _find_circle(edges, hough_radii):
    """
    Find the best circle to model points marked as `True`.

    Parameters
    ----------
    edges : boolean image
        Image containing the edges to fit.
    hough_radii : tuple
        Radii considered for the Hough transform

    Returns
    -------
    parameters : tuple
        (center_x, center_y, radius, tip_position)

    """


    hough_res = hough_circle(edges, hough_radii, full_output=True)

    centers = []
    accums = []
    radii = []

    for radius, h in zip(hough_radii, hough_res):
        # For each radius, extract two circles
        peaks = peak_local_max(h, min_distance=100, num_peaks=2)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius, radius])

    idx = np.argsort(accums)[::-1][0]

    center_x, center_y = centers[idx]
    radius = radii[idx]

    center_x = center_x - hough_radii[-1]
    center_y = center_y - hough_radii[-1]
    tip = [center_y, center_x - radius]

    return center_x, center_y, radius, tip



def _fit_circle_tip_hough_transform(shape, R, Z):
    """
    Fit a circle with a Hough transform on the points between the equator
    and the tip.


    Returns
    -------
    parameters : tuple
        (center_x, center_y, radius, tip_position)
    """

    # Assume upward bubble orientation
    mask = Z < Z.min() + 0.5 * (Z[R.argmin()] - Z.min())
    edges = np.full(shape, False, dtype=bool)
    edges[Z[mask].astype(np.int), R[mask].astype(np.int)] = True

    # Guess the maximum radius
    max_possible_radius = .5 * (R.max() - R.min())
    min_possible_radius = int(0.8 * max_possible_radius)
    # Coarse grain
    step = 5
    hough_radii = np.arange(min_possible_radius, max_possible_radius, step)
    _, _, radius, _ = _find_circle(edges, hough_radii)
    # Fine grain
    hough_radii = np.arange(radius - 2 * step, radius + 2 * step, 1)
    return _find_circle(edges, hough_radii)



def _fit_circle_tip_ransac(shape, R, Z):


    points = np.column_stack((R, Z))

    model_robust, inliers = measure.ransac(points, measure.CircleModel,
                                           min_samples=5,
                                           residual_threshold=.1,
                                           max_trials=1000)

    cy, cx, r = model_robust.params
    tip = [cy, cx - r]

    return cx, cy, r, tip


def fit_circle_tip(shape, R, Z, method='ransac'):
    """

    """
    if method.lower() == 'ransac':
        return _fit_circle_tip_ransac(shape, R, Z)
    elif method.lower() == 'hough':
        return _fit_circle_tip_hough_transform(shape, R, Z)
    else:
        raise ValueError('Wrong parameter value for `method`.')


def detect_edges(image):
    """
    Detect the edge of the drop on the image.

    Parameters
    ----------
    image : ndarray
        Grayscale image.
    """
    # Use the mean grayscale value of the image to get the contour.
    level = image.mean()
    contours = measure.find_contours(image, level)
    Z_edges, R_edges = np.column_stack(contours[0])
    # Make a binary image
    edges = np.full(image.shape, False, dtype=bool)
    edges[Z_edges.astype(np.int), R_edges.astype(np.int)] = True
    return edges, R_edges, Z_edges
