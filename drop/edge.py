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

def find_circle(edges, hough_radii):
    """
    Find the best circle to model the drop tip.


    """


    hough_res = hough_circle(edges, hough_radii, full_output=True)

    centers = []
    accums = []
    radii = []

    for radius, h in zip(hough_radii, hough_res):
        # For each radius, extract two circles
        peaks = peak_local_max(h, num_peaks=2)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius, radius])

    idx = np.argsort(accums)[::-1][0]

    center_x, center_y = centers[idx]
    radius = radii[idx]


    center_x = center_x - hough_radii[-1]
    center_y = center_y - hough_radii[-1]
    tip = [center_y, center_x - radius]

    return center_x, center_y, radius,tip


def detect_edges(image):
    """
    detect the edge of the drop on the image.
    """
    # Use the mean grayscale value of the image to get the contour.
    level = image.mean()
    contours = measure.find_contours(image, level)
    Z_edges, R_edges = np.column_stack(contours[0])
    # Make a binary image
    edges = np.full(image.shape, False, dtype=bool)
    edges[Z_edges.astype(np.int), R_edges.astype(np.int)] = True
    return edges,  R_edges, Z_edges
