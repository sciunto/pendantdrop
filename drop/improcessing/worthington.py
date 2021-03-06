# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import constants
from skimage import measure
from skimage.morphology import flood_fill

__all__ = ['worthington',
           ]


def _base_diameter(edges):
    """
    Find the base diameter (in px)

    Parameters
    ----------
    edges : boolean image
        Image containing the edges to fit.
    """
    # assume baseline at the bottom
    position = -1
    while np.sum(edges[position, :]) < 2:
        position -= 1

    pixels_on_baseline = np.where(edges[position, :] == True)
    return pixels_on_baseline[0][-1] - pixels_on_baseline[0][0]


def _drop_volume(edges):
    """
    Compute the drop volume (in px3)

    Parameters
    ----------
    edges : boolean image
        Image containing the edges to fit.

    Returns
    -------
    volume : scalar
    """
    labeled_edges = measure.label(edges)
    
    # Fill the edges
    # See https://github.com/scikit-image/scikit-image/issues/4944
    # filled_area is not reliable, so we fill the shape prevently
    # Assume that the centroid is part of the shape.
    prop = measure.regionprops_table(labeled_edges, properties=('centroid', ))
    seed = (int(prop['centroid-0']), int(prop['centroid-1']))
    filled = flood_fill(edges, seed, edges.max(), connectivity=1)

    # Take the largest area in case
    # spurious objects were detected

    labeled_edges = measure.label(filled)
    table = measure.regionprops_table(labeled_edges,
                                      properties=('area', 'centroid'))
    df = pd.DataFrame(table)
    df = df[df.area == df.area.max()]
    area = df.area.values

    # Split right and left
    edges_left = filled[:, :int(df['centroid-1'][0])]
    edges_right = filled[:, int(df['centroid-1'][0]):]

    # Left part
    labeled_edges = measure.label(edges_left)

    table = measure.regionprops_table(labeled_edges, properties=('centroid',))
    df = pd.DataFrame(table)

    distance_left = edges_left.shape[1] - df['centroid-1'].values

    # right part
    labeled_edges = measure.label(edges_right)

    table = measure.regionprops_table(labeled_edges, properties=('centroid',))
    df = pd.DataFrame(table)

    distance_right = df['centroid-1'].values

    # Compute the volume
    average_distance = (distance_right[0] + distance_left[0]) / 2.
    volume_in_px3 = 2 * np.pi * average_distance * area[0] / 2.
    return volume_in_px3


def worthington(edges, calib, surface_tension, fluid_density, gravity=None):
    """
    Compute the worthington number.

    Parameters
    ----------
    edges : boolean image
        Image containing the edges to fit.
    calib : scalar
        Calibration scale in m/px.
    surface_tension : scalar
        Surface tension in N/m.
    fluid_density : scalar,
        Fluid density in kg/m3.
    gravity : scalar, optional
        Gravitational acceleration. If None, `scipy.constants.g` is used.

    Returns
    -------
    worthington : scalar

    References
    ----------

    .. [1] Berry, Neeson, Dagastine, Chan, Tabor "Measurement of surface and 
           interfacial tension using pendant drop tensiometry"
           Journal of Colloid and Interface Science, 2015, 454, 226-237
           :DOI:`10.1016/j.jcis.2015.05.012`
    """
    if gravity is None:
        gravity = constants.g
    volume = _drop_volume(edges) * calib**3

    needle_diameter = _base_diameter(edges) * calib

    num = fluid_density * gravity * volume
    den = np.pi * surface_tension * needle_diameter
    return num / den
