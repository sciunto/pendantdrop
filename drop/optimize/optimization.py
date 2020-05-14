# -*- coding: utf-8 -*-
import numpy as np
from scipy import constants

from .theory import rotate_lines, theoretical_contour
from .deviation import radial_RMS, orthogonal_RMS


def _drop_theo_points_outside_detection(R, Z, RZ_edges):
    """
    Remove theoretical points outside the range of detected ones.

    """
    R_edges, Z_edges = RZ_edges
    # Drop the theoretical points that go beyond the latest detected pixel
    # ie outside the image
    aa = np.where(Z > np.max(Z_edges))
    R = np.delete(R, aa)
    Z = np.delete(Z, aa)
    return R, Z


def young_laplace(surface_tension, angle, center_R, center_Z, radius, density,
                  calib, *, RZ_edges=None, gravity=None, num_points=1e3):
    """
    Returns the Young Laplace solution resized and oriented to the image.

    Parameters
    ----------
    surface_tension : scalar
        Surface tension
    angle : scalar
        Tilting angle.
    center_R : scalar
        Osculating circle center R-coordinate.
    center_Z : scalar
        Osculating circle center Z-coordinate.
    radius : scalar
        Osculating circle radius.
    density : scalar
        Fluid density.
    calib : scalar
        Calibration in mm per px.
    RZ_edges : tuple of array, optional
        (Radial, Vertical) coordinates of the edge.
        Used to crop the profile to the experimental profile.
        If None, the solution is not cropped.
    gravity : scalar, optional
        Gravitational acceleration. If None, `scipy.constants.g` is used.
    num_points : scalar, optional
        Number of points used in `theoretical_contour`

    Returns
    -------
    coordinates : tuple
        (R, Z)
    """
    if gravity is None:
        gravity = constants.g

    rho_g = density * gravity
    capillary_length = np.sqrt(surface_tension / rho_g)
    r0 = radius * calib
    bond_number = (r0 / capillary_length)**2

    # For the theoretical contour, the position (0, 0) corresponds to
    # the tip of the drop.
    R, Z = theoretical_contour(bond_number, num_points=num_points)

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

    if RZ_edges:
        R, Z = _drop_theo_points_outside_detection(R, Z, RZ_edges)
    return R, Z


def deviation_edge_model_simple(variables, angle, center_R, center_Z, radius, RZ_edges, density, calib, *, RMS=None):
    """
    Return the RMS for a profile given by set of parameters to the experimental profile.

    Parameters
    ----------
    variables : tuple
        (surface tension, angle, center_R)
    angle : scalar

    center_R :

    center_Z :

    radius :

    RZ_edges : tuple of array
        (Radial, Vertical) coordinates of the edge.
    calib : scalar
        Calibration in mm per px.
    RMS : callable
        function(R_theo, Z_theo, RZ_edges) to compute the RMS.
        If None, radial_RMS is used.

    Returns
    -------
    RMS
    """
    RZ = young_laplace(*variables, angle, center_R, center_Z, radius, density, calib, RZ_edges=RZ_edges)

    if RMS is None:
        RMS = radial_RMS

    return RMS(RZ, RZ_edges)


def deviation_edge_model_full(variables, RZ_edges, density, calib, *, RMS=None):
    """
    Return the RMS for a profile given by set of parameters to the experimental profile.

    Parameters
    ----------
    variables : tuple
        (surface tension, angle, center_R)
    RZ_edges : tuple of array
        (Radial, Vertical) coordinates of the edge.
    calib : scalar
        Calibration in mm per px.
    RMS : callable
        function(R_theo, Z_theo, RZ_edges) to compute the RMS.
        If None, radial_RMS is used.

    Returns
    -------
    RMS
    """
    RZ = young_laplace(*variables, density, calib, RZ_edges=RZ_edges)

    if RMS is None:
        RMS = radial_RMS

    return RMS(RZ, RZ_edges)
