# -*- coding: utf-8 -*-
import numpy as np

from drop.theory import rotate_lines, theoretical_contour
from drop.deviation import radial_RMS, orthogonal_RMS


def young_laplace(surface_tension, angle, center_R, center_Z, radius, R_edges, Z_edges,
                  calib, rho=1000, gravity=9.81, num_points=1e3):
    """
    Returns the Young Laplace solution resized and oriented to the image.

    Parameters
    ----------
    surface_tension : scalar
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
    capillary_length = np.sqrt(surface_tension / rho_g)
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


def deviation_edge_model_simple(variables, angle, center_R, center_Z, radius, R_edges, Z_edges, calib, RMS=None):
    """
    Return the RMS for a profile given by set of parameters to the experimental profile.

    Parameters
    ----------
    variables : tuple
        (surface tension, angle, center_R)
    radius : scalar

    center_R :

    center_Z :

    radius :

    R_edges : array
        Radial coordinates of the edge.
    Z_edges : array
        Vertical coordinates of the edge.
    calib : scalar
        Calibration in mm per px.
    RMS : callable
        function(R_theo, Z_theo, R_edges, Z_edges) to compute the RMS.
        If None, radial_RMS is used.

    Returns
    -------
    RMS
    """
    R, Z = young_laplace(*variables, angle, center_R, center_Z, radius, R_edges, Z_edges, calib)

    if RMS is None:
        RMS = radial_RMS

    return RMS(R, Z, R_edges, Z_edges)


def deviation_edge_model_full(variables, R_edges, Z_edges, calib, RMS=None):
    """
    Return the RMS for a profile given by set of parameters to the experimental profile.

    Parameters
    ----------
    variables : tuple
        (surface tension, angle, center_R)
    R_edges : array
        Radial coordinates of the edge.
    Z_edges : array
        Vertical coordinates of the edge.
    calib : scalar
        Calibration in mm per px.
    RMS : callable
        function(R_theo, Z_theo, R_edges, Z_edges) to compute the RMS.
        If None, radial_RMS is used.

    Returns
    -------
    RMS
    """
    R, Z = young_laplace(*variables, R_edges, Z_edges, calib)

    if RMS is None:
        RMS = radial_RMS

    return RMS(R, Z, R_edges, Z_edges)
