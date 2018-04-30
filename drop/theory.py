# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:03:15 2018

@author: miguet
"""
import numpy as np

from scipy.integrate import odeint
from drop.utils import rotate


def young_laplace_diff_equation(variables, space, bond_number):
    """
    Return the derivatives corresponding to the Young-Laplace equation.

    Parameters
    ----------
    variables : tuple
        (phi, r_tilde, z_tilde)
    space : 1D-array
        Space variable.
    bond_number : scalar
        Bond number.

    Returns
    -------
    derivatives : tuple
        (d phi / d s_tilde, d r_tilde / d s_tilde,  d z_tilde / d s_tilde )

    Notes
    -----
    tilde means non-dimensionalized by the tip radius.

    References
    ----------
    [1] https://doi.org/10.1006/jcis.1997.5214
    """
    phi, r, z = variables

    if r != 0:
        phi_prime = 2 - bond_number * z - np.sin(phi) / r
    else:
        phi_prime = 1

    r_prime = np.cos(phi)
    z_prime = np.sin(phi)

    return phi_prime, r_prime, z_prime


def theoretical_contour(bond_number, calib):
    """
    Compute the theoretical contour from the Young-Laplace differential equation.


    Parameters
    ----------


    Returns
    -------

    Notes
    -----
    The profile is non-dimensionalized by the tip radius.

    """
    s_max = 10
    num_points = 1e4
    # TODO s_max is an arbitrary value.
    # Need to check if an increase is needed... or something.
    s_tilde = np.linspace(0, s_max, num_points)
    solution = odeint(young_laplace_diff_equation,
                      (0, 0, 0), s_tilde,
                      args=(bond_number,))
    # NOTE: phi = solution[:, 0]

    R = solution[:, 1]
    Z = solution[:, 2]

    return R, Z


def rotate_lines(R, Z, center, theta):
    """
    Rotate with specific angle conversion for our images.

    Parameters
    ----------
    R : array
        Radial coordinates.
    Z : array
        Vertical coordinates.
    center : tuple
        Rotation center coordinates.
    theta : scalar
        Rotation angle.

    Returns
    -------


    """
    theta = - theta
    return rotate(R, Z, center[0], center[1], theta)
