# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:03:15 2018

@author: miguet
"""
import numpy as np

from scipy.interpolate import interp1d
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
        phip = 2 - bond_number * z - np.sin(phi) / r
    else:
        phip = 1

    rp = np.cos(phi)
    zp = np.sin(phi)

    return phip, rp, zp


def theoretical_contour(image_shape, bond_number, tip, calib):
    """
    Compute the theoretical contour from the Young-Laplace differential equation.


    Parameters
    ----------

    Notes
    -----
    The profile is non-dimensionalized by the tip radius.

    """

    # TODO 10 is an arbitrary value. Need to check if an increase is needed... or something
    s_tilde = np.linspace(0, 10, 1e3)
    solution = odeint(young_laplace_diff_equation,
                      (0, 0, 0), s_tilde,
                      args=(bond_number,))
    # NOTE: phi = solution[:, 0]

    # TODO WARNING, non dimentionalized by lc to be consistant with Jonas
    R = solution[:, 1]
    Z = solution[:, 2]

    return R, Z


def rotate_lines(R, Z, center, theta):
    """
    Rotate with specific angle conversion for our images.
    """
    theta = - theta * np.pi / 180

    return rotate(R, Z, center[0], center[1], theta)

