# -*- coding: utf-8 -*-
import numpy as np
import math

from scipy.integrate import solve_ivp
from joblib import Memory
from drop.utils import rotate
from appdirs import AppDirs

dirs = AppDirs('pendantdrop')
cachedir = dirs.user_cache_dir
memory = Memory(cachedir, verbose=0)


def young_laplace_diff_equation(space, variables, bond_number):
    """
    Return the derivatives corresponding to the Young-Laplace equation.

    Parameters
    ----------
    space : 1D-array
        Space variable.
    variables : tuple
        (phi, r_tilde, z_tilde)
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

    .. [1] Del Rıo, O. I., and A. W. Neumann. "Axisymmetric drop shape analysis:
           computational methods for the measurement of interfacial properties
           from the shape and dimensions of pendant and sessile drops."
           Journal of colloid and interface science 196.2 (1997): 136-147.
           :DOI:`10.1006/jcis.1997.5214`
    """
    phi, r, z = variables

    sin_phi = math.sin(phi)

    if r != 0:
        phi_prime = 2 - bond_number * z - sin_phi / r
    else:
        phi_prime = 1

    r_prime = math.cos(phi)
    z_prime = sin_phi

    return phi_prime, r_prime, z_prime

@memory.cache
def theoretical_contour(bond_number, num_points=1e3, s_max=10):
    """
    Compute a theoretical contour from the Young-Laplace differential equation.


    Parameters
    ----------
    bond_number : scalar
        Bond number.
    num_points : scalar, optional
        Number of points used to compute the profile. These points are
        evenly spaces from s=0 to s=s_max.
    s_max : scalar, optional
        Maximum value for the curvilinear coordinate.

    Returns
    -------
    (R, Z) : tuple
        R and Z coordinates.

    Notes
    -----
    The profile is non-dimensionalized by the tip radius.
    The resolution is achieved with `scipy.integrate.solve_ivp`

    """
    s_tilde = np.linspace(0, s_max, int(num_points))

    # Initial parameters
    phi_prime_0 = 0
    r_prime_0 = 0
    z_prime_0 = 0
    init = (phi_prime_0, r_prime_0, z_prime_0)

    #from scipy.integrate import odeint
    #solution = odeint(young_laplace_diff_equation,
    #                  init, s_tilde,
    #                  args=(bond_number,),
    #                  tfirst=True)
    # NOTE: phi = solution[:, 0]
    # R = solution[:, 1]
    # Z = solution[:, 2]

    solution = solve_ivp(young_laplace_diff_equation,
                         (0, s_max),
                         init,
                         t_eval=s_tilde,
                         args=(bond_number,))

    # Note: phi = solution.y[0]
    R = solution.y[1]
    Z = solution.y[2]

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
