# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:03:15 2018

@author: miguet
"""
import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import odeint



def young_laplace(variables, image_shape, radius, R_edges, Z_edges, tip, guess_tipx, center_x):

    gamma = variables[0]
    theta = variables[1]
    center_y = variables[2]


    calib = 0.00124 / 400  #400 pixel = 1.24mm
    rho_g = 1000 * 9.81
    lc = np.sqrt(gamma / rho_g)  # We give capillary lengthy : may be given by the user later on
    r0 = radius * calib


    tip_x = guess_tipx

    print(center_y)

    base_center = [center_y, center_x]

    R, Z = theoretical_contour(image_shape, lc, r0, tip, calib)

    R = R * r0
    Z = Z * r0

    # Cut
    Z0 = image_shape[0] - tip[1]
    Zmax = Z0 * calib #maximum possible values of Z to be upgraded
    R = R[Z < Zmax]
    Z = Z[Z < Zmax]

    # Symetrize the contour
    R = np.concatenate((-R, R))
    Z = np.concatenate((Z, Z))

    # rescales contour to the image axes
    R = R / calib + center_y
    Z = Z / calib + tip_x - 1

    # Rotate
    R, Z = rotate_lines(R, Z, base_center, theta)


    aa = np.where(Z>max(Z_edges))
    R = np.delete(R, aa[0])
    Z = np.delete(Z, aa[0])

    return R, Z



def younglaplace_diff_equation(variables, space, bond):
    """
    Return the derivatives corresponding to the Young-Laplace equation.

    Parameters
    ----------
    variables : tuple
        (phi, r_tilde, z_tilde)
    space : 1D-array
        Space variable.
    bond : scalar
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
    See Axisymmetric Drop Shape Analysis: Computational Methods
    for the Measurement of Interfacial Properties from the Shape
    and Dimensions of Pendant and Sessile Drops
    """
    phi, r, z = variables

    if r != 0:
        phip = 2 - bond * z - np.sin(phi) / r
    else:
        phip = 1

    rp = np.cos(phi)
    zp = np.sin(phi)

    return phip, rp, zp


def theoretical_contour(image_shape, lc, r0, tip, calib):
    """
    Compute the theoretical contour.


    """
    bond = (r0 / lc)**2

    # TODO 10 is an arbitrary value. Need to check if an increase is needed... or something
    s_tilde = np.linspace(0, 10, 1e3)
    solution = odeint(younglaplace_diff_equation, (0, 0, 0), s_tilde, args=(bond,))
    #phi = solution[:, 0]

    # TODO WARNING, non dimentionalized by lc to be consistant with Jonas
    R = solution[:, 1]
    Z = solution[:, 2]

    return R, Z




def rotate_lines(R,Z, center, theta):
    from math import sin, cos#, radians
    """ Rotate self.polylines the given angle about their centers. """
#    theta = radians(deg)  # Convert angle from degrees to radians
    theta=-theta*np.pi/180##sombre histoire de convention...
    cosang, sinang = cos(theta), sin(theta)

#    for pl in self.polylines:
#        # Find logical center (avg x and avg y) of entire polyline
#        n = len(pl.lines)*2  # Total number of points in polyline
    cx = center[0]
    cy = center[1]
    R_rot=[]
    Z_rot=[]
    for i in range(len(R)):
       xr=R[i]-cx
       yr=Z[i]-cy

       nx=( xr*cosang - yr*sinang) + cx
       ny=( xr*sinang + yr*cosang) + cy

#        nx=(R[i]+Z[i]-cx-cy)/(2*cosang)+cx
#        ny=-(R[i]-Z[i]-cx+cy)/(2*sinang)+cy
#
       R_rot.append(nx)
       Z_rot.append(ny)
    return R_rot,Z_rot


#def partial_ksi(theorique, python):
#    ####calculate the location on the python contour that minimizes the distance to the theoretical one
#    dist=[]
##    m=0
##    while not dist:
##        m+=1
##        z_inds=np.where(np.abs(python[1]-theorique[1])<m)[0]
##        for i in z_inds:
##            dist.append(abs(python[0][i]-theorique[0])**2)
##    ind=np.where(dist==min(dist))[0][0]
#    m=0
#    while not dist:
#        m+=1
#        z_inds=np.where(np.abs(python[1]-theorique[1])<m)[0]
#        for i in z_inds:
#            dist.append(((python[0][i]-theorique[0])**2+(python[1][i]-theorique[1])**2)**(1/2))
#    ind=np.where(dist==min(dist))[0][0]
#    return [python[1][z_inds[ind]], python[0][z_inds[ind]]], min(dist)
#



