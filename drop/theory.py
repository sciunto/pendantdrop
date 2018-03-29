# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:03:15 2018

@author: miguet
"""
import numpy as np
import math

from scipy.interpolate import interp1d
from scipy.integrate import odeint

def theoretical_contour(image_shape, lc, r0, tip, calib):
    r_st = r0 / lc

    Z0 = image_shape[0] - tip[1]  #image.shape[0]-50##Position de Z0 w/r x (coordonnée verticale)
    Zmax = Z0 / lc * calib #maximum possible values of Z to be upgraded
    nPoints = Z0 * 2
    ds = Zmax / nPoints
    #####Bouadary for validity of sperical approx
    if r_st <1:
        sLimit=r_st*0.2
    else:
        sLimit=.2

    s=0
    R=[]
    Z=[]
    while s<sLimit:
        R.append(r_st*np.sin(s/r_st)+s**5/(40*r_st**2))
        Z.append(1/(2*r_st)*s**(2)*(1-(0.75+1/(12*r_st**2))*s**2))
        s+=ds


    ####Fonction dérivation des paramètres:
    def deriv(variables):
        k=[]
        k.append(math.cos(variables[1][-1]))#dérivée
        k.append(variables[2][-1])
        k.append(-np.sin(variables[1][-1])+np.cos(variables[1][-1])*(np.sin(variables[1][-1])/variables[0][-1]-variables[2][-1])/variables[0][-1])
        k.append(np.sin(variables[1][-1]))
        return k
    #
    ###initial conditions with approximate solution:
    nVar=4
    variables=[[0] for i in range(nVar)]
    tmp=[[] for i in range(nVar)]

    variables[0][0]=r_st*np.sin(s/r_st)+s**5/(40*r_st**2)
    variables[1][0]=s*(1-0.125*s**2)/r_st
    variables[2][0]=(1-0.375*s**2)/r_st
    variables[3][0]=1/(2*r_st)*s**(2)*(1-(0.75+1/(12*r_st**2))*s**2)
    Zlocal=0
    while Zlocal<image_shape[0]-1:
#    while variables[3][-1]<Zmax:
        k1=deriv(variables)
        for i in range(nVar):
            tmp[i]=[variables[i][-1]+0.5*ds*k1[i]]
        k2=deriv(tmp)
        for i in range(nVar):
            tmp[i]=[variables[i][-1]+0.5*ds*k2[i]]
        k3=deriv(tmp)
        for i in range(nVar):
            tmp[i]=[variables[i][-1]+0.5*ds*k3[i]]
        k4=deriv(tmp)
        for i in range(nVar):
            variables[i].append(variables[i][-1]+ds*(k1[i]+2*(k2[i]+k3[i])+k4[i]))
        Zlocal=lc/calib*variables[3][-1]+tip[1]#+ax.get_ylim()[1]
#        Zlocal=lc/calib*variables[3][-1]+tip[1]-10

    R.extend(variables[0][:])
    Z.extend(variables[3][:])
    return np.array(R), np.array(Z)


def younglaplace_diff_equation(variables, space, bond):
    """

    variables : tuple
        (phi, r_tilde, z_tilde)


    Notes
    -----
    tilde means non-dimensionalized by the tip radius.

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


def theoretical_contour2(image_shape, lc, r0, tip, calib):
    bond = (r0 / lc)**2

    # TODO 10 is an arbitrary value. Need to check if an increase is needed... or something
    s_tilde = np.linspace(0, 10, 1e3)
    solution = odeint(younglaplace_diff_equation, (0, 0, 0), s_tilde, args=(bond,))
    #phi = solution[:, 0]

    # TODO WARNING, non dimentionalized by lc to be consistant with Jonas
    R = solution[:, 1] * r0 / lc
    Z = solution[:, 2] * r0 / lc

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



