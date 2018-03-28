# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:15:31 2018

@author: miguet
"""


from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt


from skimage import io
from skimage.draw import circle, circle_perimeter
from skimage import feature
from skimage.transform import rotate


from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.optimize import fmin_powell

from pendant_drop_functions import find_circle,\
                                   theoretical_contour,\
                                   rotate_lines,\
                                   partial_ksi,\
                                   error_calculation,\
                                   error_calculation_2




def young_laplace(variables, image_shape):#,radius,R_python,Z_python):


    #    gamma=variables[0]
#    theta=variables[1][0]
#    center_yb=variables[1][1]

    gamma=variables[0]
    theta=variables[1]
    center_y=variables[2]

#    base_center=[(base_line[-1]+base_line[0])/2,image1.shape[0]-1]

#    if center_y>base_center[0] and theta>0:
#    if theta<0:
#        guess_tipy=(image.shape[0]-1-tip[1])*np.tan(abs(theta)*np.pi/180)+base_center[0]
#    else:
#        guess_tipy=-(image.shape[0]-1-tip[1])*np.tan(abs(theta)*np.pi/180)+base_center[0]

#    center_y=583.5
#    center_y=guess_tipy
#    center_y=562
    tip_x=guess_tipx

    print(center_y)

    base_center=[center_y,center_x]
#    base_center=[center_yb,tip_x]

    calib=0.00124/400#400 pixel = 1.24mm
    rho_g=9810#1000*9.81
    lc=(gamma/rho_g)**(1/2)##We give capillary lengthy : may be given by the user later on

#    print(radius)
#    R,Z=theoretical_contour(image,lc,radius,tip)
    R, Z, tak = theoretical_contour(image_shape, lc, radius, tip)


#    R=R[tak:]
#    Z=Z[tak:]
#    base_center=[0,Z[-1]]
    RPixImage=-np.flip(R,0)
    R=np.concatenate((RPixImage[1:],R),0)
    Z=np.concatenate((np.flip(Z[1:],0),Z),0)

#    R,Z=rotate_lines(R,Z,base_center,theta)

    ######rescales contour to the image axes
#    R=np.array(R)*lc/calib+center_yb
    R=np.array(R)*lc/calib+center_y
    Z=lc/calib*np.array(Z)+tip_x-1

    R,Z=rotate_lines(R,Z,base_center,theta)


    aa=np.where(Z>max(Z_python))

    R=np.delete(R,aa[0])
    Z=np.delete(Z,aa[0])


    ####calculate fit error w/r to python profile

    python=[R_python,Z_python]

    ksi_z,kk,mini_inds,RMSd=error_calculation(R,Z,python)
#    ksi_z,kk,mini_inds,RMSd=error_calculation_2(R,Z,R_python,Z_python,center_y)
#    print(tak)
#    ksi_z,kk,mini_inds,RMSd=error_calculation_2(R[tak:],Z[tak:],R_python,Z_python,center_y)
    return ksi_z,R_python,Z_python,R,Z,mini_inds,RMSd
#    return ksi_z,R_python,Z_python,R,Z,RMSd


def error_f(variables, image_shape):
    print(variables)
    error,R_python,Z_python,R,Z,mini_inds,RMSd = young_laplace(variables, image_shape)
#    error,R_python,Z_python,R,Z,RMSd=young_laplace(variables)
    return RMSd




def load_image(path, region=None):
    """
    Load an image.
    
    Parameters
    ----------
    path : string
        File path.
    region : tuple, optional
        Corner positions to crop the image.
    
    """
    image = io.imread(image_path, as_grey=True)

    if region is not None:
        image = image[region[0][0]:region[0][1],
                      region[1][0]:region[1][1]]
    return image


if __name__ == '__main__':

    image_path = 'uEye_Image_000827.bmp'
    zoom = ([100,1312], [800,1900])
    
    
    
    #image_path = 'uEye_Image_002767.bmp'
    
    
    image1 = load_image(image_path, region=zoom)
    
    edges = feature.canny(image1, sigma=2.5)
    
    hough_radii = np.arange(418, 440)
    
    center_x, center_y, radius, tip = find_circle(edges, hough_radii)
    
    rr,cc = circle(center_x, center_y, radius-5)
    image1[rr, cc] = 10

    
    Z_python=np.where(edges==True)[0]
    R_python=np.where(edges==True)[1]
    
    
    
    ###arbitrary first guess for gamma
    gamma0 = 0.040 # N/m
    
    #####guess gravity angle
#    c_center = [center_y, center_x]
#    base_line = np.where(image1[image1.shape[0]-1,:]<5)[0]
#    base_center = [(base_line[-1]+base_line[0])/2,image1.shape[0]-1]
#    hyp = np.sqrt(abs(base_center[0]-tip[0])**2+abs(base_center[1]-tip[1])**2)
#    adj = image1.shape[0]-1-tip[1]
#    theta = -np.arccos(adj/hyp) * 180 / np.pi
    
    c_center = [center_y,center_x]
    base_line = np.where(image1[image1.shape[0]-1,:]<5)[0]
    base_center = [(base_line[-1]+base_line[0])/2,image1.shape[0]-1]
    hyp = np.sqrt((base_center[0]-c_center[0])**2+(base_center[1]-c_center[1])**2)
    adj = image1.shape[0]-1-c_center[1]
    opp = abs(base_center[0]-c_center[0])
    
    theta = np.arccos(adj/hyp) * 180 / np.pi
    
    if center_y>base_center[0] and theta>0:
        guess_tipy=(edges.shape[0]-1-tip[1])*np.tan(abs(theta)*np.pi/180)+base_center[0]
    else:
        guess_tipy=-(edges.shape[0]-1-tip[1])*np.tan(abs(theta)*np.pi/180)+base_center[0]
    #guess_tipy=tip[0]
    #guess_tipx=Z[np.where(abs(np.array(R)-guess_tipy)==min(abs(np.array(R)-guess_tipy)))[0][0]]
    #guess_cy=base_center[0]
    
    ind_min = np.where(abs(np.array(R_python)-guess_tipy)==min(abs(np.array(R_python)-guess_tipy)))[0][0]
    guess_tipx = Z_python[ind_min]
    #theta=-2.06
    
    tipx=tip[1]
    tipy=tip[0]
    guess_cy=584
    
    
    initial_gammas=np.divide([-.02,.02,-.02,.02],10)#,-.005,.005,-.001,.001],10)
    #initial_radii=np.divide([-15,-15,15,15],10)#,-5,5,-1,1],10)
    initial_thetas=np.divide([-.02,-.02,.02,.02],5)*180/np.pi#,-.005,.005,-.001,.001],10),0)*180/np.pi#[-.001,.001]#[-.001,.001]#
    #initial_gammas=[.001,.002,.003,.004,.005]
    #initial_thetas=[1/radius*180/np.pi,1/radius*180/np.pi,-1/radius*180/np.pi,-1/radius*180/np.pi]#,5/radius*180/np.pi]
    initial_center_y=np.divide([1,1,-1,-1],10)#,-5,5,-1,1],10)
    #
    initial_directions=np.transpose(np.array([initial_gammas,initial_thetas,initial_center_y]))#,initial_radii,initial_center_yb]))#,initial_center_yb]))#,initial_center_xb]))
    variables=[gamma0,theta,center_y]#,radius,guess_tipy]#,guess_tipx]#,guess_cy]#,guess_tipx]#,center_x]
    
    #rotation_param=[theta,center_y]
    #variables=[gamma0,rotation_param]
    #
    #initial_gammas=np.array([-.002,.002,-.002,.002])
    #initial_rotation_param=np.array([list([-1,1,-1,1]*1/radius),[1,1,-1,-1]])
    #initial_directions=[initial_gammas,initial_rotation_param]
    
    ###http://informatik.unibas.ch/fileadmin/Lectures/HS2013/CS253/PowellAndDP1.pdf slides about minimizations methods
    res = minimize(error_f, variables, args=(edges.shape,), method='Powell',
                   options={'direc':initial_directions,'maxiter':100,'xtol': 1e-3,'ftol':1e-2, 'disp': True})
    #,options={'xtol': 1e-8, 'disp': True,'maxfev':100})
    optimal_variables = res.x
    
    #res = fmin_powell(error_f, variables, direc=initial_directions,maxiter=100)#'xtol': 1e-3,'ftol':1e-2, 'disp': True})#,options={'xtol': 1e-8, 'disp': True,'maxfev':100})
    #optimal_variables=res
    
    
    error, R_python, Z_python, R, Z, mini_inds, RMSd = young_laplace(optimal_variables, edges.shape)
    #error,R_python,Z_python,R,Z,RMSd=young_laplace(optimal_variables)
    image1b = rotate(image1,optimal_variables[1],center=base_center,resize=False)
    
    
    center_yb, center_xb = rotate_lines([center_y], [center_x], tip, optimal_variables[1])
    
    center_yb = center_yb[0]
    center_xb = center_xb[0]
    
    
    plt.figure()
    ax=plt.axes()
    plt.imshow(image1,cmap='gray')
    circle = plt.Circle((center_y, center_x), radius=radius, color='r', fill=False)
    ax.add_patch(circle)
    plt.plot(R_python,Z_python,'*g',markersize=1)
    plt.plot(R,Z,'*b',markersize=1)
    plt.plot([base_center[0],tip[0]],[base_center[1],tip[1]],'-y')
    
    #R_dist=[]
    #Z_dist=[]
    
    #R_dist_l=[]
    #Z_dist_l=[]
    #R_dist_r=[]
    #Z_dist_r=[]
    #for i in range(len(mini_inds[0])):
    ##    R_dist.append(mini_inds[i][1])
    ##    Z_dist.append(mini_inds[i][0])
    #    R_dist_l.append(mini_inds[0][i][0])
    #    Z_dist_l.append(mini_inds[0][i][1])
    #for i in range(len(mini_inds[1])):
    #    R_dist_r.append(mini_inds[1][i][0])
    #    Z_dist_r.append(mini_inds[1][i][1])
    #plt.plot(R_dist,Z_dist,'*m',markersize=1)
