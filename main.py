# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:15:31 2018

@author: miguet
"""


#from IPython import get_ipython
#get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt




from scipy.optimize import minimize
from scipy.optimize import fmin_powell

from drop.io import load_image
from drop.edge import fit_circle_tip
from drop.edge import detect_edges
from drop.theory import rotate_lines
from drop.optimization import young_laplace, deviation_edge_model




from skimage.draw import circle, circle_perimeter

if __name__ == '__main__':

    image_path = 'uEye_Image_000827.bmp'
    zoom = ([100,1312], [800,1900])
    calib = 0.00124 / 400  #400 pixel = 1.24mm


    #image_path = 'uEye_Image_002767.bmp'


    image1 = load_image(image_path, region=zoom)

    edges, R_edges, Z_edges = detect_edges(image1)


    center_x, center_y, radius, tip = fit_circle_tip(image1.shape, R_edges, Z_edges, method='ransac')

    # Display purpose only...
    rr,cc = circle(center_x, center_y, radius-5)
    image1[rr, cc] = 10



    ###arbitrary first guess for gamma
    gamma0 = 0.040 # N/m

    #####guess gravity angle


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



    ind_min = np.where(abs(np.array(R_edges)-guess_tipy)==min(abs(np.array(R_edges)-guess_tipy)))[0][0]
    guess_tipx = Z_edges[ind_min]


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
    variables=[gamma0,theta,center_y]


    ###http://informatik.unibas.ch/fileadmin/Lectures/HS2013/CS253/PowellAndDP1.pdf slides about minimizations methods
    res = minimize(deviation_edge_model,
                   variables,
                   args=(edges.shape, radius, R_edges, Z_edges, tip, guess_tipx, center_x, calib),
                   method='Powell',
                   options={'direc':initial_directions,
                            'maxiter':100,
                            'xtol': 1e-3,
                            'ftol':1e-2,
                            'disp': True})
    #,options={'xtol': 1e-8, 'disp': True,'maxfev':100})
    optimal_variables = res.x

    #res = fmin_powell(error_f, variables, direc=initial_directions,maxiter=100)#'xtol': 1e-3,'ftol':1e-2, 'disp': True})#,options={'xtol': 1e-8, 'disp': True,'maxfev':100})
    #optimal_variables=res


    R, Z = young_laplace(optimal_variables, edges.shape, radius, R_edges, Z_edges, tip, guess_tipx, center_x, calib)
    #### image1b = rotate(image1,optimal_variables[1],center=base_center,resize=False)



    print('opt vars:', optimal_variables)
    center_yb, center_xb = rotate_lines([center_y], [center_x], tip, optimal_variables[1])

    center_yb = center_yb[0]
    center_xb = center_xb[0]


    plt.figure()
    ax=plt.axes()
    plt.imshow(image1, cmap='gray')
    circle = plt.Circle((center_y, center_x), radius=radius, color='c', fill=False)
    ax.add_patch(circle)
    plt.plot(R_edges, Z_edges, '*g', markersize=1)
    plt.plot(R, Z, 'ro', markersize=1)


    plt.plot([base_center[0], tip[0]], [base_center[1], tip[1]], '-y')
    plt.show( )
