# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:15:31 2018

@author: miguet
"""


#from IPython import get_ipython
#get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin_powell, minimize


from drop.io import load_image
from drop.edge import fit_circle_tip
from drop.edge import detect_edges
from drop.edge import guess_angle
from drop.theory import rotate_lines
from drop.optimization import young_laplace, deviation_edge_model


# x = along Z
# y = along R


if __name__ == '__main__':

    image_path = 'uEye_Image_000827.bmp'
    zoom = ([100,1312], [400,1900])
    calib = 0.00124 / 400  #400 pixel = 1.24mm
    # Arbitrary first guess for gamma
    gamma0 = 0.040 # N/m

    # image_path = 'uEye_Image_002767.bmp'
    # zoom = ((814, 1020), (1920, 1772))

    image1 = load_image(image_path, region=zoom)

    edges, R_edges, Z_edges = detect_edges(image1,
                                           method='contour')

    center_x, center_y, radius = fit_circle_tip(edges.shape,
                                                R_edges, Z_edges,
                                                method='ransac',
                                                debug=False)

    tipy, tipx = [center_y, center_x - radius]
    print(center_x, center_y)
    # Guess parameters
    # theta, guess_tipx, guess_tipy = guess_parameters(edges, R_edges, Z_edges, tip, center_x, center_y)
    # It seems better to get the guess of the tip from the circle fit
    theta = guess_angle(edges, center_x, center_y)


    initial_gammas = np.divide([-.02, .02, -.02, .02], 10)
    initial_thetas = np.divide([-.02, -.02, .02, .02], 5)
    initial_center_y = np.divide([1, 1, -1, -1], 1)
    initial_directions = np.transpose(np.array([initial_gammas,
                                              initial_thetas,
                                              initial_center_y]))


    variables = np.array((gamma0, theta, center_y))

    # slides about minimizations methods
    # http://informatik.unibas.ch/fileadmin/Lectures/HS2013/CS253/PowellAndDP1.pdf
    res = minimize(deviation_edge_model,
                   variables,
                   args=(center_x, radius, R_edges, Z_edges, calib),
                   method='Powell',
                   options={'direc': initial_directions,
                            'maxiter': 100,
                            'xtol': 1e-4,
                            'ftol': 1e-4,
                            'disp': True})
    #,options={'xtol': 1e-8, 'disp': True,'maxfev':100})
    optimal_variables = res.x



    R, Z = young_laplace(*optimal_variables, center_x, radius,
                         R_edges, Z_edges,  calib)


    print('directions:', initial_center_y)
    print('ini vars:', variables)
    print('opt vars:', optimal_variables)
    print(center_y, tipy)
    print(center_x, tipx)
    center_yb, center_xb = rotate_lines([center_y], [center_x], (tipy, tipx), optimal_variables[1])

    center_yb = center_yb[0]
    center_xb = center_xb[0]

    # Display purpose only...
    # Apply a mask
    # from skimage.draw import circle
    # rr, cc = circle(center_x, center_y, radius-5)
    # image1[rr, cc] = 10


    plt.figure()
    ax = plt.axes()
    plt.imshow(image1, cmap='gray')
    circle = plt.Circle((center_y, center_x), radius=radius, color='c', fill=False)
    ax.add_patch(circle)
    plt.plot(R_edges, Z_edges, '*g', markersize=1)
    plt.plot(R, Z, 'r-o', markersize=2)
    plt.plot(center_y, center_x, 'bo')


    #plt.plot([base_center[0], tip[0]], [base_center[1], tip[1]], '-y')
    plt.title('Gamma = %f N/m' % optimal_variables[0])
    plt.show()
