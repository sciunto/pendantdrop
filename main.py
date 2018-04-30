# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:15:31 2018

@author: miguet
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize


from drop.io import load_image
from drop.edge import fit_circle_tip
from drop.edge import detect_edges
from drop.edge import guess_angle
from drop.optimization import young_laplace,\
                                deviation_edge_model_simple,\
                                deviation_edge_model_full


# x = along Z
# y = along R


if __name__ == '__main__':

    image_path = 'uEye_Image_000827.bmp'
    zoom = ([100, 1312], [400, 1900])
    calib = 0.00124 / 400  # mm / px
    # Arbitrary first guess for gamma
    initial_surface_tension = 0.04  # N/m
    min_surface_tension = 0.02
    max_surface_tension = 0.1
    # image_path = 'uEye_Image_002767.bmp'
    # zoom = ((814, 1020), (1920, 1772))

    image1 = load_image(image_path, region=zoom)

    edges, R_edges, Z_edges = detect_edges(image1, method='contour')

    # Guess parameters
    center_x, center_y, radius = fit_circle_tip(edges.shape,
                                                R_edges, Z_edges,
                                                method='ransac',
                                                debug=False)
    theta = guess_angle(edges, center_x, center_y)

    # Note that below the method method='SLSQP' can be used also.

    # Step 1: consider only the surface tension
    # as it is not guessed so far
    ini_variables = np.array((initial_surface_tension))
    res = minimize(deviation_edge_model_simple,
                   ini_variables,
                   args=(theta, center_y, center_x, radius, R_edges, Z_edges, calib),
                   method='L-BFGS-B',
                   bounds=((min_surface_tension, max_surface_tension),),
                   options={'maxiter': 10,
                            'ftol': 1e-2,
                            'disp': True})
    guessed_surface_tension = res.x[0]
    print('Step 1-RMS:', res.fun)

    # Step 2: consider all the parameters
    # as it is not guessed so far
    ini_variables2 = np.array((guessed_surface_tension,
                               theta, center_y, center_x, radius,))
    param_bounds = ((guessed_surface_tension-2e-3, guessed_surface_tension+2e-3),
                    (theta*0.7, theta*1.3),
                    (center_y-5, center_y+5),
                    (center_x-5, center_x+5),
                    (radius-10,radius+10),
                    )

    res = minimize(deviation_edge_model_full,
                   ini_variables2,
                   args=(R_edges, Z_edges, calib),
                   method='L-BFGS-B',
                   bounds=param_bounds,
                   options={'maxiter': 100,
                            'ftol': 1e-6,
                            'disp': True})
    optimal_variables = res.x
    print('Step 2-ini params BFGS:', ini_variables2)
    print('Step 2-opt params BFGS:', optimal_variables)
    print('Step 2-RMS:', res.fun)

    # Plot
    R, Z = young_laplace(*optimal_variables,
                         R_edges, Z_edges, calib, num_points=1e4)

    plt.figure()
    ax = plt.axes()
    plt.imshow(image1, cmap='gray')
    circle = plt.Circle((center_y, center_x), radius=radius,
                        color='c', fill=False)
    ax.add_patch(circle)
    plt.plot(R_edges, Z_edges, '*g', markersize=1)
    plt.plot(R, Z, 'r-', markersize=2)
    plt.plot(center_y, center_x, 'bo')
    plt.title('Gamma = %f N/m' % optimal_variables[0])
    plt.show()
