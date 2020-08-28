#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize


from drop.io import load_image
from drop.improcessing import fit_circle_tip, detect_edges, guess_angle
from drop.optimize import (young_laplace,
                               deviation_edge_model_simple,
                               deviation_edge_model_full)


from drop.optimize.deviation import shortest_RMS, radial_RMS
from drop.improcessing import worthington
from drop.utils import print_parameters


def main():
    # image_path = os.path.join('..', 'data', 'uEye_Image_002767.png')
    # zoom = ((714, 1920), (920, 1830))
    image_path = os.path.join('..', 'data', 'uEye_Image_000827.png')
    zoom = ([100, 1312], [400, 1900])
    calib = 0.00124 / 400  # m / px
    # Arbitrary first guess for gamma
    initial_surface_tension = 0.04  # N/m
    surface_tension_range = (0.02, 0.1)  # N/m
    fluid_density = 1000

    image1 = load_image(image_path, region=zoom)

    edges, RZ_edges = detect_edges(image1, method='contour')

    # Guess parameters
    center_Z, center_R, radius = fit_circle_tip(edges.shape,
                                                RZ_edges,
                                                method='ransac',
                                                debug=False)
    theta = guess_angle(edges, center_Z, center_R)

    # Note that below the method method='SLSQP' can be used also.

    # Step 1: consider only the surface tension
    # as it is not guessed so far
    print_parameters('Initial parameters',
                     initial_surface_tension,
                     theta,
                     center_R,
                     center_Z,
                     radius)

    ini_variables = np.array((initial_surface_tension))
    res = minimize(deviation_edge_model_simple,
                   ini_variables,
                   args=(theta, center_R, center_Z,
                         radius, RZ_edges, fluid_density,
                         calib),
                   method='L-BFGS-B',
                   bounds=(surface_tension_range,),
                   options={'maxiter': 10,
                            'ftol': 1e-2,
                            'disp': False})
    guessed_surface_tension = res.x[0]

    print_parameters('Step 1',
                     guessed_surface_tension,
                     theta,
                     center_R,
                     center_Z,
                     radius,
                     RMS=res.fun)

    ini_variables = np.array((initial_surface_tension))
    # Step 2: consider all the parameters
    ini_variables2 = np.array((guessed_surface_tension,
                               theta, center_R, center_Z, radius,))
    param_bounds = ((guessed_surface_tension-1e-2, guessed_surface_tension+1e-2),
                    (theta*0.7, theta*1.3),
                    (center_R-5, center_R+5),
                    (center_Z-5, center_Z+5),
                    (radius-10, radius+10),
                    )

    res = minimize(deviation_edge_model_full,
                   ini_variables2,
                   args=(RZ_edges, fluid_density, calib),
                   method='L-BFGS-B',
                   bounds=param_bounds,
                   options={'maxiter': 100,
                            'ftol': 1e-6,
                            'disp': False})
    optimal_variables = res.x
    surface_tension = optimal_variables[0]

    print_parameters('Step 2',
                     surface_tension,
                     optimal_variables[1],
                     optimal_variables[2],
                     optimal_variables[3],
                     optimal_variables[4],
                     RMS=res.fun)

    # Plot
    RZ_model = young_laplace(*optimal_variables, fluid_density,
                             calib, RZ_edges=RZ_edges, num_points=1e4)

    #oRMS = shortest_RMS(RZ_model, RZ_edges)
    rRMS = radial_RMS(RZ_model, RZ_edges)
    #print(f'OrthoRMS: {oRMS}, RadialRMS {rRMS}')


    wo_number = worthington(edges, calib, surface_tension, fluid_density)

    plt.figure()
    ax = plt.axes()
    plt.imshow(image1, cmap='gray')
    circle = plt.Circle((center_R, center_Z), radius=radius,
                        color='c', fill=False)
    ax.add_patch(circle)
    plt.plot(*RZ_edges, '*g', markersize=1)
    plt.plot(*RZ_model, 'r-', markersize=2)
    plt.plot(center_R, center_Z, 'bo')
    plt.title(f'Wo = {wo_number:.2}, Gamma = {surface_tension:.4} N/m, RMS = {rRMS:.3}')
    plt.show()


if __name__ == '__main__':
    main()
