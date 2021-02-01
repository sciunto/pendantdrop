"""
=================================
Determination of RANSAC residuals
=================================

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from drop.io import load_image
from drop.improcessing import fit_circle_tip, detect_edges, guess_angle
from drop.optimize import (young_laplace,
                                       deviation_edge_model_simple)
from drop.optimize.deviation import shortest_RMS, radial_RMS

from drop.improcessing import elbow_curve_ransac_residuals



image_path = os.path.join('..', '..', 'data', 'gwater20_70.tiff')
zoom =  ([550, 1600], [0, 1200])
calib = 0.001/289.2921  # m / px
# Arbitrary first guess for gamma
initial_surface_tension = 0.04  # N/m
surface_tension_range = (0.02, 0.1)  # N/m
fluid_density = 1000
gray_level = 57




image1 = load_image(image_path, region=zoom)
edges, RZ_edges = detect_edges(image1, level=gray_level)

######################################################################
# Define the fitting procedure
# ----------------------------


def get_surf_tension(**ransac_params):

    center_Z, center_R, radius = fit_circle_tip(edges.shape,
                                                RZ_edges,
                                                method='ransac',
                                                debug=False, **ransac_params)
    theta = guess_angle(edges, center_Z, center_R)


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

    return guessed_surface_tension

######################################################################
# Find the best residual threshold
# --------------------------------

min_residuals = 0.001
max_residuals = 0.01
num_residuals = 6

ransac_params = {'min_samples' : 3,
                 'max_trials' : 5000}

r, std = elbow_curve_ransac_residuals(get_surf_tension,
                                      min_residuals, max_residuals, num_residuals,
                                      num_test=15,
                                      mode='log',
                                      **ransac_params)

plt.xlabel('residuals')
plt.ylabel('std(surface tension)')
plt.plot(r, std, '-o')
