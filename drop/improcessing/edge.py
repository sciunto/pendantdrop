# -*- coding: utf-8 -*-

import numpy as np
from joblib import Parallel, delayed
from skimage import measure

from .circlemodel import CircleModelLinearized, CircleModel

__all__ = ['detect_edges',
           'fit_circle_tip',
           'guess_angle',
           ]


def detect_edges(image, **kwargs):
    """
    Detect the edge of the drop on the image.

    Parameters
    ----------
    image : ndarray
        Grayscale image.
    kwargs : dict, optional
        Optional arguments passed to `find_contours`.

    Returns
    -------
    results : tuple
        (edges, R_edges, Z_edges)

    Notes
    -----
    `contour` method calls `skimage.measure.find_countours`.
    The default level is the average of the max and min intensity
    values.
    """
    # By default:
    # Use the mean grayscale value of the image to get the contour.
    contour_kwargs = {'level': 0.5 * (image.max() - image.min())}
    contour_kwargs.update(**kwargs)

    contours = measure.find_contours(image, **contour_kwargs)
    Z_edges, R_edges = np.column_stack(contours[0])

    # Make a binary image
    edges = np.full(image.shape, False, dtype=bool)
    edges[Z_edges.astype(np.int), R_edges.astype(np.int)] = True

    return edges, (R_edges, Z_edges)


def _fit_circle_tip_ransac(shape, R, Z, *,
                           linearized=False, debug=False,
                           **kwargs):
    """
    Fit the tip of the drop with a circle by RANSAC.

    The points between the '45th parallel' and the tip are considered.

    Parameters
    ----------
    shape : tuple
        Image shape.
    R : array
        Radial coordinates.
    Z : array
        Vertical coordinates.
    linearized : boolean, optional
        If ``True`` use a simplified linearized residual estimator.
    debug : boolean, optional
        If `True`, activate plots to visualize the fit.
    kwargs : dict, optional
        Arguments passed to `skimage.measure.ransac`.

    Returns
    -------
    parameters : tuple
        (center_Z, center_R, radius)

    Notes
    -----
    See https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
    for details on the linearization.
    """
    # Default ransac parameters
    ransac_kwarg = {'min_samples': 6,
                    'residual_threshold': .1,
                    'max_trials': 1000,
                    }
    ransac_kwarg.update(kwargs)

    # Apply a mask to keep only the considered pixels.
    mask = Z < Z.min() + 0.5 * (Z[R.argmin()] - Z.min())
    Z_cropped = Z[mask]
    R_cropped = R[mask]

    points = np.column_stack((R_cropped, Z_cropped))

    if linearized:
        model_robust, inliers = measure.ransac(points, CircleModelLinearized,
                                               **ransac_kwarg)
    else:
        model_robust, inliers = measure.ransac(points, CircleModel,
                                               **ransac_kwarg)

    cy, cx, r = model_robust.params

    if debug:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 1, figsize=(15, 8))

        circle = plt.Circle((cy, cx), radius=r, facecolor='black', linewidth=2)
        axes.add_patch(circle)
        axes.plot(R, Z, 'g.')
        axes.plot(points[inliers, 0], points[inliers, 1],
                  'b.', markersize=1, label='inliers')
        axes.plot(points[~inliers, 0], points[~inliers, 1],
                  'r.', markersize=1, label='outliers')
        plt.title(f'radius: {r:.2f}, center: {cy:.2f}, {cx:.2f}')
        plt.legend()
        plt.show()

    return cx, cy, r


def fit_circle_tip(shape, RZ_edges, *, method='ransac', debug=False, **kwargs):
    """
    Fit the tip of the drop with a circle.

    Parameters
    ----------
    shape : tuple
        Image shape.
    RZ_edges : tuple of array
        (Radial, Vertical) coordinates of the edge.
    method : str, optional
        Name of the method: `ransac` or `ransac-lin`.
    debug : boolean, optional
        If `True`, activate plots to visualize the fit.
    kwargs : dict, optional
        Arguments passed to `skimage.measure.ransac`.

    Returns
    -------
    parameters : tuple
        (center_Z, center_R, radius)

    Notes
    -----
    `ransac` and `ransac-lin` apply RANSAC fitting procedure without or with
    a linear approximation in the residual. See
    https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
    for details on the linearization.
    """
    R, Z = RZ_edges
    if method.lower() == 'ransac':
        return _fit_circle_tip_ransac(shape, R, Z,
                                      debug=debug, linearized=False,
                                      **kwargs)
    elif method.lower() == 'ransac-lin':
        return _fit_circle_tip_ransac(shape, R, Z,
                                      debug=debug, linearized=True,
                                      **kwargs)
    else:
        raise ValueError('Wrong parameter value for `method`.')


def guess_angle(edges, center_Z, center_R):
    """
    Guess tilting angle value.

    The value is guessed based on the center position of the
    tip circle and the center of the drop base.

    Parameters
    ----------
    edges : boolean image
        Image containing the edges to fit.
    center_Z :
        Vertical position of tip circle center.
    center_R :
        Horizontal position of tip circle center.

    Returns
    -------
    angle : scalar
    """
    c_center = np.array((center_R, center_Z))

    # assume orientation base at bottom of image

    # Find non-empty bottom position
    position = -1
    while np.sum(edges[position, :]) < 2:
        position -= 1

    pixels_on_baseline = np.where(edges[position, :] == True)
    baseline_center = (np.mean((pixels_on_baseline[0],
                                pixels_on_baseline[-1])),
                       edges.shape[0]-1)

    distance_base_to_center = baseline_center - c_center
    hyp = np.linalg.norm(distance_base_to_center)
    opp = np.abs(baseline_center[0] - c_center[0])

    theta = np.arcsin(opp/hyp)

    return theta


def elbow_curve_ransac_residuals(get_surf_tension,
                                 min_residuals, max_residuals, num_residuals,
                                 num_test=10, mode='lin',
                                 **ransac_params):
    """
    Compute an elbow curve to help to choose the residual value in RANSAC.

    Parameters
    ==========
    get_surf_tension : callable
        Function returning the surface tension.
        `ransac_params` are passed to it.
    min_residuals : float
        Lowest tested residual value.
    max_residuals : float
        Highest tested residual value.
    num_residuals : int
        Number of tested values.
    num_test : int, optional
        Number of trials for each tested value.
    mode : string, optional
        Specify the spacing mode. Either `lin` or `log`.
    ransac_params : dict, optional
        Arguments passed to the ransac function.

    Returns
    =======
    (residuals, std_surface_tension)

    """

    if mode.lower() == 'log':
        residuals = np.geomspace(min_residuals, max_residuals, num_residuals)
    elif mode.lower() == 'lin':
        residuals = np.linspace(min_residuals, max_residuals, num_residuals)
    else:
        raise ValueError('Wrong `mode` value. Must be `lin` or `log`.')

    def compute_surf_tension_stat(get_surf_tension, num_test, **ransac_params):
        """
        Compute the surface tension std deviation over num_test.

        Notes
        =====
        The number of jobs is equal to the number of CPUs.
        """
        gamma = Parallel(n_jobs=-1)(delayed(get_surf_tension)(**ransac_params) for i in range(num_test))
        gamma = np.array(gamma)
        return gamma.std()

    gamma_std = []
    for residual in residuals:
        ransac_params.update({'residual_threshold': residual})
        gamma_std.append(compute_surf_tension_stat(get_surf_tension, num_test,
                                                   **ransac_params))

    std_surface_tension = np.array(gamma_std)
    return residuals, std_surface_tension
