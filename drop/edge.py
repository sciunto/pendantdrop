# -*- coding: utf-8 -*-

import numpy as np
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage import measure


def detect_edges(image, method='contour', **kwarg):
    """
    Detect the edge of the drop on the image.

    Parameters
    ----------
    image : ndarray
        Grayscale image.
    filter : string, optional
        'contour' or 'canny'

    Returns
    -------
    results : tuple
        (edges, R_edges, Z_edges)
    """
    if method.lower() == 'contour':
        # Use the mean grayscale value of the image to get the contour.
        level = image.mean()
        contours = measure.find_contours(image, level)
        Z_edges, R_edges = np.column_stack(contours[0])
        # Make a binary image
        edges = np.full(image.shape, False, dtype=bool)
        edges[Z_edges.astype(np.int), R_edges.astype(np.int)] = True
    elif method.lower() == 'canny':
        edges = canny(image, **kwarg)
        Z_edges=np.where(edges==True)[0]
        R_edges=np.where(edges==True)[1]
    else:
        raise ValueError('Wrong method value')

    return edges, (R_edges, Z_edges)


def _find_circle(edges, hough_radii):
    """
    Find the best circle to model points marked as `True`.

    Parameters
    ----------
    edges : boolean image
        Image containing the edges to fit.
    hough_radii : tuple
        Radii considered for the Hough transform

    Returns
    -------
    parameters : tuple
        (center_Z, center_R, radius, tip_position)

    """
    hough_res = hough_circle(edges, hough_radii, full_output=True)

    centers = []
    accums = []
    radii = []

    for radius, h in zip(hough_radii, hough_res):
        # For each radius, extract two circles
        peaks = peak_local_max(h, min_distance=100, num_peaks=2)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius, radius])

    idx = np.argsort(accums)[::-1][0]

    center_Z, center_R = centers[idx]
    radius = radii[idx]

    center_Z = center_Z - hough_radii[-1]
    center_R = center_R - hough_radii[-1]
    tip = [center_R, center_Z - radius]

    return center_Z, center_R, radius, tip


def _fit_circle_tip_hough_transform(shape, R, Z):
    """
    Fit a circle with a Hough transform on the points between
    the '45th parallel' and the tip.

    Parameters
    ----------
    shape : tuple
        Image shape.
    R : array
        Radial coordinates.
    Z : array
        Vertical coordinates.

    Returns
    -------
    parameters : tuple
        (center_Z, center_R, radius, tip_position)
    """
    # Assume upward bubble orientation
    # Mask to select the 45th parallel
    mask = Z < Z.min() + 0.5 * (Z[R.argmin()] - Z.min())
    edges = np.full(shape, False, dtype=bool)
    edges[Z[mask].astype(np.int), R[mask].astype(np.int)] = True

    # Guess the maximum radius
    max_possible_radius = .5 * (R.max() - R.min())
    min_possible_radius = int(0.8 * max_possible_radius)
    # Coarse grain
    step = 5
    hough_radii = np.arange(min_possible_radius, max_possible_radius, step)
    _, _, radius, _ = _find_circle(edges, hough_radii)
    # Fine grain
    hough_radii = np.arange(radius - 2 * step, radius + 2 * step, 1)
    return _find_circle(edges, hough_radii)


def _fit_circle_tip_ransac(shape, R, Z, debug=False):
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
    debug : boolean, optional
        If `True`, activate plots to visualize the fit.

    Returns
    -------
    parameters : tuple
        (center_Z, center_R, radius)

    """

    mask = Z < Z.min() + 0.5 * (Z[R.argmin()] - Z.min())
    Z_cropped = Z[mask]
    R_cropped = R[mask]

    points = np.column_stack((R_cropped, Z_cropped))

    model_robust, inliers = measure.ransac(points, measure.CircleModel,
                                           min_samples=4,
                                           residual_threshold=.1,
                                           max_trials=1000)

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
        plt.legend()

    return cx, cy, r


def fit_circle_tip(shape, RZ_edges, method='ransac', debug=False):
    """
    Fit the tip of the drop with a circle.

    Parameters
    ----------
    shape : tuple
        Image shape.
    R : array
        Radial coordinates.
    Z : array
        Vertical coordinates.
    method : str, optional
        Name of the method: ransac or hough.
    debug : boolean, optional
        If `True`, activate plots to visualize the fit.

    Returns
    -------
    parameters : tuple
        (center_Z, center_R, radius)
    """
    R, Z = RZ_edges
    if method.lower() == 'ransac':
        return _fit_circle_tip_ransac(shape, R, Z, debug=debug)
    elif method.lower() == 'hough':
        return _fit_circle_tip_hough_transform(shape, R, Z)
    else:
        raise ValueError('Wrong parameter value for `method`.')


def guess_angle(edges, center_Z, center_R):
    """
    Guess values for the angle and the circle's center.

    Parameters
    ----------
    edges : boolean image
        Image containing the edges to fit.
    center_Z :

    center_R :


    Returns
    -------
    angle : scalar
    """
    c_center = np.array((center_R, center_Z))

    # assume orientation base at bottom of image
    pixels_on_baseline = np.where(edges[-1, :] == True)
    baseline_center = (np.mean((pixels_on_baseline[0],
                                pixels_on_baseline[-1])),
                       edges.shape[0]-1)

    distance_base_to_center = baseline_center - c_center
    hyp = np.linalg.norm(distance_base_to_center)
    opp = np.abs(baseline_center[0] - c_center[0])

    theta = np.arcsin(opp/hyp)

    return theta



def guess_parameters(edges, RZ_edges, tip, center_Z, center_R):
    """
    Guess values for the angle and the tip position.

    Parameters
    ----------
    edges : boolean image
        Image containing the edges to fit.
    RZ_edges : tuple of array
        (Radial, Vertical) coordinates of the edge.
    Z_edges : array
        Vertical coordinates of the edge.
    tip :

    center_Z :

    center_R :


    Returns
    -------
    guessed_parameters : tuple
        (theta, tipx, tipy)
    """
    c_center = np.array((center_R, center_Z))
    R_edges, Z_edges = RZ_edges

    # assume orientation base at bottom of image
    pixels_on_baseline = np.where(edges[-1, :] == True)
    baseline_center = (np.mean((pixels_on_baseline[0],
                                pixels_on_baseline[-1])),
                       edges.shape[0]-1)

    distance_base_to_center = baseline_center - c_center
    hyp = np.linalg.norm(distance_base_to_center)
    opp = np.abs(baseline_center[0] - c_center[0])

    theta = np.arcsin(opp/hyp)

    shift = (edges.shape[0]-1-tip[1]) * np.tan(np.abs(theta))
    if center_R > baseline_center[0] and theta > 0:
        guess_tipy = baseline_center[0] + shift
    else:
        guess_tipy = baseline_center[0] - shift

    ind_min = np.argmin(np.abs(R_edges - guess_tipy))
    guess_tipx = Z_edges[ind_min]
    guess_tipy = R_edges[ind_min]

    return theta, guess_tipx, guess_tipy
