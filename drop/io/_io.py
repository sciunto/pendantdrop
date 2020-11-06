# -*- coding: utf-8 -*-
from skimage import io

__all__ = ['load_image',]

def _auto_crop(image, expand=20):
    """
    Automated-crop based on the biggest darkest object.


    Parameters
    ----------
    image : ndarray
        Full image.
    expand : scalar
        Expand by this number of pixels.

    Returns
    -------
    cropped


    Notes
    -----
    The original image is thresholded with a minimum threshold to get the
    darkest pixels.
    """
    import numpy as np
    from skimage import measure
    from skimage import filters
    darkest = image < filters.threshold_minimum(image)
    labels = measure.label(darkest)
    props = measure.regionprops(labels)
    areas = np.array([prop.area for prop in props])
    bb = props[areas.argmax()].bbox
    return image[bb[0]-expand:bb[2]+expand, bb[1]-expand:bb[3]+expand]


def load_image(path, region=None, *, flip_updown=False, check_orientation=True):
    """
    Load an image.

    The library assumes that the drop base it at the bottom.

    Parameters
    ----------
    path : string
        File path.
    region : tuple, optional
        Tuple: corner positions to crop the image.
    flip_updown : bool, optional
        If True, flip the image upside down
    check_orientation : bool, optional
        Check the orientation of the final image.
        If the drop base is not at the bottom, an exception is raised.

    Returns
    -------
    image
    """
    image = io.imread(path, as_gray=True)

    if region is not None:
        image = image[region[0][0]:region[0][1],
                      region[1][0]:region[1][1]]
        if len(image.ravel()) == 0:
            raise ValueError('Zero size image. Check that the region is appropriate.')

    if flip_updown:
        image = image[::-1, :]

    if check_orientation:
        top = np.max(image[0, :])
        bottom = np.max(image[-1, :])
        if top < bottom:
            raise RuntimeError('The image orientation is detected to be incorrect. '
                               'The needle must be at the bottom.')

    return image
