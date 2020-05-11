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


def load_image(path, region=None):
    """
    Load an image.

    Parameters
    ----------
    path : string
        File path.
    region : tuple, optional
        Tuple: corner positions to crop the image.

    Returns
    -------
    image
    """
    image = io.imread(path, as_gray=True)

    if region is not None:
        image = image[region[0][0]:region[0][1],
                      region[1][0]:region[1][1]]
    return image
