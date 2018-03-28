#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 18:57:09 2018

@author: fr
"""


from skimage import io


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
    image = io.imread(path, as_grey=True)

    if region is not None:
        image = image[region[0][0]:region[0][1],
                      region[1][0]:region[1][1]]
    return image

