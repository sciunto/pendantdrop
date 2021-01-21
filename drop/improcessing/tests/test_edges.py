import pytest
import numpy as np
from skimage import draw, filters, morphology

from numpy.testing import assert_equal, assert_almost_equal

from drop.improcessing import *


def test_detect_edge_simple_edge_contour():
    width = 30
    height = 40
    black = np.ones((width, height), dtype=np.uint8) * 255

    position = 20
    black[:, :position] = 0

    edges, (R, Z) = detect_edges(black)

    # Edge position is between position-1 and position
    expected_R = np.ones(width) * (position + position - 1) / 2.
    assert_equal(R, expected_R)

    expected_Z = np.arange(0, width)[::-1]
    assert_equal(Z, expected_Z)

    # Edge position is shifted by one pixel
    assert np.argwhere(edges.sum(axis=0)==width)[0][0] == position - 1


def test_guess_angle_full_bottom_zero_angle():

    # Prepare a synthetic drop
    img = np.ones((100, 100), dtype=np.uint8) * 255
    img[80:, 40:61] = 0
    rr, cc = draw.disk((60, 50), 25)
    img[rr, cc] = 0
    img = filters.sobel(img)
    assert img[-1].sum() != 0
    img = img > img.mean()

    res = guess_angle(img, 60, 50)
    assert_equal(res,  0)


def test_guess_angle_full_bottom_positive_angle():

    # Prepare a synthetic drop
    img = np.ones((100, 100), dtype=np.uint8) * 255
    img[80:, 40:61] = 0
    rr, cc = draw.disk((60, 54), 25)
    img[rr, cc] = 0
    img = filters.sobel(img)
    assert img[-1].sum() != 0
    img = img > img.mean()

    res = guess_angle(img, 60, 54)
    assert_almost_equal(res,  0.10, decimal=2)


def test_guess_angle_empty_bottom_zero_angle():

    # Prepare a synthetic drop
    img = np.ones((100, 100), dtype=np.uint8) * 255
    img[80:, 40:62] = 0
    rr, cc = draw.disk((60, 50), 25)
    img[rr, cc] = 0
    img = filters.sobel(img)
    img = img > img.mean()
    img = morphology.skeletonize(img)
    assert img[-1].sum() == 0
    assert img[-2].sum() != 0

    res = guess_angle(img, 60, 50)
    assert_equal(res, 0)
