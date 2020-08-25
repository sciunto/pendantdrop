import pytest
import numpy as np
from skimage import draw, filters, morphology

from numpy.testing import assert_equal, assert_almost_equal

from drop.improcessing.worthington import _drop_volume


def test_drop_volume_circle():
    img = np.zeros((3000, 3000))
    r, c = (1455, 1200)
    radius = 1200
    rr, cc = draw.circle_perimeter(r, c, radius, method='bresenham')
    img[rr, cc] = 1
    expected = 4/3 * np.pi * radius**3
    res = _drop_volume(img) / expected
    assert_almost_equal(res, 1., decimal=3)
