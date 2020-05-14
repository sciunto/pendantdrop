import pytest
import numpy as np
from skimage import draw

from numpy.testing import assert_equal

from drop.improcessing import detect_edges


def test_wrong_method():
    black = np.ones((10, 10), dtype=np.uint8) * 255
    with pytest.raises(ValueError):
        detect_edges(black, method='wrong')


def test_simple_edge_contour():
    width = 30
    height = 40
    black = np.ones((width, height), dtype=np.uint8) * 255

    position = 20
    black[:, :position] = 0

    edges, (R, Z) = detect_edges(black, method='contour')

    # Edge position is between position-1 and position
    expected_R = np.ones(width) * (position + position - 1) / 2.
    assert_equal(R, expected_R)

    expected_Z = np.arange(0, width)[::-1]
    assert_equal(Z, expected_Z)

    # Edge position is shifted by one pixel
    assert np.argwhere(edges.sum(axis=0)==width)[0][0] == position - 1


def test_simple_edge_canny():
    width = 30
    height = 40
    black = np.ones((width, height), dtype=np.uint8) * 255

    position = 20
    black[:, :position] = 0

    edges, (R, Z) = detect_edges(black, method='canny')

    # Edge position at position-1
    # Top and bottom are grinded by 2 pixels
    expected_R = np.ones(width - 2 * 2) * (position - 1)
    assert_equal(R, expected_R)

    expected_Z = np.arange(2, width - 2)
    assert_equal(Z, expected_Z)

    # Edge position is shifted by one pixel
    assert np.argwhere(edges.sum(axis=0)==width - 2*2)[0][0] == position - 1

