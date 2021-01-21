import pytest
import numpy as np

from numpy.testing import assert_equal, assert_almost_equal

from drop.optimize import *


def test_radialsquared_distance_ShiftedLines():

    a = 2
    b = 0
    shift = 1.2

    X1 = np.linspace(0, 10, 10)
    Y1 = a * X1 + b

    X2 = X1.copy() + shift
    Y2 = Y1.copy()

    res = deviation._radial_squared_distance(X1, Y1, X2, Y2)
    expected = np.full(10, shift**2)
    assert_almost_equal(res, expected)


def test_verticalsquared_distance_ShiftedLines():

    a = 2
    b = 0
    shift = 1.2

    X1 = np.linspace(0, 10, 10)
    Y1 = a * X1 + b

    X2 = X1.copy()
    Y2 = Y1.copy() + shift

    res = deviation._vertical_squared_distance(X1, Y1, X2, Y2)
    expected = np.full(10, shift**2)
    assert_almost_equal(res, expected)


def test_shortest_squared_distance_ShiftedLines():
    """
    Note: this test relies on successful radial and vertical dist.
    """

    a = 2
    b = 0
    shift = 10.2

    X1 = np.linspace(0, 10, 10)
    Y1 = a * X1 + b

    X2 = X1.copy()
    Y2 = Y1.copy() + shift

    radial = deviation._radial_squared_distance(X1, Y1, X2, Y2)
    radial = np.sqrt(radial[0])

    vertical = deviation._vertical_squared_distance(X1, Y1, X2, Y2)
    vertical = np.sqrt(vertical[0])

    orthogonal = radial * np.sin(np.arctan(vertical / radial))
    expected = np.full(10, orthogonal**2)

    res = deviation._shortest_squared_distance(X1, Y1, X2, Y2)
    assert_almost_equal(res, expected)


def test_orthogonal_squared_distance_ShiftedLines():
    """
    Note: this test relies on successful radial and vertical dist.
    """

    a = 2
    b = 0
    shift = 10.2

    X1 = np.linspace(0, 10, 10)
    Y1 = a * X1 + b

    X2 = X1.copy()
    Y2 = Y1.copy() + shift

    radial = deviation._radial_squared_distance(X1, Y1, X2, Y2)
    radial = np.sqrt(radial[0])

    vertical = deviation._vertical_squared_distance(X1, Y1, X2, Y2)
    vertical = np.sqrt(vertical[0])

    orthogonal = radial * np.sin(np.arctan(vertical / radial))
    expected = np.full(10, orthogonal**2)

    res = deviation._orthogonal_squared_distance(X1, Y1, X2, Y2)
    print(res)
    print(expected)
    assert_almost_equal(res, expected)
