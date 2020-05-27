import pytest
import numpy as np
import tempfile
import shutil
import os

from numpy.testing import assert_equal, assert_almost_equal

from skimage import io
from drop.io import load_image


def test_load():
    # Prepare
    directory = tempfile.gettempdir()
    filepath = os.path.join(directory, 'img.png')
    img = np.random.randint(255, size=(100, 100), dtype=np.uint8)
    io.imsave(filepath, img)
    # Load
    img2 = load_image(filepath)
    assert_equal(img, img2)
    # Tear down
    os.remove(filepath)


def test_load_crop_in():
    # Prepare
    directory = tempfile.gettempdir()
    filepath = os.path.join(directory, 'img.png')
    img = np.random.randint(255, size=(100, 100), dtype=np.uint8)
    io.imsave(filepath, img)
    # Load
    img2 = load_image(filepath, region=((10, 20), (40, 50)))
    assert_equal(img[10:20, 40:50], img2)
    # Tear down
    os.remove(filepath)


def test_load_crop_in():
    # Prepare
    directory = tempfile.gettempdir()
    filepath = os.path.join(directory, 'img.png')
    img = np.random.randint(255, size=(100, 100), dtype=np.uint8)
    io.imsave(filepath, img)
    # Load
    with pytest.raises(ValueError):
        img2 = load_image(filepath, region=((200, 300), (40, 50)))
    # Tear down
    os.remove(filepath)
