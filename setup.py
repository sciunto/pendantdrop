#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError as ex:
    print('setuptools not found. Falling back to distutils.core')
    from distutils.core import setup

setup(
    name         = 'pendantdrop',
    version      = 'devel',
    url          = "https://github.com/sciunto-org/pendantdrop",
    author       = "Francois Boulogne and other contributors",
    license      = "LGPLv3 or BSD",
    author_email = "devel@sciunto.org",
    description  = "",
    packages     = ['drop'],
    install_requires = [],
    extra_requires = {}
)
