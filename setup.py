#!/usr/bin/env python

import setuptools

setuptools.setup(
    name         = 'pendantdrop',
    version      = '0.1',
    url          = "https://github.com/sciunto-org/pendantdrop",
    author       = "Francois Boulogne and other contributors",
    license      = "LGPLv3 or BSD",
    author_email = "devel@sciunto.org",
    description  = "",
    packages     = setuptools.find_packages(exclude=['doc', 'benchmarks']),
    install_requires = [],
)
