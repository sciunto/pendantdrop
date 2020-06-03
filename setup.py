#!/usr/bin/env python

import setuptools

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setuptools.setup(
    name         = 'pendantdrop',
    version      = '0.1.0',
    url          = "https://github.com/sciunto-org/pendantdrop",
    author       = "Francois Boulogne and other contributors",
    license      = "GPLv3",
    author_email = "devel@sciunto.org",
    description  = "Measure surface tension with pendant drops",
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    include_package_data = True,  # activate MANIFEST.in
    packages     = setuptools.find_packages(exclude=['doc', 'benchmarks']),
    install_requires = [],
)
