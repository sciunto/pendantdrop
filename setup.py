#!/usr/bin/env python
import setuptools
from os import path


def parse_requirements_file(filename):
    with open(filename, encoding='utf-8') as fid:
        requires = [l.strip() for l in fid.readlines() if l]

    return requires


INSTALL_REQUIRES = parse_requirements_file('requirements/default.txt')
# The `requirements/extras.txt` file is explicitely omitted because
# it contains requirements that do not have wheels uploaded to pip
# for the platforms we wish to support.
extras_require = {
        dep: parse_requirements_file('requirements/' + dep + '.txt')
        for dep in ['docs', 'optional', 'test']
        }

# requirements for those browsing PyPI
REQUIRES = [r.replace('>=', ' (>= ') + ')' for r in INSTALL_REQUIRES]
REQUIRES = [r.replace('==', ' (== ') for r in REQUIRES]
REQUIRES = [r.replace('[array]', '') for r in REQUIRES]
REQUIRES = [r.replace('-', '') for r in REQUIRES]


# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setuptools.setup(
        name         = 'pendantdrop',
        version      = '0.3.0',
        url          = "https://github.com/sciunto-org/pendantdrop",
        author       = "Francois Boulogne and other contributors",
        license      = "GPLv3",
        author_email = "devel@sciunto.org",
        description  = "Measure surface tension with pendant drops",
        long_description = long_description,
        long_description_content_type = 'text/markdown',
        include_package_data = True,  # activate MANIFEST.in
        packages     = setuptools.find_packages(exclude=['doc', 'benchmarks']),
        install_requires=INSTALL_REQUIRES,
        requires=REQUIRES,
        extras_require=extras_require,
        python_requires='>=3.6',
        )
