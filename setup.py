#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os.path import dirname, join
from pip.req import parse_requirements

from setuptools import (
    find_packages,
    setup,
)

with open(join(dirname(__file__), 'FreshQuant/VERSION.txt'), 'rb') as f:
    version = f.read().decode('ascii').strip()

requirements = [str(ir.req) for ir in parse_requirements("requirements.txt", session=False)]

# if sys.version_info.major == 2:
#     requirements += [str(ir.req) for ir in parse_requirements("requirements-py2.txt", session=False)]

setup(
    name='freshquant',
    version=version,
    description='Quantitative',
    classifiers=[],
    keywords='',
    author='Jessica.Sun',
    author_email='sjj6love@126.com',
    url='https://github.com/sjj6love/FreshQuant',
    license='',
    packages=find_packages(exclude=[]),
    package_data={'': ['*.*']},
    include_package_data=True,
    zip_safe=True,
    install_requires=requirements,
)