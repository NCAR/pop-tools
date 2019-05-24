#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import sys
from setuptools import setup
from os.path import exists

if exists('README.md'):
    with open('README.md') as f:
        long_description = f.read()
else:
    long_description = ''

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

test_requirements = ['pytest']

setup(
    maintainer='Matthew Long',
    maintainer_email='mclong@ucar.edu',
    description='POP2-CESM tools',
    install_requires=install_requires,
    license='Apache License 2.0',
    long_description=long_description,
    keywords='ocean modeling',
    name='pop-tools',
    packages=['pop_tools'],
    test_suite='tests',
    tests_require=test_requirements,
    include_package_data=True,
    url='https://github.com/NCAR/pop-tools',
    zip_safe=False,
)
