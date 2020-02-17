#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import sys
from os.path import exists

from setuptools import setup

if exists('README.rst'):
    with open('README.rst') as f:
        long_description = f.read()
else:
    long_description = ''

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

test_requirements = ['pytest']


CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Topic :: Scientific/Engineering',
]

setup(
    maintainer='Matthew Long',
    maintainer_email='mclong@ucar.edu',
    description='POP2-CESM tools',
    install_requires=install_requires,
    license='Apache License 2.0',
    long_description=long_description,
    classifiers=CLASSIFIERS,
    keywords='ocean modeling cesm',
    name='pop-tools',
    packages=['pop_tools'],
    test_suite='tests',
    tests_require=test_requirements,
    include_package_data=True,
    url='https://github.com/NCAR/pop-tools',
    use_scm_version={'version_scheme': 'post-release', 'local_scheme': 'dirty-tag'},
    setup_requires=['setuptools_scm', 'setuptools>=30.3.0'],
    zip_safe=False,
)
