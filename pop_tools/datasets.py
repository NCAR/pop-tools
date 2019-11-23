"""
Functions to load sample data
"""

import os

import pkg_resources
import pooch

DATASETS = pooch.create(
    path=['~', '.pop_tools', 'data'],
    version_dev='master',
    base_url='ftp://ftp.cgd.ucar.edu/archive/aletheia-data/cesm-data/ocn/',
)

DATASETS.load_registry(pkg_resources.resource_stream('pop_tools', 'data_registry.txt'))
