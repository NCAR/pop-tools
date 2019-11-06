"""
Functions to load sample data
"""

import os

import aletheia_data

URLS = {
    'tend_zint_100m_Fe.nc': 'ftp://ftp.cgd.ucar.edu/archive/aletheia-data/cesm-data/ocn/tend_zint_100m_Fe.nc',
    'iron_tracer.nc': 'ftp://ftp.cgd.ucar.edu/archive/aletheia-data/cesm-data/ocn/iron_tracer.nc',
    'daily_surface_potential_temperature.nc': 'ftp://ftp.cgd.ucar.edu/archive/aletheia-data/cesm-data/ocn/daily_surface_potential_temperature.nc',
    'monthly_dissolved_oxygen.nc': 'ftp://ftp.cgd.ucar.edu/archive/aletheia-data/cesm-data/ocn/monthly_dissolved_oxygen.nc',
}

DATASETS = aletheia_data.create(
    path=['~', '.aletheia', 'data'],
    version_dev='master',
    base_url='ftp://ftp.cgd.ucar.edu/archive/aletheia-data',
    urls=URLS,
)

DATASETS.load_registry(os.path.join(os.path.dirname(__file__), 'registry.txt'))
