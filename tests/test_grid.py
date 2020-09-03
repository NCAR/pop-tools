import os

import xarray as xr

import pop_tools
from pop_tools import DATASETS

from .util import ds_compare


def test_template():
    print(pop_tools.grid_defs)


def test_get_grid():
    for grid in pop_tools.grid_defs.keys():
        print('-' * 80)
        print(grid)
        ds = pop_tools.get_grid(grid)
        ds.info()
        assert isinstance(ds, xr.Dataset)
        print()


def test_get_grid_scrip():
    ds_test = pop_tools.get_grid('POP_gx3v7', scrip=True)
    ds_ref = xr.open_dataset(DATASETS.fetch('POP_gx3v7.nc'))
    assert ds_compare(ds_test, ds_ref, assertion='allclose', rtol=1e-14, atol=1e-14)


def test_get_grid_to_netcdf():
    for grid in pop_tools.grid_defs.keys():
        print('-' * 80)
        print(grid)
        ds = pop_tools.get_grid(grid)
        for format in ['NETCDF4', 'NETCDF3_64BIT']:
            gridfile = f'{grid}_{format}.nc'
            ds.to_netcdf(gridfile, format=format)
            os.system(f'rm -f {gridfile}')
