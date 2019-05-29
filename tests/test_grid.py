import os

import xarray as xr

import pop_tools

from .util import ds_compare

testdata_dir = os.path.join(os.path.dirname(__file__), 'data')


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
    ds_ref = xr.open_zarr(f'{testdata_dir}/POP_gx3v7.zarr')
    assert ds_compare(ds_test, ds_ref, assertion='allclose', rtol=1e-14, atol=1e-14)
