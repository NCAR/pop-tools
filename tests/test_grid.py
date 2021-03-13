import os

import pytest
import xarray as xr
from xarray.testing import assert_equal

import pop_tools
from pop_tools import DATASETS
from pop_tools.datasets import UnzipZarr

from .util import ds_compare, is_ncar_host


@pytest.mark.parametrize('grid', pop_tools.grid_defs.keys())
def test_get_grid(grid):
    print(grid)
    ds = pop_tools.get_grid(grid)
    ds.info()
    assert isinstance(ds, xr.Dataset)


def test_get_grid_scrip():
    ds_test = pop_tools.get_grid('POP_gx3v7', scrip=True)
    ds_ref = xr.open_dataset(pop_tools.DATASETS.fetch('POP_gx3v7.nc'))
    assert ds_compare(ds_test, ds_ref, assertion='allclose', rtol=1e-14, atol=1e-14)


@pytest.mark.skipif(not is_ncar_host(), reason="Requires access to one of NCAR's machines.")
def test_cesm_local_inputdata():
    cesm_dataroot = os.environ.get('CESMDATAROOT', None)
    assert pop_tools.grid.INPUTDATA.path.as_posix() == cesm_dataroot


def test_get_grid_twice():
    ds1 = pop_tools.get_grid('POP_gx1v7')
    ds2 = pop_tools.get_grid('POP_gx1v7')
    xr.testing.assert_identical(ds1, ds2)


def test_get_grid_to_netcdf():
    for grid in pop_tools.grid_defs.keys():
        print('-' * 80)
        print(grid)
        ds = pop_tools.get_grid(grid)
        for format in ['NETCDF4', 'NETCDF3_64BIT']:
            gridfile = f'{grid}_{format}.nc'
            ds.to_netcdf(gridfile, format=format)
            os.system(f'rm -f {gridfile}')
        ds.info()
        assert isinstance(ds, xr.Dataset)
        print()


def test_four_point_min_kmu():
    zstore = DATASETS.fetch('comp-grid.tx9.1v3.20170718.zarr.zip', processor=UnzipZarr())
    ds = xr.open_zarr(zstore)

    # topmost row is wrong because we need to account for tripole seam
    # rightmost nlon is wrong because it doesn't matter
    expected = ds.KMU.isel(nlat=slice(-1), nlon=slice(-1))
    actual = pop_tools.grid.four_point_min(ds.KMT).isel(nlat=slice(-1), nlon=slice(-1))
    assert_equal(expected, actual)

    # make sure dask & numpy results check out
    actual = pop_tools.grid.four_point_min(ds.KMT.compute()).isel(nlat=slice(-1), nlon=slice(-1))
    assert_equal(expected, actual)


def test_dzu_dzt():

    zstore = DATASETS.fetch('comp-grid.tx9.1v3.20170718.zarr.zip', processor=UnzipZarr())
    ds = xr.open_zarr(zstore).sel(nlat=slice(100, 300))

    dzu, dzt = pop_tools.grid.calc_dzu_dzt(ds)
    # northernmost row will be wrong since we are working on a subset
    assert_equal(dzu.isel(nlat=slice(-1)), ds['DZU'].isel(nlat=slice(-1)))
    assert_equal(dzt, ds['DZT'])

    _, xds = pop_tools.to_xgcm_grid_dataset(ds)
    with pytest.raises(ValueError):
        pop_tools.grid.calc_dzu_dzt(xds)

    expected_vars = ['dz', 'KMT', 'DZBC']
    for var in expected_vars:
        dsc = ds.copy()
        del dsc[var]
        with pytest.raises(ValueError):
            pop_tools.grid.calc_dzu_dzt(dsc)
