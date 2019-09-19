import os

import numpy as np
import xarray as xr

import pop_tools

testdata_dir = os.path.join(os.path.dirname(__file__), 'data')


def test_eos_numpy_1():
    rho = pop_tools.eos(35.0, 20.0, pressure=2000.0)
    np.testing.assert_almost_equal(rho, 1033.2133865866824181, decimal=14)


def test_eos_numpy_2():
    rho = pop_tools.eos(35.0, 20.0, depth=1976.5347494030665985)
    np.testing.assert_almost_equal(rho, 1033.2133865866824181, decimal=14)


def test_eos_numpy_3():
    rho, drhodS, drhodT = pop_tools.eos(35.0, 20.0, pressure=2000.0, return_coefs=True)
    np.testing.assert_almost_equal(rho, 1033.2133865866824181, decimal=14)


def test_eos_xarray_1():
    ds = xr.open_dataset(
        f'{testdata_dir}/cesm_pop_monthly.T62_g17.nc', decode_times=False, decode_coords=False
    )
    rho = pop_tools.eos(ds.SALT, ds.TEMP, depth=ds.z_t * 1e-2)
    assert isinstance(rho, xr.DataArray)


def test_eos_xarray_2():
    ds = xr.open_dataset(
        f'{testdata_dir}/cesm_pop_monthly.T62_g17.nc', decode_times=False, decode_coords=False
    )
    rho, drhodS, drhodT = pop_tools.eos(ds.SALT, ds.TEMP, depth=ds.z_t * 1e-2, return_coefs=True)
    assert isinstance(rho, xr.DataArray)
    assert isinstance(drhodS, xr.DataArray)
    assert isinstance(drhodT, xr.DataArray)


def test_eos_ds_dask():
    ds = xr.open_dataset(
        f'{testdata_dir}/cesm_pop_monthly.T62_g17.nc',
        decode_times=False,
        decode_coords=False,
        chunks={'z_t': 20},
    )
    rho = pop_tools.eos(ds.SALT, ds.TEMP, depth=ds.z_t * 1e-2)
    assert isinstance(rho, xr.DataArray)


def test_eos_ds_dask_2():
    ds = xr.open_dataset(
        f'{testdata_dir}/cesm_pop_monthly.T62_g17.nc',
        decode_times=False,
        decode_coords=False,
        chunks={'z_t': 20},
    )
    rho, drhodS, drhodT = pop_tools.eos(ds.SALT, ds.TEMP, depth=ds.z_t * 1e-2, return_coefs=True)
    assert isinstance(rho, xr.DataArray)
    assert isinstance(drhodS, xr.DataArray)
    assert isinstance(drhodT, xr.DataArray)
