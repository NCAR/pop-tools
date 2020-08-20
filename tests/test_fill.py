import os

import numpy as np
import xarray as xr

import pop_tools
from pop_tools import DATASETS


def test_lateral_fill_np_array():

    # generate psuedo-data
    dx, dy = 0.05, 0.05
    y, x = np.mgrid[slice(1, 3 + dy, dy), slice(1, 5 + dx, dx)]
    z_orig = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

    # construct mask and apply mask
    valid_points = np.ones(z_orig.shape, dtype=np.bool)
    valid_points = np.where(y < 0.5 * np.sin(5 * x) + 1.5, False, valid_points)
    z_orig = np.where(~valid_points, np.nan, z_orig)

    # add missing values
    z_miss = z_orig.copy()
    z_miss[:20, 62:] = np.nan
    z_miss[15:18, 0:2] = 10.0

    # compute lateral fill
    z_fill = pop_tools.lateral_fill_np_array(z_miss, valid_points)

    # load reference data
    ref_data_file = DATASETS.fetch('lateral_fill_np_array_filled_ref.npz')
    with np.load(ref_data_file) as data:
        z_fill_ref = data['arr_0']

    # assert that we match the reference solution
    np.testing.assert_allclose(z_fill, z_fill_ref, atol=1e-14, equal_nan=True, verbose=True)


def test_lateral_fill_np_array_ltripole():

    # generate psuedo-data
    dx, dy = 0.05, 0.05
    y, x = np.mgrid[slice(1 - dy, 3 + dy, dy), slice(1 - dx, 5 + dx, dx)]
    z_orig = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

    # construct mask and apply mask
    valid_points = np.ones(z_orig.shape, dtype=np.bool)
    valid_points = np.where(y < 0.5 * np.sin(5 * x) + 1.5, False, valid_points)
    z_orig = np.where(~valid_points, np.nan, z_orig)

    # add missing values
    z_miss = z_orig.copy()
    z_miss[:20, 62:] = np.nan
    z_miss[35:, 55:70] = np.nan
    z_miss[15:18, 0:2] = 10.0
    z_miss[-2:, 12:20] = 10.0

    # compute lateral fill
    z_fill = pop_tools.lateral_fill_np_array(z_miss, valid_points, ltripole=True)

    # load reference data
    ref_data_file = DATASETS.fetch('lateral_fill_np_array_tripole_filled_ref.20200818.npz')
    with np.load(ref_data_file) as data:
        z_fill_ref = data['arr_0']

    # assert that we match the reference solution
    np.testing.assert_allclose(z_fill, z_fill_ref, atol=1e-14, equal_nan=True, verbose=True)


def test_lateral_fill_np_array_SOR():

    # generate psuedo-data
    dx, dy = 0.05, 0.05
    y, x = np.mgrid[slice(1, 3 + dy, dy), slice(1, 5 + dx, dx)]
    z_orig = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

    # construct mask and apply mask
    valid_points = np.ones(z_orig.shape, dtype=np.bool)
    valid_points = np.where(y < 0.5 * np.sin(5 * x) + 1.5, False, valid_points)
    z_orig = np.where(~valid_points, np.nan, z_orig)

    # add missing values
    z_miss = z_orig.copy()
    z_miss[:20, 62:] = np.nan
    z_miss[15:18, 0:2] = 10.0

    # compute lateral fill
    z_fill = pop_tools.lateral_fill_np_array(z_miss, valid_points, use_sor=True)

    # load reference data
    ref_data_file = DATASETS.fetch('lateral_fill_np_array_filled_SOR_ref.20200820.npz')
    with np.load(ref_data_file) as data:
        z_fill_ref = data['arr_0']

    # assert that we match the reference solution
    np.testing.assert_allclose(z_fill, z_fill_ref, atol=1e-14, equal_nan=True, verbose=True)


def test_lateral_fill_np_array_ltripole_SOR():

    # generate psuedo-data
    dx, dy = 0.05, 0.05
    y, x = np.mgrid[slice(1 - dy, 3 + dy, dy), slice(1 - dx, 5 + dx, dx)]
    z_orig = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

    # construct mask and apply mask
    valid_points = np.ones(z_orig.shape, dtype=np.bool)
    valid_points = np.where(y < 0.5 * np.sin(5 * x) + 1.5, False, valid_points)
    z_orig = np.where(~valid_points, np.nan, z_orig)

    # add missing values
    z_miss = z_orig.copy()
    z_miss[:20, 62:] = np.nan
    z_miss[35:, 55:70] = np.nan
    z_miss[15:18, 0:2] = 10.0
    z_miss[-2:, 12:20] = 10.0

    # compute lateral fill
    z_fill = pop_tools.lateral_fill_np_array(z_miss, valid_points, ltripole=True, use_sor=True)

    # load reference data
    ref_data_file = DATASETS.fetch('lateral_fill_np_array_tripole_filled_SOR_ref.20200820.npz')
    with np.load(ref_data_file) as data:
        z_fill_ref = data['arr_0']

    # assert that we match the reference solution
    np.testing.assert_allclose(z_fill, z_fill_ref, atol=1e-14, equal_nan=True, verbose=True)


def test_lateral_fill_2D():
    ds = pop_tools.get_grid('POP_gx3v7')
    field = ds.KMT.copy() * 1.0
    field = field.where(ds.KMT > 0)
    field.values[20:40, 80:] = np.nan
    da_in = field
    attrs = {'long_name': 'test field', 'units': 'none'}
    da_in.attrs = attrs

    valid_points = ds.KMT > 0
    da_out = pop_tools.lateral_fill(da_in, valid_points)
    assert (da_out.notnull() == valid_points).all()
    assert da_out.attrs == attrs


def test_lateral_fill_3D():
    ds = pop_tools.get_grid('POP_gx3v7')
    field = ds.KMT.copy() * 1.0
    field = field.where(ds.KMT > 0)
    field.values[20:40, 80:] = np.nan
    da_in = xr.DataArray(np.ones((3)), dims=('z_t')) * field
    attrs = {'long_name': 'test field', 'units': 'none'}
    da_in.attrs = attrs

    valid_points = ds.KMT > 0
    da_out = pop_tools.lateral_fill(da_in, valid_points)

    for k in range(0, da_out.shape[0]):
        if k == 0:
            arr_0 = da_out[k, :, :]
            continue
        arr_i = da_out[k, :, :]
        np.testing.assert_array_equal(arr_0, arr_i)

    assert da_out.attrs == attrs


def test_lateral_fill_4D():
    ds = pop_tools.get_grid('POP_gx3v7')
    field = ds.KMT.copy() * 1.0
    field = field.where(ds.KMT > 0)
    field.values[20:40, 80:] = np.nan

    da_in = (
        xr.DataArray(np.ones((3)), dims=('time')) * xr.DataArray(np.ones((5)), dims=('z_t')) * field
    )

    attrs = {'long_name': 'test field', 'units': 'none'}
    da_in.attrs = attrs

    valid_points = ds.KMT > 0
    da_out = pop_tools.lateral_fill(da_in, valid_points)

    arr_0 = da_out[0, 0, :, :]
    for k in range(0, da_out.shape[1]):
        for l in range(0, da_out.shape[0]):
            arr_i = da_out[l, k, :, :]
            np.testing.assert_array_equal(arr_0, arr_i)

    assert da_out.attrs == attrs


def test_lateral_fill_4D_3Dmask():
    ds = pop_tools.get_grid('POP_gx3v7')
    field = ds.KMT.copy() * 1.0
    field = field.where(ds.KMT > 0)
    field.values[20:40, 80:] = np.nan

    da_in = (
        xr.DataArray(np.ones((3)), dims=('time'))
        * xr.DataArray(np.ones((len(ds.z_t))), dims=('z_t'))
        * field
    )

    attrs = {'long_name': 'test field', 'units': 'none'}
    da_in.attrs = attrs

    # make 3D mask
    nk = len(ds.z_t)
    nj, ni = ds.KMT.shape

    # make 3D array of 0:km
    zero_to_km = xr.DataArray(np.arange(0, nk), dims=('z_t'))
    ONES_3d = xr.DataArray(np.ones((nk, nj, ni)), dims=('z_t', 'nlat', 'nlon'))
    ZERO_TO_KM = zero_to_km * ONES_3d

    # mask out cells where k is below KMT
    valid_points = ZERO_TO_KM.where(ZERO_TO_KM < ds.KMT)
    valid_points = xr.where(valid_points.notnull(), True, False)

    da_out = pop_tools.lateral_fill(da_in, valid_points)

    for k in range(0, da_out.shape[1]):
        for l in range(0, da_out.shape[0]):
            if l == 0:
                arr_0 = da_out[0, k, :, :]
            arr_i = da_out[l, k, :, :]
            np.testing.assert_array_equal(arr_0, arr_i)

    assert da_out.attrs == attrs
