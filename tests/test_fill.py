import os

import numpy as np
import xarray as xr

import pop_tools

testdata_dir = os.path.join(os.path.dirname(__file__), 'data')


def test_lateral_fill_np_array():

    # generate psuedo-data
    dx, dy = 0.05, 0.05
    y, x = np.mgrid[slice(1, 3 + dy, dy),
                    slice(1, 5 + dx, dx)]
    z_orig = np.sin(x)**10 + np.cos(10 + y * x) * np.cos(x)

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
    ref_data_file = f'{testdata_dir}/lateral_fill_np_array_filled_ref.npz'
    with np.load(ref_data_file) as data:
        z_fill_ref = data['arr_0']

    # assert that we match the reference solution
    np.testing.assert_allclose(
        z_fill,
        z_fill_ref,
        atol=1e-10,
        equal_nan=True,
        verbose=True)
