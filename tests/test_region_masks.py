import os

import numpy as np
import pytest
import xarray as xr

import pop_tools


def test_make_masks():
    for grid in pop_tools.grid_defs.keys():
        region_masks = pop_tools.list_region_masks(grid)
        for region_mask in region_masks:
            mask3d = pop_tools.region_mask_3d(grid, mask_name=region_mask)
            assert isinstance(mask3d, xr.DataArray)
            assert mask3d.dims == ('region', 'nlat', 'nlon')


def test_make_default_mask():
    for grid in pop_tools.grid_defs.keys():
        mask3d = pop_tools.region_mask_3d(grid)
        sum_over_region = mask3d.sum('region').values.ravel()
        np.testing.assert_equal(sum_over_region[sum_over_region != 0.0], 1.0)
        np.testing.assert_equal(sum_over_region[sum_over_region != 1.0], 0.0)


def test_user_defined_mask():
    region_defs = {
        'NAtl-STG': [
            {
                'match': {'REGION_MASK': [6]},
                'bounds': {'TLAT': [32.0, 42.0], 'TLONG': [310.0, 350.0]},
            }
        ],
        'NAtl-SPG': [
            {
                'match': {'REGION_MASK': [6]},
                'bounds': {'TLAT': [50.0, 60.0], 'TLONG': [310.0, 350.0]},
            }
        ],
    }

    mask3d = pop_tools.region_mask_3d('POP_gx1v7', region_defs=region_defs)
    assert isinstance(mask3d, xr.DataArray)
    assert mask3d.dims == ('region', 'nlat', 'nlon')


def test_bad_schema():
    region_defs = {'NAtl-STG': {'match': {'REGION_MASK': [6]}, 'bounds': {'TLAT': [32.0, 42.0]}}}
    with pytest.raises(AssertionError):
        pop_tools.region_mask_3d('POP_gx1v7', region_defs=region_defs)
