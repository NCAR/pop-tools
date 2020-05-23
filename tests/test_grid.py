import os

import pytest
import xarray as xr

import pop_tools
from pop_tools import DATASETS

from .util import ds_compare


@pytest.mark.parametrize('grid', pop_tools.grid_defs.keys())
def test_get_grid(grid):
    print(grid)
    ds = pop_tools.get_grid(grid)
    ds.info()
    assert isinstance(ds, xr.Dataset)


def test_get_grid_scrip():
    ds_test = pop_tools.get_grid('POP_gx3v7', scrip=True)
    ds_ref = xr.open_dataset(DATASETS.fetch('POP_gx3v7.nc'))
    assert ds_compare(ds_test, ds_ref, assertion='allclose', rtol=1e-14, atol=1e-14)
