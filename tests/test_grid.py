import os

import pytest
import xarray as xr

import pop_tools

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
