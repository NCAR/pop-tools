import sys

import pytest
import xarray as xr
import xgcm

import pop_tools
from pop_tools import DATASETS


@pytest.mark.parametrize(
    'file', ['tend_zint_100m_Fe.nc', 'g.e20.G.TL319_t13.control.001_hfreq-coarsen.nc']
)
def test_to_xgcm_grid_dataset(file):
    filepath = DATASETS.fetch(file)
    ds = xr.open_dataset(filepath)
    grid, ds_new = pop_tools.to_xgcm_grid_dataset(ds, metrics=None)
    assert isinstance(grid, xgcm.Grid)
    assert set(['X', 'Y', 'Z']) == set(grid.axes.keys())
    new_spatial_coords = set(['nlon_u', 'nlat_u', 'nlon_t', 'nlat_t'])
    for coord in new_spatial_coords:
        assert coord in ds_new.coords
        assert coord not in ds.coords


def test_to_xgcm_grid_dataset_missing_xgcm():
    from unittest import mock

    with pytest.raises(ImportError):
        with mock.patch.dict(sys.modules, {'xgcm': None}):
            filepath = DATASETS.fetch('tend_zint_100m_Fe.nc')
            ds = xr.open_dataset(filepath)
            _, _ = pop_tools.to_xgcm_grid_dataset(ds, metrics=None)
