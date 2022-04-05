import sys

import numpy as np
import pytest
import xarray as xr
import xgcm

import pop_tools
from pop_tools import DATASETS

ds_a = xr.open_dataset(DATASETS.fetch('tend_zint_100m_Fe.nc'), chunks={})
ds_b = xr.open_dataset(DATASETS.fetch('g.e20.G.TL319_t13.control.001_hfreq-coarsen.nc'), chunks={})
ds_c = ds_a[['TLAT', 'ULONG', 'ULAT', 'TLONG']].isel(nlat=slice(0, 2), nlon=slice(0, 2))
ds_c['anom'] = xr.DataArray(
    np.arange(48).reshape(2, 2, 3, 2, 2),
    dims=['S', 'L', 'M', 'nlat', 'nlon'],
    coords={'S': [2000, 2001], 'L': [1, 2], 'M': [1, 2, 3]},
    attrs={
        'grid_loc': '2110',
        'long_name': 'Dissolved Inorganic Iron Tendency Vertical Integral, 0-100m',
        'cell_methods': 'time: mean',
        'units': 'mmol/m^3 cm/s',
    },
)


@pytest.mark.parametrize(
    'ds, old_spatial_coords, axes',
    [
        (ds_a, ['nlat', 'nlon', 'z_w'], ['X', 'Y', 'Z']),
        (ds_b, ['nlat', 'nlon', 'z_w'], ['X', 'Y', 'Z']),
        (ds_c, ['nlat', 'nlon'], ['X', 'Y']),
    ],
)
def test_to_xgcm_grid_dataset(ds, old_spatial_coords, axes):
    grid, ds_new = pop_tools.to_xgcm_grid_dataset(ds)
    assert isinstance(grid, xgcm.Grid)
    assert set(axes) == set(grid.axes.keys())
    new_spatial_coords = ['nlon_u', 'nlat_u', 'nlon_t', 'nlat_t']
    assert set(new_spatial_coords).issubset(set(ds_new.coords))
    assert not set(new_spatial_coords).intersection(set(ds.coords))
    assert not set(old_spatial_coords).intersection(set(ds_new.coords))


def test_to_xgcm_grid_dataset_missing_xgcm():
    from unittest import mock

    with pytest.raises(ImportError):
        with mock.patch.dict(sys.modules, {'xgcm': None}):
            filepath = DATASETS.fetch('tend_zint_100m_Fe.nc')
            ds = xr.open_dataset(filepath)
            _, _ = pop_tools.to_xgcm_grid_dataset(ds)


def test_set_metrics():
    from pop_tools.xgcm_util import get_metrics

    ds = xr.Dataset({'DXU': 1, 'DYT': 2, 'DZT': 3})
    actual = get_metrics(ds)
    expected = {('X',): ['DXU'], ('Y',): ['DYT'], ('Z',): ['DZT']}
    assert actual == expected

    assert not get_metrics(xr.Dataset({}))


@pytest.mark.parametrize(
    'ds',
    [ds_a, ds_b, ds_c],
)
def test_metrics_assignment(ds):
    grid, _ = pop_tools.to_xgcm_grid_dataset(ds)
    expected = {
        ('X',): ['DXU', 'DXT'],  # X distances
        ('Y',): ['DYU', 'DYT'],  # Y distances
        # ('Z',): ['DZU', 'DZT'],  # Z distances
        ('X', 'Y'): ['UAREA', 'TAREA'],  # Areas
    }

    if 'DXU' not in ds:
        # no metrics variables in ds_a
        assert not grid._metrics
    else:
        actual = {
            tuple(sorted(key)): [metric.name for metric in metrics]
            for key, metrics in grid._metrics.items()
        }
        if 'S_FLUX_ROFF_VSF' in ds:
            expected[('X', 'Y')] = ['TAREA']
            expected[('X',)] = ['DXU']
        assert actual == expected
