import pytest
import xarray as xr
import xgcm

import pop_tools
from pop_tools import DATASETS


@pytest.mark.parametrize('file', ['tend_zint_100m_Fe.nc', 'g.e20.G.TL319_t13.control.001_hfreq.nc'])
def test_xgcm_grid(file):
    filepath = DATASETS.fetch(file)
    ds = xr.open_dataset(filepath)
    grid = pop_tools.get_xgcm_grid(ds, metrics=None)
    assert isinstance(grid, xgcm.Grid)
    assert set(['X', 'Y', 'Z']) == set(grid.axes.keys())
