import xarray as xr
import xgcm

import pop_tools
from pop_tools import DATASETS


def test_xgcm_grid():
    fname = DATASETS.fetch('tend_zint_100m_Fe.nc')
    ds = xr.open_dataset(fname)
    grid = pop_tools.get_xgcm_grid(ds, metrics=None)
    assert isinstance(grid, xgcm.Grid)
    assert set(['X', 'Y', 'Z']) == set(grid.axes.keys())
