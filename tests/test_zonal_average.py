import pytest
import xarray as xr

from pop_tools import DATASETS
from pop_tools.regrid import zonal_average

test_ds = xr.open_dataset(DATASETS.fetch('cesm_pop_monthly.T62_g17.nc'), chunks={})


@pytest.mark.parametrize(
    'ds, grid_name, dest_grid_method',
    [(test_ds, 'POP_gx1v6', 'regular_grid'), (test_ds, 'POP_gx1v6', 'lat_aux_grid')],
)
def test_zonal_average(ds, grid_name, dest_grid_method):
    out = zonal_average(ds, grid_name, dest_grid_method)
    assert isinstance(out, xr.Dataset)
    assert 'region' in out.dims
    assert 'TEMP' in out.variables
    assert 14 == len(out.region)
