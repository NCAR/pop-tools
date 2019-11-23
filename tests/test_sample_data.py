import xarray as xr

from pop_tools import DATASETS


def test_dataset_loading():
    fname = DATASETS.fetch('tend_zint_100m_Fe.nc')
    ds = xr.open_dataset(fname)
    assert isinstance(ds, xr.Dataset)
