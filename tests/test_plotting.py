import numpy as np
import xarray as xr

import pop_tools
from pop_tools.plotting.util import add_cyclic_point


def test_add_cyclic_point():
    ds = pop_tools.get_grid('POP_gx1v7')
    dsp = add_cyclic_point(ds)
    xr.testing.assert_equal(dsp.nlat, ds.nlat)
    np.testing.assert_equal(np.arange(0, 321), dsp.nlon)
