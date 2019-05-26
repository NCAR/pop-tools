import numpy as np
import xarray as xr

import pop_tools


def test_eos_1():
    rho = pop_tools.eos(35., 20., pressure=2000.)
    np.testing.assert_almost_equal(rho, 1033.2133865866824181, decimal=14)

def test_eos_2():
    rho = pop_tools.eos(35., 20., depth=1976.5347494030665985)
    np.testing.assert_almost_equal(rho, 1033.2133865866824181, decimal=14)
