import numpy as np
import pop_tools


def test_cfc12_numpy_1():
    sol = pop_tools.cfc12sol(35., 15.)
    np.testing.assert_almost_equal(sol, 3.103297535228039e-12, decimal=14)

    
def test_cfc11_numpy_1():
    sol = pop_tools.cfc11sol(35., 15.)
    np.testing.assert_almost_equal(sol, 1.149632255646373e-11, decimal=14)    