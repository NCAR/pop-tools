import numpy as np
from numba import jit


def iterative_fill(var, isvalid_mask, ltripole=False, tol=1.0e-4):
    """Fill missing values by interative smoothing operation."""

    fillmask = (np.isnan(var) & isvalid_mask)
    nlat, nlon = var.shape[-2:]
    msv = 1e36

    var = var.copy()
    var[np.isnan(var)] = msv
    _iterative_fill_POP_core(nlat, nlon, var, fillmask, msv, tol, ltripole)
    var[var==msv] = np.nan

    return var


@jit(nopython=True)
def _iterative_fill_POP_core(nlat, nlon, var, fillmask, msv, tol, ltripole):
    """Iterative smoothing algorithm."""

    done = False
    iter = 0

    work = np.empty((nlat, nlon))

    while not done:
        done = True
        iter += 1

        # assume bottom row is land, so skip it
        for j in range(1, nlat):
            jm1 = j - 1
            jp1 = j + 1

            for i in range(0, nlon):
                # assume periodic in x
                im1 = i - 1
                if i == 0: im1 = nlon - 1
                ip1 = i + 1
                if i == nlon - 1: ip1 = 0

                work[j, i] = var[j, i]

                if not fillmask[j, i]:
                    continue

                numer = 0.0
                denom = 0.0

                # East
                if var[j, ip1] != msv:
                    numer += var[j, ip1]
                    denom += 1.0

                # North
                if j < nlat - 1:
                    if var[jp1, i] != msv:
                        numer += var[jp1, i]
                        denom += 1.0

                else:
                    # assume only tripole has non-land top row
                    if ltripole:
                        if var[j, nlon-i] != msv:
                            numer += var[j, nlon-i]
                            denom += 1.0

                # West
                if var[j, im1] != msv:
                    numer += var[j, im1]
                    denom += 1.0

                # South
                if var[jm1, i] != msv:
                    numer += var[jm1, i]
                    denom += 1.0

                # self
                if var[j, i] != msv:
                    numer += denom * var[j, i]
                    denom *= 2.

                if denom > 0.0:
                    work[j, i] = numer / denom
                    if var[j, i] == msv:
                        done = False
                    else:
                        delta = np.fabs(var[j, i] - work[j, i])
                        if delta > tol * np.abs(var[j, i]):
                            done = False


        var[1:nlat, :] = work[1:nlat, :]
