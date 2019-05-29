import numpy as np
import xarray as xr
from numba import jit


def lateral_fill(da_in, isvalid_mask, ltripole=False, tol=1.0e-4):
    """Perform lateral fill on xarray.DataArray

    Parameters
    ----------
    da_in : xarray.DataArray
      DataArray on which to fill NaNs. Fill is performed on the two
      rightmost dimenions. Grid is assumed periodic in `x` direction
      (last dimension).

    isvalid_mask : xarray.DataArray, boolean
      Valid values mask: `True` where data should be filled. Must have the
      same rightmost dimenions as `da_in`.

    ltripole : boolean, optional [default=False]
      Logical flag; if `True` then treat the top row of the grid as periodic
      in the sense of a tripole grid.

    tol : float, optional [default=1.0e-4]
      Convergence criteria: stop filling when values change is less or equal
      to `tol * var`; i.e. `delta <= tol * np.abs(var[j, i])`.

    Returns
    -------
    da_out : xarray.DataArray
      DataArray with NaNs filled by iterative smoothing.

    """

    dims_in = da_in.dims
    non_lateral_dims = dims_in[:-2]

    attrs = da_in.attrs
    encoding = da_in.encoding
    coords = da_in.coords

    da_in, isvalid_mask = xr.broadcast(da_in, isvalid_mask)

    if len(non_lateral_dims) > 0:
        da_in_stack = da_in.stack(non_lateral_dims=non_lateral_dims)
        da_out_stack = xr.full_like(da_in_stack, fill_value=np.nan)
        isvalid_mask_stack = isvalid_mask.stack(non_lateral_dims=non_lateral_dims)
        for i in range(da_in_stack.shape[-1]):
            arr = da_in_stack.data[:, :, i]
            da_out_stack[:, :, i] = lateral_fill_np_array(arr, isvalid_mask_stack.data[:, :, i])

        da_out = da_out_stack.unstack('non_lateral_dims').transpose(*dims_in)

    else:
        da_out = xr.full_like(da_in, fill_value=np.nan)
        da_out[:, :] = lateral_fill_np_array(da_in.data, isvalid_mask.data)

    da_out.attrs = attrs
    da_out.encoding = encoding
    for k, da in coords.items():
        da_out[k].attrs = da.attrs

    return da_out


def lateral_fill_np_array(var, isvalid_mask, ltripole=False, tol=1.0e-4):
    """Perform lateral fill on numpy.array

    Parameters
    ----------

    var : numpy.array
      Array on which to fill NaNs. Fill is performed on the two
      rightmost dimenions. Grid is assumed periodic in `x` direction
      (last dimension).

    isvalid_mask : numpy.array, boolean
      Valid values mask: `True` where data should be filled. Must have the
      same rightmost dimenions as `da_in`.

    ltripole : boolean, optional [default=False]
      Logical flag; if `True` then treat the top row of the grid as periodic
      in the sense of a tripole grid.

    tol : float, optional [default=1.0e-4]
      Convergence criteria: stop filling when values change is less or equal
      to `tol * var`; i.e. `delta <= tol * np.abs(var[j, i])`.

    Returns
    -------

    da_out : xarray.DataArray
      DataArray with NaNs filled by iterative smoothing.

    """
    fillmask = np.isnan(var) & isvalid_mask
    nlat, nlon = var.shape[-2:]
    missing_value = 1e36

    var = var.copy()
    var[np.isnan(var)] = missing_value
    _iterative_fill_POP_core(nlat, nlon, var, fillmask, missing_value, tol, ltripole)
    var[var == missing_value] = np.nan

    return var


@jit(nopython=True)
def _iterative_fill_POP_core(nlat, nlon, var, fillmask, missing_value, tol, ltripole):
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
                if i == 0:
                    im1 = nlon - 1
                ip1 = i + 1
                if i == nlon - 1:
                    ip1 = 0

                work[j, i] = var[j, i]

                if not fillmask[j, i]:
                    continue

                numer = 0.0
                denom = 0.0

                # East
                if var[j, ip1] != missing_value:
                    numer += var[j, ip1]
                    denom += 1.0

                # North
                if j < nlat - 1:
                    if var[jp1, i] != missing_value:
                        numer += var[jp1, i]
                        denom += 1.0

                else:
                    # assume only tripole has non-land top row
                    if ltripole:
                        if var[j, nlon - i] != missing_value:
                            numer += var[j, nlon - i]
                            denom += 1.0

                # West
                if var[j, im1] != missing_value:
                    numer += var[j, im1]
                    denom += 1.0

                # South
                if var[jm1, i] != missing_value:
                    numer += var[jm1, i]
                    denom += 1.0

                # self
                if var[j, i] != missing_value:
                    numer += denom * var[j, i]
                    denom *= 2.0

                if denom > 0.0:
                    work[j, i] = numer / denom
                    if var[j, i] == missing_value:
                        done = False
                    else:
                        delta = np.fabs(var[j, i] - work[j, i])
                        if delta > tol * np.abs(var[j, i]):
                            done = False

        var[1:nlat, :] = work[1:nlat, :]
