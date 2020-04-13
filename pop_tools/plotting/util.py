"""
This module contains utilities that are useful when plotting
POP model output.
"""

import numpy as np
import xarray as xr


# Reference:
# https://gist.github.com/matt-long/50433da346da8ac17cde926eec90a87c
def add_cyclic_point(ds):
    """
    Add a cyclic point to POP model output dataset

    Parameters
    ----------
    ds : xarray.Dataset
        POP output dataset

    Returns
    -------
    dso : xarray.Dataset
        modified POP model output dataset with cyclic point added
    """
    ni = ds.TLONG.shape[1]

    xL = int(ni / 2 - 1)
    xR = int(xL + ni)

    tlon = ds.TLONG.data
    tlat = ds.TLAT.data

    tlon = np.where(np.greater_equal(tlon, min(tlon[:, 0])), tlon - 360.0, tlon)
    lon = np.concatenate((tlon, tlon + 360.0), 1)
    lon = lon[:, xL:xR]

    if ni == 320:
        lon[367:-3, 0] = lon[367:-3, 0] + 360.0
    lon = lon - 360.0

    lon = np.hstack((lon, lon[:, 0:1] + 360.0))
    if ni == 320:
        lon[367:, -1] = lon[367:, -1] - 360.0

    # Trick cartopy into doing the right thing:
    # Cartopy gets confused when the cyclic coords are identical
    lon[:, 0] = lon[:, 0] - 1e-8

    # Periodicity
    lat = np.concatenate((tlat, tlat), 1)
    lat = lat[:, xL:xR]
    lat = np.hstack((lat, lat[:, 0:1]))

    TLAT = xr.DataArray(lat, dims=('nlat', 'nlon'))
    TLONG = xr.DataArray(lon, dims=('nlat', 'nlon'))

    dso = xr.Dataset({'TLAT': TLAT, 'TLONG': TLONG})

    # Copy vars
    varlist = [v for v in ds.data_vars if v not in ['TLAT', 'TLONG']]
    for v in varlist:
        v_dims = ds[v].dims
        if not ('nlat' in v_dims and 'nlon' in v_dims):
            dso[v] = ds[v]
        else:
            # Determine and sort other dimensions
            other_dims = set(v_dims) - {'nlat', 'nlon'}
            other_dims = tuple([d for d in v_dims if d in other_dims])
            lon_dim = ds[v].dims.index('nlon')
            field = ds[v].data
            field = np.concatenate((field, field), lon_dim)
            field = field[..., :, xL:xR]
            field = np.concatenate((field, field[..., :, 0:1]), lon_dim)
            dso[v] = xr.DataArray(field, dims=other_dims + ('nlat', 'nlon'), attrs=ds[v].attrs)

    # Copy coords
    for v, da in ds.coords.items():
        if not ('nlat' in da.dims and 'nlon' in da.dims):
            dso = dso.assign_coords(**{v: da})
    return dso
