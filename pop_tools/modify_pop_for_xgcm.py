"""
Module for modifying pop model output to be compatible with xgcm

# Ref: https://gist.github.com/rabernat/933bc785c99828352f343e48d0da6e22
"""

import numpy as np
import xarray as xr


def _add_pop_dims_to_dataset(ds):
    ds_new = ds.copy()
    ds_new['nlon_u'] = xr.Variable(
        ('nlon_u'), np.arange(len(ds.nlon)) + 1, {'axis': 'X', 'c_grid_axis_shift': 0.5}
    )
    ds_new['nlat_u'] = xr.Variable(
        ('nlat_u'), np.arange(len(ds.nlat)) + 1, {'axis': 'Y', 'c_grid_axis_shift': 0.5}
    )
    ds_new['nlon_t'] = xr.Variable(('nlon_t'), np.arange(len(ds.nlon)) + 0.5, {'axis': 'X'})
    ds_new['nlat_t'] = xr.Variable(('nlat_t'), np.arange(len(ds.nlat)) + 0.5, {'axis': 'Y'})

    # add metadata to z grid
    ds_new['z_t'].attrs.update({'axis': 'Z'})
    ds_new['z_w'].attrs.update({'axis': 'Z', 'c_grid_axis_shift': -0.5})
    ds_new['z_w_top'].attrs.update({'axis': 'Z', 'c_grid_axis_shift': -0.5})
    ds_new['z_w_bot'].attrs.update({'axis': 'Z', 'c_grid_axis_shift': 0.5})

    return ds_new


def _dims_from_grid_loc(grid_loc):
    grid_loc = str(grid_loc)
    ndim = int(grid_loc[0])
    x_loc_key = int(grid_loc[1])
    y_loc_key = int(grid_loc[2])
    z_loc_key = int(grid_loc[3])

    x_loc = {1: 'nlon_t', 2: 'nlon_u'}[x_loc_key]
    y_loc = {1: 'nlat_t', 2: 'nlat_u'}[y_loc_key]
    z_loc = {0: 'surface', 1: 'z_t', 2: 'z_w', 3: 'z_w_bot', 4: 'z_t_150m'}[z_loc_key]

    if ndim == 3:
        return z_loc, y_loc, x_loc
    elif ndim == 2:
        return y_loc, x_loc


def _label_coord_grid_locs(ds):
    grid_locs = {
        'ANGLE': '2220',
        'ANGLET': '2110',
        'DXT': '2110',
        'DXU': '2220',
        'DYT': '2110',
        'DYU': '2220',
        'DZU': '3221',
        'DZT': '3111',
        'HT': '2110',
        'HU': '2220',
        'HTE': '2210',
        'HTN': '2120',
        'HUS': '2210',
        'HUW': '2120',
        'KMT': '2110',
        'KMU': '2220',
        'REGION_MASK': '2110',
        'TAREA': '2110',
        'TLAT': '2110',
        'TLONG': '2110',
        'UAREA': '2220',
        'ULAT': '2220',
        'ULONG': '2220',
    }
    ds_new = ds.copy()
    for vname, grid_loc in grid_locs.items():
        if vname in ds.variables:
            ds_new[vname].attrs['grid_loc'] = grid_loc
    return ds_new


def relabel_pop_dims(ds):
    """Return a new xarray dataset with distinct dimensions for variables at different
    grid points.
    """
    ds_new = _label_coord_grid_locs(ds)
    ds_new = _add_pop_dims_to_dataset(ds_new)
    for vname in ds_new.variables:
        if 'grid_loc' in ds_new[vname].attrs:
            da = ds_new[vname]
            dims_orig = da.dims
            new_spatial_dims = _dims_from_grid_loc(da.attrs['grid_loc'])
            if dims_orig[0] == 'time':
                dims = ('time',) + new_spatial_dims
            else:
                dims = new_spatial_dims
            ds_new[vname] = xr.Variable(dims, da.data, da.attrs, da.encoding, fastpath=True)
    return ds_new
