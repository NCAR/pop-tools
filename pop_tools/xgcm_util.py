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

    x_loc = {1: 'nlon_t', 2: 'nlon_u', 3: 'nlon_u'}[x_loc_key]
    y_loc = {1: 'nlat_t', 2: 'nlat_u', 3: 'nlat_t'}[y_loc_key]
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


def get_xgcm_grid(ds):
    """Return an xgcm Grid object

    Parameters
    ----------
    ds : xarray.Dataset
        An xarray Dataset

    Returns
    -------
    grid : xgcm.Grid
        An xgcm Grid object

    Examples
    --------
    >>> from pop_tools import get_xgcm_grid
    >>> import xarray as xr
    >>> ds = xr.open_dataset("./tests/data/cesm_pop_monthly.T62_g17.nc")
    >>> ds
    <xarray.Dataset>
    Dimensions:       (d2: 2, lat_aux_grid: 395, nlat: 384, nlon: 320, time: 1, z_t: 60)
    Coordinates:
        TLAT          (nlat, nlon) float64 ...
        TLONG         (nlat, nlon) float64 ...
        ULAT          (nlat, nlon) float64 ...
        ULONG         (nlat, nlon) float64 ...
    * lat_aux_grid  (lat_aux_grid) float32 -79.48815 -78.952896 ... 89.47441 90.0
    * time          (time) object 0173-01-01 00:00:00
    * z_t           (z_t) float32 500.0 1500.0 2500.0 ... 512502.8 537500.0
    Dimensions without coordinates: d2, nlat, nlon
    Data variables:
        SALT          (time, z_t, nlat, nlon) float32 ...
        TEMP          (time, z_t, nlat, nlon) float32 ...
        UVEL          (time, z_t, nlat, nlon) float32 ...
        VVEL          (time, z_t, nlat, nlon) float32 ...
        time_bound    (time, d2) object ...
    Attributes:
        title:             g.e21.G1850ECOIAF.T62_g17.004
        history:           Sun May 26 14:13:02 2019: ncks -4 -L 9 cesm_pop_monthl...
        Conventions:       CF-1.0; http://www.cgd.ucar.edu/cms/eaton/netcdf/CF-cu...
        time_period_freq:  month_1
        model_doi_url:     https://doi.org/10.5065/D67H1H0V
        contents:          Diagnostic and Prognostic Variables
        source:            CCSM POP2, the CCSM Ocean Component
        revision:          $Id: tavg.F90 90507 2019-01-18 20:54:19Z altuntas@ucar...
        calendar:          All years have exactly  365 days.
        start_time:        This dataset was created on 2019-05-26 at 11:20:07.5
        cell_methods:      cell_methods = time: mean ==> the variable values are ...
        NCO:               netCDF Operators version 4.7.4 (http://nco.sf.net)
    >>> grid = get_xgcm_grid(ds)
    >>> grid
    <xgcm.Grid>

    """
    import xgcm

    grid = xgcm.Grid(ds)
    return grid
