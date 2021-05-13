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
    if 'z_t' in ds_new.variables:
        ds_new['z_t'].attrs.update({'axis': 'Z'})
    if 'z_w' in ds_new.variables:
        ds_new['z_w'].attrs.update({'axis': 'Z', 'c_grid_axis_shift': -0.5})
    if 'z_w_top' in ds_new.variables:
        ds_new['z_w_top'].attrs.update({'axis': 'Z', 'c_grid_axis_shift': -0.5})
    if 'z_w_bot' in ds_new.variables:
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
            dims = dims_orig[: -len(new_spatial_dims)] + new_spatial_dims
            ds_new[vname] = xr.Variable(dims, da.data, da.attrs, da.encoding, fastpath=True)
    old_coords = ['nlat', 'nlon']
    for coord in old_coords:
        if coord in ds_new.coords:
            ds_new = ds_new.drop_vars(coord)
    if 'z_w_top' in ds_new.dims and 'z_w' in ds_new.dims:
        ds_new = ds_new.drop('z_w_top').rename({'z_w': 'z_w_top'})
    return ds_new


def to_xgcm_grid_dataset(ds, **kwargs):
    """Modify POP model output to be compatible with xgcm.

    Parameters
    ----------
    ds : xarray.Dataset
        An xarray Dataset
    kwargs:
       Additional keyword arguments are passed through to `xgcm.Grid` class.

    Returns
    -------
    grid : xgcm.Grid
        An xgcm Grid object
    ds_new : xarray.Dataset
        Xarray dataset with distinct dimensions for variables at different
        grid points.

    Examples
    --------
    >>> from pop_tools import DATASETS
    >>> import xarray as xr
    >>> filepath = DATASETS.fetch("g.e20.G.TL319_t13.control.001_hfreq-coarsen.nc")
    >>> ds = xr.open_dataset(filepath)
    >>> ds
    <xarray.Dataset>
    Dimensions:          (d2: 2, nlat: 240, nlon: 360, time: 1, z_t: 62, z_t_150m: 15, z_w: 62, z_w_top: 62)
    Coordinates:
    * time             (time) object 0050-01-01 00:00:00
    * z_t              (z_t) float32 500.0 1500.0 2500.0 ... 562499.06 587499.06
    * z_t_150m         (z_t_150m) float32 500.0 1500.0 2500.0 ... 13500.0 14500.0
    * z_w              (z_w) float32 0.0 1000.0 2000.0 ... 549999.06 574999.06
    * z_w_top          (z_w_top) float32 0.0 1000.0 2000.0 ... 549999.06 574999.06
        TLONG            (nlat, nlon) float64 ...
        TLAT             (nlat, nlon) float64 ...
    Dimensions without coordinates: d2, nlat, nlon
    Data variables:
        dz               (z_t) float32 ...
        dzw              (z_w) float32 ...
        KMT              (nlat, nlon) float64 ...
        REGION_MASK      (nlat, nlon) float64 ...
        TAREA            (nlat, nlon) float64 ...
        DXU              (nlat, nlon) float64 ...
        DYU              (nlat, nlon) float64 ...
        DYT              (nlat, nlon) float64 ...
        time_bound       (time, d2) object ...
        UEU              (time, z_t, nlat, nlon) float32 ...
        VNU              (time, z_t, nlat, nlon) float32 ...
        S_FLUX_ROFF_VSF  (time, z_w_top, nlat, nlon) float32 ...
    >>> grid, ds_new = to_xgcm_grid_dataset(ds)
    >>> grid
    <xgcm.Grid>
    X Axis (periodic):
        * center   nlon_t --> right
        * right    nlon_u --> center
    Z Axis (periodic):
        * center   z_t --> left
        * left     z_w_top --> center
    Y Axis (periodic):
        * center   nlat_t --> right
        * right    nlat_u --> center
    >>> ds_new
    <xarray.Dataset>
    Dimensions:          (d2: 2, nlat_t: 240, nlat_u: 240, nlon_t: 360, nlon_u: 360, time: 1, z_t: 62, z_t_150m: 15, z_w: 62, z_w_top: 62)
    Coordinates:
    * time             (time) object 0050-01-01 00:00:00
    * z_t              (z_t) float32 500.0 1500.0 2500.0 ... 562499.06 587499.06
    * z_t_150m         (z_t_150m) float32 500.0 1500.0 2500.0 ... 13500.0 14500.0
    * z_w              (z_w) float32 0.0 1000.0 2000.0 ... 549999.06 574999.06
    * z_w_top          (z_w_top) float32 0.0 1000.0 2000.0 ... 549999.06 574999.06
      TLONG            (nlat_t, nlon_t) float64 nan nan nan nan ... nan nan nan
      TLAT             (nlat_t, nlon_t) float64 nan nan nan nan ... nan nan nan
    * nlon_u           (nlon_u) int64 1 2 3 4 5 6 7 ... 355 356 357 358 359 360
    * nlat_u           (nlat_u) int64 1 2 3 4 5 6 7 ... 235 236 237 238 239 240
    * nlon_t           (nlon_t) float64 0.5 1.5 2.5 3.5 ... 357.5 358.5 359.5
    * nlat_t           (nlat_t) float64 0.5 1.5 2.5 3.5 ... 237.5 238.5 239.5
    Dimensions without coordinates: d2
    Data variables:
        dz               (z_t) float32 ...
        dzw              (z_w) float32 ...
        KMT              (nlat_t, nlon_t) float64 nan nan nan nan ... nan nan nan
        REGION_MASK      (nlat_t, nlon_t) float64 nan nan nan nan ... nan nan nan
        TAREA            (nlat_t, nlon_t) float64 nan nan nan nan ... nan nan nan
        DXU              (nlat_u, nlon_u) float64 nan nan nan nan ... nan nan nan
        DYU              (nlat_u, nlon_u) float64 nan nan nan nan ... nan nan nan
        DYT              (nlat_t, nlon_t) float64 nan nan nan nan ... nan nan nan
        time_bound       (time, d2) object ...
        UEU              (time, z_t, nlat_u, nlon_u) float32 inf inf inf ... inf inf
        VNU              (time, z_t, nlat_t, nlon_u) float32 inf inf inf ... inf inf
        S_FLUX_ROFF_VSF  (time, z_w, nlat_t, nlon_t) float32 inf inf inf ... inf inf

    """

    try:
        import xgcm
    except ImportError:
        raise ImportError(
            """to_xgcm_grid_dataset() function requires the `xgcm` package. \nYou can install it via PyPI or Conda"""
        )
    ds_new = relabel_pop_dims(ds)
    grid = xgcm.Grid(ds_new, **kwargs)
    return grid, ds_new
