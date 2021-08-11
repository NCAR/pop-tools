import os
import warnings

import numpy as np
import xarray as xr

try:
    import xesmf as xe
except ImportError:
    message = 'Zonal averaging requires xesmf package.\n\n'
    'Please conda install as follows:\n\n'
    ' conda install -c conda-forge xesmf>=0.4.0'

    raise ImportError(message)

from tqdm import tqdm

from .. import get_grid, region_mask_3d


def _generate_dest_grid(dy=None, dx=None, method_gen_grid='regular_lat_lon'):
    """
    Generates the destination grid

    Parameters
    ----------
    dy: float
      Horizontal grid spacing in y-direction (latitudinal)

    dy: float
      Horizontal grid spcaing in x-direction (longitudinal)
    """

    # Use regular lat/lon with regular spacing
    if method_gen_grid == 'regular_lat_lon':
        if dy is None:
            dy = 0.25

        if dx is None:
            dx = dy

    # Able to add other options at a later point
    else:
        raise ValueError(f'Input method_gen_grid: {method_gen_grid} is not supported.')

    # Use xESMF to generate the destination grid
    return xe.util.grid_global(dx, dy)


def _get_default_filename(src_grid, dst_grid, method):

    # Get the source grid shape
    src_shape = src_grid.lat.shape

    # Get the destination grid shape
    dst_shape = dst_grid.lat.shape

    filename = f'{method}_{src_shape[0]}x{src_shape[1]}_{dst_shape[0]}x{dst_shape[1]}.nc'

    return filename


def _generate_weights(src_grid, dst_grid, method, weight_file=None):
    """
    Generate regridding weights by calling xESMF
    """

    # Allow user to input weights file, if there is not one, use default check
    if weight_file is None:
        weight_file = _get_default_filename(src_grid, dst_grid, method)

    # Check to see if the weights file already exists - if not, generate weights
    if not os.path.exists(weight_file):
        regridder = xe.Regridder(src_grid, dst_grid, method)
        print(f'Saving weights file: {os.path.abspath(weight_file)}')
        regridder.to_netcdf(weight_file)

    else:
        regridder = xe.Regridder(src_grid, dst_grid, method, weights=weight_file, periodic=True)

    return regridder


# Setup method for regridding a dataarray
def _regrid_dataset(da_in, dst_grid, regrid_method=None):
    src_grid = _convert_to_xesmf(da_in)

    # If the user does not specify a regridding method, use default conservative
    if regrid_method is None:
        regridder = _generate_weights(src_grid, dst_grid, 'conservative')

    else:
        regridder = _generate_weights(src_grid, dst_grid, regrid_method)

    # Regrid the input data array, assigning the original attributes
    da_out = regridder(src_grid)
    da_out.attrs = da_in.attrs

    return da_out.drop(['lat_b', 'lon_b'])


def _convert_to_xesmf(ds):

    if isinstance(ds, xr.DataArray):
        grid_center_lat = ds.cf['latitude'].name
        grid_center_lon = ds.cf['longitude'].name

    elif isinstance(ds, xr.Dataset):
        var = list(ds.variables)[0]
        grid_center_lat = ds[var].cf['latitude'].name
        grid_center_lon = ds[var].cf['longitude'].name

    else:
        TypeError('input data must be xarray DataArray or xarray Dataset!')

    # Rename grid center lat/lon
    ds_rename = ds.rename({grid_center_lat: 'lat', grid_center_lon: 'lon'})

    if grid_center_lat == 'TLAT':

        # Use u-grid as corners
        grid_corner_lat = 'ULAT'
        grid_corner_lon = 'ULONG'

        # Find grid corners
        ds_rename = ds_rename.isel({'nlon': ds_rename.nlon[1:], 'nlat': ds_rename.nlat[1:]})

    elif grid_center_lat == 'ULAT':

        # Use t-grid as corners
        grid_corner_lat = 'TLAT'
        grid_corner_lon = 'TLONG'

        # Find grid corners
        ds_rename = ds_rename.isel({'nlon': ds_rename.nlon[:-1], 'nlat': ds_rename.nlat[:-1]})

        warnings.warn('Regridding using u-grid as cell center')

    grid_corners = ds.rename(
        {'nlat': 'nlat_b', 'nlon': 'nlon_b', grid_corner_lat: 'lat_b', grid_corner_lon: 'lon_b'}
    )

    # Merge datasets with data and grid corner information
    out_ds = xr.merge([ds_rename, grid_corners], compat='override')

    return out_ds.drop([grid_corner_lat, grid_corner_lon, grid_center_lat, grid_center_lon])


def to_uniform_grid(obj, dx=0.25, dy=0.25, regrid_method='conservative', **kwargs):
    """
    Transform the POP C-Grid to a regular lat-lon grid.

    Parameters
    ----------
    obj: `xarray.Dataset` or `xarray.DataArray`
         Input dataset or dataarray to regrid
    dx = float
         Longitudinal grid spacing for output grid

    dy = float
         Latitudinal grid spacing for output grid

    regrid_method = str
         Regridding method for xESMF, default is conservative

    **kwargs
    """

    # Generate destination grid
    dst_grid = _generate_dest_grid(dx, dy)

    # Convert input dataset to be ready for xESMF
    if isinstance(obj, xr.Dataset):

        scalar_vars = []

        for var in obj:
            if 'nlat' in obj[var].dims and 'nlon' in obj[var].dims:

                if obj[var].cf['latitude'].name == 'TLAT':
                    scalar_vars.append(var)

            else:
                None

        if len(scalar_vars) > 0:
            out = _regrid_dataset(obj[scalar_vars], dst_grid, regrid_method)

        else:
            raise AttributeError('Input no variables with nlat/nlon dimensions')

        return out

    elif isinstance(obj, xr.DataArray):
        return _regrid_dataset(obj, dst_grid, **kwargs)

    raise TypeError('input data must be xarray DataArray or xarray Dataset!')


def zonal_average(data, mask_grid_name):
    """
    Computes zonal average of some pop-dataset using some mask grid name.

    Parameters
    ----------
    data: `xarray.Dataset`
           Input dataset containing

    mask_grid_name: str
           POP-grid name
    """

    # Read in the grid file - set coordinates to TLAT and TLONG
    grid = get_grid(mask_grid_name).set_coords(['TLAT', 'TLONG'])

    # Merge the grid with the data, which will help with dealing with coordinates
    grid_ds = xr.merge([data, grid[['REGION_MASK']]], compat='override')

    # Convert the data to a uniform grid, using default conservative regridding
    data_regrid = to_uniform_grid(data)

    # Convert the mask to a uniform grid, using nearest neighbor interpolation
    mask_regrid = to_uniform_grid(grid_ds[['REGION_MASK']], regrid_method='nearest_s2d')

    ds_list = [
        data_regrid.where(mask_regrid.REGION_MASK == region).mean('lon')
        for region in tqdm(np.unique(mask_regrid.REGION_MASK))
        if region != 0
    ]

    # Merge the datasets
    out = xr.concat(ds_list, dim='region')

    # Use the conventional basin names
    out['region'] = grid.region_name.values

    return out
