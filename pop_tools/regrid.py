import os
import warnings

import cf_xarray
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

from .grid import _compute_corners, get_grid


def gen_corner_calc(ds, cell_corner_lat='ULAT', cell_corner_lon='ULONG'):
    """
    Generates corner information and creates single dataset with output
    """

    cell_corner_lat = ds[cell_corner_lat]
    cell_corner_lon = ds[cell_corner_lon]
    # Use the function in pop-tools to get the grid corner information
    corn_lat, corn_lon = _compute_corners(cell_corner_lat, cell_corner_lon)

    lon_shape, lat_shape = corn_lon[:, :, 0].shape
    out_shape = (lon_shape + 1, lat_shape + 1)

    # Generate numpy arrays to store destination lats/lons
    out_lons = np.zeros(out_shape)
    out_lats = np.zeros(out_shape)

    # Assign the northeast corner information
    out_lons[1:, 1:] = corn_lon[:, :, 0]
    out_lats[1:, 1:] = corn_lat[:, :, 0]

    # Assign the northwest corner information
    out_lons[1:, :-1] = corn_lon[:, :, 1]
    out_lats[1:, :-1] = corn_lat[:, :, 1]

    # Assign the southwest corner information
    out_lons[:-1, :-1] = corn_lon[:, :, 2]
    out_lats[:-1, :-1] = corn_lat[:, :, 2]

    # Assign the southeast corner information
    out_lons[:-1, 1:] = corn_lon[:, :, 3]
    out_lats[:-1, 1:] = corn_lat[:, :, 3]

    return out_lats, out_lons


def _add_region_mask(ds, grid, cell_center='TLAT'):
    """
    Grabs and combines a grid
    """

    grid = get_grid(grid)

    return xr.merge([ds, grid[['REGION_MASK', 'region_name']]])


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
        # Get the source grid shape
        src_shape = src_grid.lat.shape

        # Get the destination grid shape
        dst_shape = dst_grid.lat.shape

        weight_file = f'{method}_{src_shape[0]}x{src_shape[1]}_{dst_shape[0]}x{dst_shape[1]}.nc'

    # Check to see if the weights file already exists - if not, generate weights
    if not os.path.exists(weight_file):
        regridder = xe.Regridder(src_grid, dst_grid, method, ignore_degenerate=True)
        print(f'Saving weights file: {os.path.abspath(weight_file)}')
        regridder.to_netcdf(weight_file)

    else:
        regridder = xe.Regridder(
            src_grid, dst_grid, method, weights=weight_file, ignore_degenerate=True
        )

    return regridder


# Setup method for regridding a dataarray
def _regrid_dataset(da_in, dst_grid, regrid_method='conservative'):
    src_grid = _convert_to_xesmf(da_in)

    # If the user does not specify a regridding method, use default conservative
    regridder = _generate_weights(src_grid, dst_grid, method=regrid_method)

    # Regrid the input data array, assigning the original attributes
    da_out = regridder(src_grid)
    da_out.attrs = da_in.attrs

    return da_out.drop(['lat_b', 'lon_b'])


def _convert_to_xesmf(ds):

    if isinstance(ds, xr.DataArray):
        grid_center_lat = ds.cf['latitude'].name
        grid_center_lon = ds.cf['longitude'].name
        ds = ds.to_dataset()

    elif isinstance(ds, xr.Dataset):
        var = list(ds.variables)[0]
        grid_center_lat = ds[var].cf['latitude'].name
        grid_center_lon = ds[var].cf['longitude'].name

    else:
        TypeError('input data must be xarray DataArray or xarray Dataset!')

    if grid_center_lat == 'TLAT':

        # Use u-grid as corners
        grid_corner_lat = 'ULAT'
        grid_corner_lon = 'ULONG'

        # Find grid corners
        corner_lats, corner_lons = gen_corner_calc(
            ds, cell_corner_lat=grid_corner_lat, cell_corner_lon=grid_corner_lon
        )

    elif grid_center_lat == 'ULAT':

        # Use t-grid as corners
        grid_corner_lat = 'TLAT'
        grid_corner_lon = 'TLONG'

        # Find grid corners
        ds = ds.isel({'nlon': ds.nlon[:-1], 'nlat': ds.nlat[:-1]})

        warnings.warn('Regridding using u-grid as cell center')

    # Rename the dimensions
    ds = ds.rename(
        {
            grid_center_lat: 'lat',
            grid_center_lon: 'lon',
            grid_corner_lat: 'lat_b',
            grid_corner_lon: 'lon_b',
        }
    )

    # Add the grid corner information
    ds['lat_b'] = (['nlat_b', 'nlon_b'], corner_lats)
    ds['lon_b'] = (['nlat_b', 'nlon_b'], corner_lons)

    return ds.set_coords(['lat', 'lon', 'lat_b', 'lon_b'])


def _prep_for_xesmf(ds, grid, lat_axis=None, cell_center='TLAT', cell_corners='ULAT'):
    """
    Prepares data for xesmf by - adding the grid
    """

    # add in the region mask
    ds = _add_region_mask(ds, grid)

    # Calculate the grid corners
    if cell_center == 'TLAT':
        corner_lats, corner_lons = gen_corner_calc(
            ds, cell_corner_lat='ULAT', cell_corner_lon='ULONG'
        )

    else:
        return ValueError('Does not support center coordinates other than TLAT')

    # Rename the dimensions
    ds = ds.rename({'TLAT': 'lat', 'TLONG': 'lon', 'ULAT': 'lat_b', 'ULONG': 'lon_b'})

    # Add the corner lats/lons
    ds['lat_b'] = (['nlat_b', 'nlon_b'], corner_lats)
    ds['lon_b'] = (['nlat_b', 'nlon_b'], corner_lons)

    return ds.set_coords(['lat', 'lon', 'lat_b', 'lon_b']).drop(['lat_aux_grid'])


def gen_dest_grid(
    method, dx=0.25, dy=0.25, lat_axis_bnds=None, lat_axis=None, lon_axis_bnds=None, lon_axis=None
):
    """Generates a destination grid to use with xesmf

    Parameters
    ----------
    method: str
       Method to use for generating the destination grid, options include 'regular_grid' or 'lat_axis'

    dx: float default = 0.25
       Longitudinal grid spacing to use in regular grid

    dy: float default = 0.25
       Latitudinal grid spacing to use in regular grid

    lat_axis_bnds: list, array
       Latitude axis bounds to use when defining the destination latitudes

    lat_axis: list, array
       Latitude axis to use when defining the destination latitudes

    lon_axis_bounds: list, array
       Longitude axis bounds to use when defining the destination longitudes

    lon_axis: list, array
       Longitude axis to use when defining the destination longitudes

    Returns
    -------
    xarray.Dataset
      An xarray.Dataset with the associated coordinates, formatted with lat, lon, lat_b, lon_b
    """

    reg_grid = xe.util.grid_global(dx, dy)

    if method == 'regular_grid':
        out_ds = reg_grid

    elif method == 'lat_aux_grid':
        out_ds = xr.Dataset()

        if lat_axis_bnds is None:
            raise TypeError('Missing defined latitude axis bounds')
        lat_b = lat_axis_bnds
        if lat_axis is not None:
            lat = lat_axis
        else:
            lat = (lat_axis_bnds[1:] + lat_axis_bnds[:-1]) / 2

        if lon_axis_bnds is None:
            lon_b = reg_grid.lon_b.values[0, :]

        if lon_axis is None:
            lon = reg_grid.lon.values[0, :]

        else:
            raise TypeError('Missing defined longitude axis bounds')

        lons, lats = np.meshgrid(lon, lat)
        lons_b, lats_b = np.meshgrid(lon_b, lat_b)

        out_ds['lat'] = (('y', 'x'), lats)
        out_ds['lon'] = (('y', 'x'), lons)
        out_ds['lat_b'] = (('y_b', 'x_b'), lats_b)
        out_ds['lon_b'] = (('y_b', 'x_b'), lons_b)

    else:
        raise ValueError(
            f"Expected method to be one of ('regular_grid', 'define_lat_aux'). Received {method} instead."
        )

    return out_ds.set_coords(['lat', 'lat_b', 'lon', 'lon_b'])


def to_uniform_grid(obj, dst_grid, regrid_method='conservative'):
    """
    Transform the POP C-Grid to a regular lat-lon grid, using similar grid spacing

    Parameters
    ----------
    obj: `xarray.Dataset` or `xarray.DataArray`
         Input dataset or dataarray to regrid
    dx : float
         Longitudinal grid spacing for the output grid

    dy : float
         Latitudinal grid spacing for the output grid

    regrid_method : str
         Regridding method for xESMF, default is conservative

    **kwargs
    """

    # Convert input dataset to be ready for xESMF
    if isinstance(obj, xr.Dataset):

        scalar_vars = []

        for var in obj:
            if (
                'nlat' in obj[var].dims
                and 'nlon' in obj[var].dims
                and obj[var].cf['latitude'].name == 'TLAT'
            ):

                scalar_vars.append(var)

            else:
                pass

        if scalar_vars:
            out = _regrid_dataset(obj[scalar_vars], dst_grid, regrid_method)

        else:
            raise ValueError('Input has no variables at scalar points (TLON, TLAT)')

        return out

    elif isinstance(obj, xr.DataArray):
        return _regrid_dataset(obj, dst_grid, regrid_method)

    raise TypeError('input data must be xarray DataArray or xarray Dataset!')


def zonal_average(
    data,
    grid_name,
    dest_grid_method='regular_grid',
    lat_axis=None,
    lat_axis_bnds=None,
    lon_axis=None,
    lon_axis_bnds=None,
    dx=0.25,
    dy=0.25,
):
    """
    Computes zonal average of some pop-dataset using some mask grid name.

    Parameters
    ----------
    data: `xarray.Dataset`
           Input dataset containing

    grid_name: str
           POP-grid name

    dest_grid_method: str, {'regular_grid', 'lat_aux_grid'}, default = 'regular_grid'
       Method to use for generating the destination grid. Two main options include 'regular grid',
       which is regular lat-lon grid, or 'lat_axis' which uses the 'lat_aux_grid' specified in the file

    lat_axis: list, array, optional
       Latitude axis to use for regridding

    lat_axis_bnds: list, array, optional
       Latitude axis bounds to use for regridding

    lon_axis: list, array, optional
       Longitudinal axis to use for regridding

    lon_axis_bnds: list, array, optional
       Longitudinal axis bounds to use for regridding

    dx = float
         Longitudinal grid spacing for output grid when using regular_grid

    dy = float
         Latitudinal grid spacing for output grid when using regular grid
    """

    # Read in the grid file - set coordinates to TLAT and TLONG
    data = _add_region_mask(data, grid_name)

    # Add the latitude axis if needed
    if (
        (dest_grid_method == 'lat_aux_grid')
        and ('lat_aux_grid' in data.variables)
        and (lat_axis is None)
    ):
        lat_axis_bnds = data.lat_aux_grid.values

    # Generate the destination grid
    dst_grid = gen_dest_grid(
        dest_grid_method, dx, dy, lat_axis_bnds, lat_axis, lon_axis_bnds, lon_axis
    )

    # Convert the mask to a uniform grid, using nearest neighbor interpolation
    mask_regrid = to_uniform_grid(
        data[['REGION_MASK']], dst_grid, regrid_method='nearest_s2d'
    ).REGION_MASK
    weights_regrid = to_uniform_grid(
        data[['REGION_MASK']], dst_grid, regrid_method='conservative'
    ).REGION_MASK

    # Add a mask to the regridding
    dst_grid['mask'] = np.logical_not(mask_regrid)

    # Convert the data to a uniform grid, using default conservative regridding
    data_regrid = to_uniform_grid(data, dst_grid)

    ds_list = [
        data_regrid.where(mask_regrid == region).weighted(weights_regrid).mean('x')
        for region in tqdm(np.unique(mask_regrid))
        if region != 0
    ]

    # Merge the datasets
    out = xr.concat(ds_list, dim='region')

    # Use the conventional basin names
    out['region'] = data.region_name.values

    # Fix the latitude values for the output dataset
    out = out.rename({'y': 'lat'})
    out['lat'] = data_regrid.lat.values[:, 0]

    return out.drop(['REGION_MASK'])
