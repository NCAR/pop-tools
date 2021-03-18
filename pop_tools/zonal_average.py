import os
import warnings

import numpy as np
import xarray as xr
import xesmf as xe

from .grid import get_grid


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

    # Check to see if there is a weights file already existing
    # Use xESMF to generate the destination grid
    return xe.util.grid_global(dx, dy)


def _get_default_filename(src_grid, dst_grid, method):

    # Get the source grid shape
    src_shape = src_grid.lat.shape

    # Get the destination grid shape
    dst_shape = dst_grid.lat.shape

    filename = '{0}_{1}x{2}_{3}x{4}.nc'.format(
        method, src_shape[0], src_shape[1], dst_shape[0], dst_shape[1]
    )

    return filename


def _convert_to_xesmf(data_ds, grid_ds):
    """
    Format xarray datasets to be read in easily to xESMF

    Parameters
    ----------
    data_ds : `xarray.Dataset`
      Dataset which includes fields to regrid

    grid_ds : `xarray.Dataset`
      Dataset including the POP grid

    Returns
    -------

    out_ds : `xarray.Dataset`
      Clipped dataset including fields to regrid with grid

    """

    # Merge datasets into single dataset
    data_ds = xr.merge(
        [grid_ds.reset_coords(), data_ds.reset_coords()], compat='override', join='right'
    ).rename({'TLAT': 'lat', 'TLONG': 'lon'})

    # Inlcude only points that will have surrounding corners
    data_ds = data_ds.isel({'nlon': data_ds.nlon[1:], 'nlat': data_ds.nlat[1:]})

    # Use ulat and ulong values as grid corners, rename variables to match xESMF syntax
    grid_corners = grid_ds[['ULAT', 'ULONG']].rename(
        {'nlat': 'nlat_b', 'nlon': 'nlon_b', 'ULAT': 'lat_b', 'ULONG': 'lon_b'}
    )

    # Merge datasets with data and grid corner information
    out_ds = xr.merge([data_ds, grid_corners])

    return out_ds


def _generate_weights(src_grid, dst_grid, method, weight_file=None, clobber=False):
    """
    Generate regridding weights by calling xESMF
    """

    # Allow user to input weights file, if there is not one, use default check
    if weight_file is None:
        weight_file = _get_default_filename(src_grid, dst_grid, method)

    # Check to see if the weights file already exists - if not, generate weights
    if not os.path.exists(weight_file) or clobber:
        xe.Regridder(src_grid, dst_grid, method).to_netcdf(weight_file)

    regridder = xe.Regridder(src_grid, dst_grid, method, weights=weight_file)

    return regridder


class regridder(object):
    def __init__(
        self,
        grid_name,
        grid=None,
        dx=None,
        dy=None,
        mask=None,
        regrid_method='conservative',
        method_gen_grid='regular_lat_lon',
    ):
        """
        A regridding class which uses xESMF and Xarray tools to both regrid and
        calculate a zonal averge.

        Parameters
        ----------
        grid_name
        """
        if grid_name is not None:
            self.grid_name = grid_name

            # Use pop-tools to retrieve the grid
            self.grid = get_grid(grid_name)

        elif grid is not None:
            self.grid = grid

        else:
            raise ValueError('Failed to input grid name or grid dataset')

        # Set the dx/dy parameters for generating the grid
        self.dx = dx
        self.dy = dy

        # Set the regridding method
        self.regrid_method = regrid_method

        # Set the grid generation method
        self.method_gen_grid = method_gen_grid

        # If the user does not input a mask, use default mask
        if not mask:
            self.mask = self.grid['REGION_MASK']
            self.mask_labels = self.grid['region_name']

        else:
            self.mask = mask

    # Setup method for regridding a dataarray
    def _regrid_dataarray(self, da_in, regrid_mask=False, regrid_method=None):

        src_grid = _convert_to_xesmf(da_in, self.grid)
        dst_grid = _generate_dest_grid(self.dy, self.dx, self.method_gen_grid)

        # If the user does not specify a regridding method, use default conservative
        if regrid_method is None:
            regridder = _generate_weights(src_grid, dst_grid, self.regrid_method)

        else:
            regridder = _generate_weights(src_grid, dst_grid, regrid_method)

        # Regrid the input data array, assigning the original attributes
        da_out = regridder(src_grid[da_in.name])
        da_out.attrs = da_in.attrs

        return da_out

    def regrid(self, obj, **kwargs):
        """generic interface for regridding DataArray or Dataset"""
        if isinstance(obj, xr.Dataset):
            return obj.map(self._regrid_dataarray, keep_attrs=True, **kwargs)
        elif isinstance(obj, xr.DataArray):
            return self._regrid_dataarray(obj, **kwargs)
        else:
            raise ValueError('unknown type')

    def za(self, obj, vertical_average=False, **kwargs):

        data = self.regrid(obj, **kwargs)
        mask = self.regrid(self.mask, regrid_method='nearest_s2d', **kwargs)

        # Store the various datasets seperated by basin in this list
        ds_list = []
        for region in np.unique(mask):

            if region != 0:
                ds_list.append(data.where(mask == region).groupby('lat').mean())

        # Merge the datasets
        out = xr.concat(ds_list, dim='nreg')

        # Check to see if a weighted vertical average is needed
        if vertical_average:

            # Run the vertical, weighted average
            out = out.weighted(out['z_t'].fillna(0)).mean(dim=['z_t'])

        # Add in the region name
        out['region_name'] = data.region_name

        return out
