import numpy as np
import pkg_resources
import xarray as xr
import yaml

from .grid import get_grid, grid_defs

region_def_file = pkg_resources.resource_filename('pop_tools', 'region_mask_definitions.yaml')
# open defined region masks
with open(region_def_file) as f:
    all_region_defs = yaml.safe_load(f)


def region_mask_3d(grid_name, mask_name=None, region_defs=None):
    """Generate as 3-D region mask with dimensions ('region', 'nlat', 'nlon').

    Parameters
    ----------

    grid_name : str
        Name of grid (i.e., POP_gx3v7, POP_gx1v7, POP_tx0.1v3)

    mask_name : str, optional
        Name of the pre-defined mask. If not provided and `region_defs` is
        not provided, the default `REGION_MASK` is used.

    region_defs : dict, optional
        Dictionary containing region definitions. The dictionary has the
        following form:

            region_defs = {region1_name: list_of_criteria_dicts_1, region2_name: list_of_criteria_dicts_2,...}

        `list_of_criteria_dicts` is a list of dictionaries; each must include
        the keys 'match' or 'bounds'. For instance:

            list_of_criteria_dicts = [{'match': {'REGION_MASK': [1, 2, 3, 6]}, 'bounds': {'TLAT': [-90., -30.]}}]

        will identify a region where the default POP grid `REGION_MASK` values match
        the specified list and `TLAT` is between 90°S and 30°S. Multiple entries
        in the `list_of_criteria_dicts` are applied with an "or" condition.

    Returns
    -------

    mask3d : xarray.DataArray
        Region mask DataArray with dimensions ('region', 'nlat', 'nlon').
        `mask3d` has "ones" within a region and "zeroes" outside it. A `region`
        coordinate is included that contains the region names.
    """
    ds = get_grid(grid_name)
    nlat, nlon = ds.REGION_MASK.shape

    if region_defs is None and mask_name is None:
        mask_name = 'default'

    elif region_defs is not None and mask_name is None:
        mask_name = 'user-defined'

    if region_defs is None:
        region_defs = _get_region_definitions(ds, grid_name, mask_name)

    _verify_region_def_schema(region_defs)

    region_names = list(region_defs.keys())
    region_coord = xr.DataArray(region_names, dims=('region'))

    mask_shape = region_coord.shape + (nlat, nlon)
    mask_dims = region_coord.dims + ds.REGION_MASK.dims
    mask3d = xr.DataArray(
        np.zeros(mask_shape),
        dims=mask_dims,
        coords={'region': region_coord},
        attrs={'mask_name': mask_name},
    )

    not_land = np.where(ds.KMT > 0, 1, 0)

    # loop over region names
    for i, (region_name, crit_list) in enumerate(region_defs.items()):

        # loop over list of criteria groups
        mask = np.zeros((nlat, nlon), dtype=bool)
        for crit in crit_list:

            mask_and = np.ones((nlat, nlon), dtype=bool)

            # loop over each criterion
            for crit_type, crit_dict in crit.items():

                # apply bounds or match
                if crit_type == 'bounds':
                    for field, bounds in crit_dict.items():
                        if bounds[0] > bounds[1]:
                            mask_and = (bounds[0] <= ds[field]) | (
                                ds[field] <= bounds[1]
                            ) & mask_and
                        else:
                            mask_and = (
                                (bounds[0] <= ds[field]) & (ds[field] <= bounds[1]) & mask_and
                            )

                elif crit_type == 'match':
                    mask_or = np.zeros((nlat, nlon), dtype=bool)
                    for field, value in crit_dict.items():
                        for value_i in value:
                            mask_or = (ds[field] == value_i) | mask_or
                    mask_and = mask_or & mask_and

                else:
                    raise ValueError(f'unknown criteria type: {crit_type}')

            # update mask logic
            mask = mask | mask_and

        # make mask
        mask3d[i, :, :] = xr.where(mask & not_land, 1.0, 0.0)

    return mask3d


def _verify_region_def_schema(region_defs):
    """Ensure region_defs dictionary conforms to expected schema."""
    # loop over region names
    for i, (region_name, crit_list) in enumerate(region_defs.items()):
        assert isinstance(crit_list, list)
        # loop over list of criteria groups
        for crit in crit_list:
            assert isinstance(crit, dict)
            # loop over each criterion
            for crit_type, crit_dict in crit.items():
                assert crit_type in ['bounds', 'match']


def _get_region_definitions(ds, grid_name, mask_name):
    """Return the region mask definition from `region_mask_definitions.yaml`."""
    if mask_name == 'default':
        grid_attrs = grid_defs[grid_name]

        regions = np.unique(ds.REGION_MASK)
        regions = regions[regions != 0]

        if 'region_mask_regions' in grid_attrs:
            region_names = list(grid_attrs['region_mask_regions'].keys())
            region_index = list(grid_attrs['region_mask_regions'].values())

        else:
            region_names = regions
            region_index = regions

        return {k: [{'match': {'REGION_MASK': [v]}}] for k, v in zip(region_names, region_index)}
    else:
        if mask_name not in all_region_defs[grid_name]:
            raise ValueError(
                f'''unknown region mask: {mask_name}
                             the following regions masks are defined: {list_region_masks(grid_name)}
                             '''
            )

        return all_region_defs[grid_name][mask_name]


def list_region_masks(grid_name):
    """Get a list of the pre-defined region masks.

    Parameters
    ----------

    grid_name : str
        Name of grid (i.e., POP_gx3v7, POP_gx1v7, POP_tx0.1v3)

    Returns
    -------

    region_mask_list : list
        List of defined region mask names.
    """
    if grid_name in all_region_defs:
        return list(all_region_defs[grid_name].keys())
    else:
        return []
