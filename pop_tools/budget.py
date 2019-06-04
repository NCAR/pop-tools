import datetime
import glob

import numpy as np
import xarray as xr

from .grid import get_grid

NMOLS_TO_MOLYR = 1e-9 * 86400.0 * 365.0


def _process_grid(grid):
    """Load in POP grid and process coordinates required for budget calculation."""
    grid_f = get_grid(grid)
    dz = grid_f.dz.drop('z_t')
    area = (grid_f.TAREA).rename('area')
    vol = (area * dz).rename('vol')
    z = (grid_f.z_w_bot.drop('z_w_bot')).rename('z')
    mask = grid_f.REGION_MASK.rename('mask')
    return xr.merge([area, vol, z, mask])


def _compute_kmax(budget_depth, z):
    """Compute the k-index of the maximum budget depth"""
    kmax = (z <= budget_depth).argmin() - 1
    kmax = kmax.values
    # This only occurs when the full depth is requested (or `budget_depth` is greater than the maximum POP depth`)
    if kmax == -1:
        kmax = None
    return kmax


def _convert_to_tendency(ds, normalizer, sign=1, kmax=None):
    """Converts from volume- and area-dependent units to tendencies"""
    if (kmax is not None) and ('z_t' in normalizer.dims):
        normalizer = normalizer.isel(z_t=slice(0, kmax + 1))
    ds = ds * normalizer * sign
    ds.attrs['units'] = 'nmol/s'
    return ds


def _convert_units(ds):
    """Converts from nmol/s to mol/yr and adds attribute label"""
    ds *= NMOLS_TO_MOLYR
    ds.attrs['units'] = 'mol/yr'
    return ds


def _compute_horizontal_divergence(da, mask, direction=None):
    """Computes divergence of a tracer flux in the horizontal

    Parameters
    ----------
    da : `xarray.DataArray`
      DataArray with the tracer flux for which the divergence is being computed.
    direction : str
      Direction to compute divergence ('zonal' or 'meridional')
    """
    if direction == 'zonal':
        # Moves mask to the left of the region to compute divergence of flow
        # entering the volume.
        mask_roll = mask.roll(nlon=-1, roll_coords=True)
        div = da.where(mask_roll).roll(nlon=1, roll_coords=True) - da.where(mask)
    elif direction == 'meridional':
        mask_roll = mask.roll(nlat=-1, roll_coords=True)
        div = da.where(mask_roll).roll(nlat=1, roll_coords=True) - da.where(mask)
    else:
        raise ValueError("Please input either 'zonal' or 'meridional' for direction.")
    return div


def _compute_vertical_divergence(da):
    """Computes divergence of a tracer flux in the vertical"""
    ny, nx = da.nlat.size, da.nlon.size
    # Places a cap of zeros on top of the ocean. This makes it easy to use the `diff` function
    # with a positive z heading toward shallower depths.
    zero_cap = xr.DataArray(np.zeros((ny, nx)), dims=['nlat', 'nlon'])
    vdiv = xr.concat([zero_cap, da.drop('time')], dim='z_t')
    vdiv = vdiv.diff('z_t')
    return vdiv


def _compute_lateral_advection(ds, grid, mask, kmax=None):
    """Compute lateral advection component of budget"""
    print('Computing lateral advection...')
    ds = ds[['UE', 'VN']]
    if kmax is not None:
        ds = ds.isel(z_t=slice(0, kmax + 1))
    ds = ds.groupby('time.year').mean('time').rename({'year': 'time'})
    ds = _convert_to_tendency(ds, grid.vol, kmax=kmax)
    ladv_zonal = _compute_horizontal_divergence(ds.UE, mask, direction='zonal')
    ladv_merid = _compute_horizontal_divergence(ds.VN, mask, direction='meridional')
    ladv = (ladv_zonal + ladv_merid).rename('ladv').sum('z_t')
    ladv = _convert_units(ladv)
    ladv.attrs['long_name'] = 'lateral advection'
    return ladv.load()


def _compute_lateral_mixing(ds, grid, mask, kmax=None):
    """Compute lateral mixing component."""
    print('Computing lateral mixing...')
    ds = ds[['HDIFN', 'HDIFE', 'HDIFB']]
    if kmax is not None:
        ds = ds.isel(z_t=slice(0, kmax + 1))
    ds = ds.groupby('time.year').mean('time').rename({'year': 'time'})
    # Flip sign so that positive direction is upwards.
    ds = _convert_to_tendency(ds, grid.vol, sign=-1, kmax=kmax)
    lmix_zonal = _compute_horizontal_divergence(ds.HDIFE, mask, direction='zonal')
    lmix_merid = _compute_horizontal_divergence(ds.HDIFN, mask, direction='meridional')
    lmix_B = ds.HDIFB.where(mask)
    lmix_B = _compute_vertical_divergence(lmix_B)

    # Sum all lateral mixing components
    lmix = (lmix_merid + lmix_zonal + lmix_B).rename('lmix').sum('z_t')
    lmix = _convert_units(lmix)
    lmix.attrs['long_name'] = 'lateral mixing'
    return lmix.load()


def _compute_vertical_advection(ds, grid, mask, kmax=None):
    """Compute vertical advection (WT)"""
    print('Computing vertical advection...')
    ds = ds['WT']
    if kmax is not None:
        # Need one level below max depth to compute divergence into bottom layer.
        ds = ds.isel(z_t=slice(0, kmax + 2))
    ds = ds.where(mask)
    ds = ds.groupby('time.year').mean('time').rename({'year': 'time'})
    if kmax is not None:
        ds = _convert_to_tendency(ds, grid.vol, kmax=kmax + 1)
    else:
        ds = _convert_to_tendency(ds, grid.vol)

    # Compute divergence of vertical advection.
    vadv = (ds.shift(z_t=-1).fillna(0) - ds).isel(z_t=slice(0, -1))
    vadv = vadv.sum('z_t').rename('vadv')
    vadv = _convert_units(vadv)
    vadv.attrs['long_name'] = 'vertical advection'
    return vadv.load()


def _compute_vertical_mixing(ds, grid, mask, kmax=None):
    """Compute contribution from vertical mixing."""
    print('Computing vertical mixing...')
    ds = ds[['DIA_IMPVF', 'KPP_SRC']]
    if kmax is not None:
        ds = ds.isel(z_t=slice(0, kmax + 1))
    ds = ds.where(mask)
    ds = ds.groupby('time.year').mean('time').rename({'year': 'time'})
    # Only need to flip sign of DIA_IMPVF.
    ds['DIA_IMPVF'] = _convert_to_tendency(ds['DIA_IMPVF'], grid.area, sign=-1, kmax=kmax)
    ds['KPP_SRC'] = _convert_to_tendency(ds['KPP_SRC'], grid.vol, kmax=kmax)

    # Compute divergence of diapycnal mixing
    diadiff = _compute_vertical_divergence(ds.DIA_IMPVF)

    # Sum KPP and diapycnal mixing to create vmix term.
    vmix = (ds.KPP_SRC + diadiff).rename('vmix').sum('z_t')
    vmix = _convert_units(vmix)
    vmix.attrs['long_name'] = 'vertical mixing'
    return vmix.load()


def _compute_SMS(ds, grid, mask, kmax=None):
    """Compute SMS term from biology."""
    print('Computing source/sink...')
    ds = ds['SMS']
    if kmax is not None:
        ds = ds.isel(z_t=slice(0, kmax + 1))
    ds = ds.where(mask)
    ds = _convert_to_tendency(ds, grid.vol, kmax=kmax).rename('sms').sum('z_t')
    ds = _convert_units(ds)
    # SMS comes as monthly output. Need to resample to annual for comparison to the other tracer budget terms.
    ds = ds.groupby('time.year').mean('time').rename({'year': 'time'})
    ds.attrs['long_name'] = 'source/sink'
    return ds.load()


def _compute_surface_fluxes(ds, grid, mask):
    """Computes surface fluxes of tracer."""
    print('Computing surface fluxes...')
    ds = ds[['FvICE', 'FvPER', 'STF']]
    ds = ds.where(mask)
    ds = _convert_to_tendency(ds, grid.area)
    ds = _convert_units(ds)
    ds = ds.groupby('time.year').mean('time').rename({'year': 'time'})
    vf = (ds.FvICE + ds.FvPER).rename('vf')
    stf = (ds.STF).rename('stf')
    vf.attrs['long_name'] = 'virtual flux'
    stf.attrs['long_name'] = 'surface tracer flux'
    return vf.load(), stf.load()


def _process_input_dataset(ds):
    """Checks that input dataset has appropriate variables, etc."""
    mandatory_vars = ['UE', 'VN', 'WT', 'HDIFE', 'HDIFN', 'HDIFB', 'DIA_IMPVF', 'KPP_SRC']
    if not set(mandatory_vars).issubset(ds):
        missing_vars = [v for v in mandatory_vars if v not in ds]
        error_msg = (
            'Input dataset does not contain the mandatory variables for budget analysis. '
            + f'`ds` is missing {missing_vars}'
        )
        raise IOError(error_msg)

    # Drop coordinates for shift, roll, etc.
    coord_vars = [c for c in ds.coords if c != 'time']
    ds = ds.drop(coord_vars)
    if 'z_w_top' in ds.dims:
        ds = ds.rename({'z_w_top': 'z_t'})
    if 'z_w_bot' in ds.dims:
        ds = ds.rename({'z_w_bot': 'z_t'})

    # Force chunking.
    if not ds.chunks:
        raise IOError('Please input a dataset with chunks.')
    return ds


def regional_tracer_budget(ds, grid, mask=None, mask_int=None, budget_depth=None, sum_area=True):
    """Return a regional tracer budget on the POP grid.

    Parameters
    ----------
    ds : `xarray.Dataset`
      Dataset containing global POP output, with the tracer suffix removed:
      * UE
      * VN
      * WT
      * HDIFE
      * HDIFN
      * HDIFB
      * DIA_IMPVF
      * KPP_SRC
      * FvICE (optional)
      * FvPER (optional)
      * STF (optional; rename from FG_{tracer} for CO2)
      * SMS (optional; rename from J_{tracer})
    grid : str
      POP grid (e.g., POP_gx3v7, POP_gx1v7, POP_tx0.1v3)
    mask : `xarray.DataArray`, optional
      Mask on POP grid with integers for region of interest. If None, use REGION_MASK.
    mask_int : int
      Number corresponding to integer on mask. E.g., 1 for the Southern Ocean for REGION_MASK.
    budget_depth : int, optional
      Depth to compute budget to in m. If None, compute for full depth.
    sum_area : bool, optional
      If True, sum across nlat/nlon to return a budget integrated over the full control volume.

    Returns
    -------
    reg_budget: `xarray.Dataset`
      Dataset containing integrated budget terms over masked POP volume.
    """
    ds = _process_input_dataset(ds)
    grid = _process_grid(grid)
    if mask is None:
        # Default to REGION_MASK from POP.
        mask = grid.mask
    if mask_int is None:
        raise ValueError('Please supply an integer for your mask via the `mask_int` keyword.')
    mask = mask == mask_int
    mask = mask.drop(mask.coords)

    # Compute maximum k-index for budget.
    if budget_depth is not None:
        # Convert from m to cm.
        budget_depth *= 100
        kmax = _compute_kmax(budget_depth, grid.z)
    else:
        kmax = None

    ladv = _compute_lateral_advection(ds, grid, mask, kmax=kmax)
    vadv = _compute_vertical_advection(ds, grid, mask, kmax=kmax)
    lmix = _compute_lateral_mixing(ds, grid, mask, kmax=kmax)
    vmix = _compute_vertical_mixing(ds, grid, mask, kmax=kmax)
    sms = _compute_SMS(ds, grid, mask, kmax=kmax)
    vf, stf = _compute_surface_fluxes(ds, grid, mask)

    # Merge into dataset.
    reg_budget = xr.merge([ladv, vadv, lmix, vmix, sms, vf, stf])
    reg_budget.attrs['units'] = 'mol/yr'
    if sum_area:
        reg_budget = reg_budget.sum(['nlat', 'nlon'])
    return reg_budget
