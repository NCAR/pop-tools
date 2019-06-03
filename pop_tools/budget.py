import datetime
import glob

import numpy as np
import xarray as xr

from .grid import get_grid

NMOLS_TO_MOLYR = 1e-9 * 86400.0 * 365.0


def _process_grid(grid):
    """Load in POP grid and process coordinates for budget calculation"""
    grid_f = get_grid(grid)

    # Extract needed coordinates.
    dz = grid_f.dz.drop('z_t')
    area = (grid_f.TAREA).rename('area')
    vol = (area * dz).rename('vol')
    z = (grid_f.z_w_bot.drop('z_w_bot')).rename('z')
    mask = grid_f.REGION_MASK.rename('mask')
    return xr.merge([dz, area, vol, z, mask])


def _process_coords(ds, concat_dim='time', drop=True, extra_coord_vars=['time_bound']):
    """Preprocessor function to drop all non-dim coords, which slows down concatenation

    Borrowed from @sridge in an issue thread
    """

    time_new = ds.time - datetime.timedelta(seconds=1)
    ds['time'] = time_new  # shift time back 1s so that the months are correct
    # ex: 12/31 is misrepresented as 1/01 without this correction

    coord_vars = [v for v in ds.data_vars if concat_dim not in ds[v].dims]
    for ecv in extra_coord_vars:
        if ecv in ds:
            coord_vars += extra_coord_vars

    if drop:
        return ds.drop(coord_vars)
    else:
        return ds.set_coords(coord_vars)


def _load_tracer_terms(basepath, filebase, tracer, var_list=None, drop_time=True):
    """Loads in the tracer terms for current budget calculation"""
    model_vars = {
        'UE': f'UE_{tracer}',
        'VN': f'VN_{tracer}',
        'WT': f'WT_{tracer}',
        'HDIFE': f'HDIFE_{tracer}',
        'HDIFN': f'HDIFN_{tracer}',
        'HDIFB': f'HDIFB_{tracer}',
        'DIA_IMPVF': f'DIA_IMPVF_{tracer}',
        'KPP_SRC': f'KPP_SRC_{tracer}',
        'FvICE': f'FvICE_{tracer}',
        'FvPER': f'FvPER_{tracer}',
        # hard-coded for DIC for now
        'STF': f'FG_CO2',
        'SMS': f'J_{tracer}',
    }

    loadVars = dict((k, model_vars[k]) for k in var_list)
    ds = xr.Dataset()
    for new_var, raw_var in loadVars.items():
        # Currently assumes that the directory is full of a single
        # simulation output.
        ds_i = xr.open_mfdataset(f'{basepath}/{filebase}*{raw_var}.*', preprocess=_process_coords)
        ds_i = ds_i.rename({raw_var: new_var})
        # Resample the monthly output to annual.
        ds = ds.merge(ds_i)
    # Drop coordinates, since they get in the way of roll, shift, diff.
    if drop_time:
        dropCoords = ds.coords
    else:
        dropCoords = [c for c in ds.coords if c != 'time']
    ds = ds.drop(dropCoords)
    # Rename all to z_t. Indices are handled with POP grid.
    if 'z_w_top' in ds.dims:
        ds = ds.rename({'z_w_top': 'z_t'})
    if 'z_w_bot' in ds.dims:
        ds = ds.rename({'z_w_bot': 'z_t'})
    return ds


def _compute_kmax(budget_depth, z):
    """Compute the k-index of the maximum budget depth"""
    kmax = (z <= budget_depth).argmin() - 1
    kmax = kmax.values
    # This only occurs when the full depth is requested (or budget_depth
    # is greater than the maximum model depth)
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
    """Converts from nmol/s to mol/yr and adds attribute label."""
    ds *= NMOLS_TO_MOLYR
    ds.attrs['units'] = 'mol/yr'
    return ds


def _compute_lateral_advection(basepath, filebase, tracer, grid, mask, kmax=None):
    """Compute lateral advection (UE and VN)"""
    print('Computing lateral advection...')
    ds = _load_tracer_terms(basepath, filebase, tracer, var_list=['UE', 'VN'])

    if kmax is not None:
        ds = ds.isel(z_t=slice(0, kmax + 1))

    # Convert to tendency.
    ds = _convert_to_tendency(ds, grid.vol, kmax=kmax)

    # Compute advection.
    mask_shift = mask.roll(nlon=-1, roll_coords=True)
    ladv_zonal = ds.UE.where(mask_shift).roll(nlon=1, roll_coords=True) - ds.UE.where(mask)
    mask_shift = mask.roll(nlat=-1, roll_coords=True)
    ladv_merid = ds.VN.where(mask_shift).roll(nlat=1, roll_coords=True) - ds.VN.where(mask)

    # Sum zonal and meridional components..
    ladv = (ladv_zonal + ladv_merid).rename('ladv')

    # Sum over control volume.
    ladv = ladv.sum(['z_t', 'nlat', 'nlon'])
    ladv = _convert_units(ladv)
    ladv.attrs['long_name'] = 'lateral advection'
    return ladv.load()


def _compute_vertical_advection(basepath, filebase, tracer, grid, mask, kmax=None):
    """Compute vertical advection (WT)"""
    print('Computing vertical advection...')
    ds = _load_tracer_terms(basepath, filebase, tracer, var_list=['WT'])

    if kmax is not None:
        # Need one level below max depth to compute divergence into bottom layer.
        ds = ds.isel(z_t=slice(0, kmax + 2))
    ds = ds.where(mask)

    # Convert to tendency.
    if kmax is not None:
        ds = _convert_to_tendency(ds, grid.vol, kmax=kmax + 1)
    else:
        ds = _convert_to_tendency(ds, grid.vol)

    # Compute divergence of vertical advection.
    vadv = (ds.WT.shift(z_t=-1).fillna(0) - ds.WT).isel(z_t=slice(0, -1))
    vadv = vadv.sum(['z_t', 'nlat', 'nlon']).rename('vadv')
    vadv = _convert_units(vadv)
    vadv.attrs['long_name'] = 'vertical advection'
    return vadv.load()


def _compute_lateral_mixing(basepath, filebase, tracer, grid, mask, kmax=None):
    """Compute lateral mixing component."""
    print('Computing lateral mixing...')
    ds = _load_tracer_terms(basepath, filebase, tracer, var_list=['HDIFN', 'HDIFE', 'HDIFB'])

    if kmax is not None:
        ds = ds.isel(z_t=slice(0, kmax + 1))

    # Flip sign so that positive direction is upwards.
    ds = _convert_to_tendency(ds, grid.vol, sign=-1, kmax=kmax)

    # Compute divergence for HDIFN/HDIFE.
    mask_shift = mask.roll(nlon=-1, roll_coords=True)
    lmix_E = ds.HDIFE.where(mask_shift).roll(nlon=1, roll_coords=True) - ds.HDIFE.where(mask)
    mask_shift = mask.roll(nlat=-1, roll_coords=True)
    lmix_N = ds.HDIFN.where(mask_shift).roll(nlat=1, roll_coords=True) - ds.HDIFN.where(mask)

    # Compute divergence for HDIFB
    lmix_B = ds.HDIFB.where(mask)
    # Place "cap" of zeros above top layer to diff upwards.
    ny, nx = lmix_B.nlat.size, lmix_B.nlon.size
    zero_cap = xr.DataArray(np.zeros((ny, nx)), dims=['nlat', 'nlon'])
    lmix_B = xr.concat([zero_cap, lmix_B], dim='z_t')
    lmix_B = lmix_B.diff('z_t')

    # Sum zonal and meridional components.
    lmix = (lmix_N + lmix_E).rename('lmix')
    lmix = lmix.sum(['z_t', 'nlat', 'nlon'])
    lmix = _convert_units(lmix)
    lmix.attrs['long_name'] = 'lateral mixing'

    # Bottom lateral mixing
    # NOTE: Should this combine with the above terms?
    lmix_B = lmix_B.rename('lmix_B')
    lmix_B = lmix_B.sum(['z_t', 'nlat', 'nlon'])
    lmix_B = _convert_units(lmix_B)
    lmix_B.attrs['long_name'] = 'lateral mixing (vertical component)'
    return lmix.load(), lmix_B.load()


def _compute_vertical_mixing(basepath, filebase, tracer, grid, mask, kmax=None):
    """Compute contribution from vertical mixing."""
    print('Computing vertical mixing...')
    ds = _load_tracer_terms(basepath, filebase, tracer, var_list=['DIA_IMPVF', 'KPP_SRC'])

    if kmax is not None:
        ds = ds.isel(z_t=slice(0, kmax + 1))

    # Apply mask.
    ds = ds.where(mask)

    # Only need to flip sign of DIA_IMPVF.
    ds['DIA_IMPVF'] = _convert_to_tendency(ds['DIA_IMPVF'], grid.area, sign=-1, kmax=kmax)
    ds['KPP_SRC'] = _convert_to_tendency(ds['KPP_SRC'], grid.vol, kmax=kmax)

    # Compute divergence of diapycnal mixing.
    diadiff = ds.DIA_IMPVF
    # Place cap of zeros above top layer to diff upwards.
    ny, nx = diadiff.nlat.size, diadiff.nlon.size
    zero_cap = xr.DataArray(np.zeros((ny, nx)), dims=['nlat', 'nlon'])
    diadiff = xr.concat([zero_cap, diadiff], dim='z_t')
    diadiff = diadiff.diff('z_t')

    # Sum KPP and diapycnal mixing to create vmix term.
    vmix = (ds.KPP_SRC + diadiff).rename('vmix')
    vmix = vmix.sum(['z_t', 'nlat', 'nlon'])
    vmix = _convert_units(vmix)
    vmix.attrs['long_name'] = 'vertical mixing'
    return vmix.load()


def _compute_SMS(basepath, filebase, tracer, grid, mask, kmax=None):
    """Compute SMS term from biology."""
    print('Computing source/sink...')
    ds = _load_tracer_terms(basepath, filebase, tracer, var_list=['SMS'], drop_time=False)
    if kmax is not None:
        ds = ds.isel(z_t=slice(0, kmax + 1))
    ds = ds.where(mask)
    ds = _convert_to_tendency(ds, grid.vol, kmax=kmax).rename({'SMS': 'sms'})
    sms = ds.sms.sum(['z_t', 'nlat', 'nlon'])
    sms = _convert_units(sms)
    sms = sms.groupby('time.year').mean('time').rename({'year': 'time'})
    sms.attrs['long_name'] = 'source/sink'
    return sms.load()


def _compute_surface_fluxes(basepath, filebase, tracer, grid, mask):
    """Computes surface fluxes of tracer."""
    ds = _load_tracer_terms(
        basepath, filebase, tracer, var_list=['FvICE', 'FvPER', 'STF'], drop_time=False
    )
    ds = ds.where(mask)
    ds = _convert_to_tendency(ds, grid.area)
    vf = (ds.FvICE + ds.FvPER).rename('vf').sum(['nlat', 'nlon'])
    stf = (ds.STF).rename('stf').sum(['nlat', 'nlon'])
    vf = _convert_units(vf)
    stf = _convert_units(stf)
    vf = vf.groupby('time.year').mean('time').rename({'year': 'time'})
    stf = stf.groupby('time.year').mean('time').rename({'year': 'time'})
    vf.attrs['long_name'] = 'virtual flux'
    stf.attrs['long_name'] = 'surface tracer flux'
    return vf.load(), stf.load()


def regional_tracer_budget(
    basepath, filebase, tracer, grid, mask=None, mask_int=None, budget_depth=None
):
    """Return a regional tracer budget on the POP grid.

    Parameters
    ----------
    basepath : str
      Path to folder with raw POP output
    filebase : str
      Base name of file (e.g., 'g.DPLE.GECOIAF.T62_g16.009.chey.pop.h.')
    tracer : str
      Tracer variable name (e.g., 'DIC')
    grid : str
      POP grid (e.g., POP_gx3v7, POP_gx1v7, POP_tx0.1v3)
    mask : `xarray.DataArray`, optional
      Mask on POP grid with integers for region of interest. If None, use REGION_MASK.
    mask_int : int
      Number corresponding to integer on mask. E.g., 1 for the Southern Ocean for REGION_MASK.
    budget_depth : int, optional
      Depth to compute budget to in m. If None, compute for full depth.

    Returns
    -------
    reg_budget: `xarray.Dataset`
      Dataset containing integrated budget terms over masked POP volume.
    """

    grid = _process_grid(grid)
    if mask is None:
        # Default to REGION_MASK from POP.
        mask = grid.mask
    if mask_int is None:
        raise ValueError('Please supply an integer for your mask via the `mask_int` keyword.')
    mask = mask == mask_int
    mask = mask.drop(mask.coords)

    # Compute k-index for budget depth.
    if budget_depth is not None:
        # Convert from m to cm.
        budget_depth *= 100
        kmax = _compute_kmax(budget_depth, grid.z)
    else:
        kmax = None

    ladv = _compute_lateral_advection(basepath, filebase, tracer, grid, mask, kmax=kmax)
    vadv = _compute_vertical_advection(basepath, filebase, tracer, grid, mask, kmax=kmax)
    lmix, lmix_B = _compute_lateral_mixing(basepath, filebase, tracer, grid, mask, kmax=kmax)
    vmix = _compute_vertical_mixing(basepath, filebase, tracer, grid, mask, kmax=kmax)
    sms = _compute_SMS(basepath, filebase, tracer, grid, mask, kmax=kmax)
    vf, stf = _compute_surface_fluxes(basepath, filebase, tracer, grid, mask)

    # Merge into dataset.
    reg_budget = xr.merge([ladv, vadv, lmix, lmix_B, vmix, sms, vf, stf])
    reg_budget.attrs['units'] = 'mol/yr'
    return reg_budget
