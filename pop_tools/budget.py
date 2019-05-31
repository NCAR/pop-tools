import datetime
import glob

import xarray as xr

from .grid import get_grid

MOLS_TO_MOLYR = 1e-9 * 86400.0 * 365.0


def _process_coords(ds, concat_dim='time', drop=True, extra_coord_vars=['time_bound']):
    """Preprocessor function to drop all non-dim coords, which slows down concatenation.

    Borrowed from @sridge in an issue thread.
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


def _load_tracer_terms(basepath, filebase, tracer, var_list=None):
    """Loads in the output necessary to compute tracer budget."""
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
    ds = ds.drop(ds.coords)
    # Rename all to z_t. Indices are handled with POP grid.
    try:
        ds = ds.rename({'z_w_top': 'z_t'})
    except ValueError:
        pass
    try:
        ds = ds.rename({'z_w_bot': 'z_t'})
    except ValueError:
        pass
    return ds


def _compute_kmax(budget_depth, z):
    """Compute the k-index of the maximum budget depth."""
    kmax = (z <= budget_depth).argmin() - 1
    kmax = kmax.values
    kmax_model = z.size

    # This only occurs when the full depth is requested (or budget_depth
    # is greater than the maximum model depth)
    if kmax == -1:
        kmax = kmax_model
        full_depth = True
    else:
        full_depth = False
    return kmax, full_depth


def _volume_area_normalize(ds, tracer, vol, area):
    """Converts from volume- and area-dependent units to just tendencies."""
    volNormalize = ['UE', 'VN', 'WT', 'KPP_SRC', 'SMS']
    volNormalizeSignFlip = ['HDIFE', 'HDIFN', 'HDIFB', 'DIA_IMPVF']
    areaNormalize = ['FvICE', 'FvPER', 'STF']
    volNorm = ds[volNormalize] * vol
    volNormFlip = ds[volNormalizeSignFlip] * -vol
    areaNorm = ds[areaNormalize] * area
    ds = xr.merge([volNorm, volNormFlip, areaNorm])
    ds.attrs['units'] = 'nmol/s'
    return ds


def regional_tracer_budget(basepath, filebase, tracer, budget_depth=None):
    """Compute regional tracer budget."""
    ds = _load_tracer_terms(basepath, filebase, tracer)

    # Currently hard-coded.
    grid_f = get_grid('POP_gx1v7')

    # Extract needed coordinates.
    dz = grid_f.dz.drop('z_t')
    area = grid_f.TAREA
    vol = area * dz
    z = grid_f.z_w_bot.drop('z_w_bot')

    # Compute k-index for budget depth.
    if budget_depth is not None:
        # Convert from m to cm.
        budget_depth *= 100
        kmax, full_depth = _compute_kmax(budget_depth, z)
    else:
        kmax = z.size
        full_depth = True
    # NOTE: remove. just using to avoid flake8 issues.
    kmax = kmax
    full_depth = full_depth

    # Convert raw units to tendency terms.
    ds = _volume_area_normalize(ds, tracer, vol, area)
    return ds
