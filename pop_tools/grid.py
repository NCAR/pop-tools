import numpy as np
import xarray as xr
from numba import jit, prange

from .config import grid_defs

deg2rad = np.pi / 180.0


def get_grid(grid_name, scrip=False):
    """Return a xarray.Dataset() with POP grid variables.

    Parameters
    ----------

    grid_name : str
      Name of grid (i.e., POP_gx3v7, POP_gx1v7, POP_tx0.1v3)

    scrip : boolean, optional
      Return grid in SCRIP format

    Returns
    -------

    dso : `xarray.Dataset`
      Dataset containing POP grid variables.
    """

    if grid_name not in grid_defs:
        raise ValueError(f'Unknown grid: {grid_name}')

    grid_attrs = grid_defs[grid_name]

    nlat = grid_attrs['lateral_dims'][0]
    nlon = grid_attrs['lateral_dims'][1]
    # dims_xy = (nlat, nlon)

    # read horizontal grid
    grid_file_data = np.fromfile(grid_attrs['horiz_grid_fname'], dtype='>f8', count=-1)
    grid_file_data = grid_file_data.reshape((7, nlat, nlon))

    ULAT = grid_file_data[0, :, :].astype(np.float)
    ULONG = grid_file_data[1, :, :].astype(np.float)
    HTN = grid_file_data[2, :, :].astype(np.float)
    HTE = grid_file_data[3, :, :].astype(np.float)

    # compute TLAT, TLONG
    TLAT = np.empty((nlat, nlon), dtype=np.float)
    TLONG = np.empty((nlat, nlon), dtype=np.float)
    _compute_TLAT_TLONG(ULAT, ULONG, TLAT, TLONG, nlat, nlon)

    # generate DXT, DYT, TAREA
    DXT = np.empty((nlat, nlon))
    DXT[1:nlat, :] = 0.5 * (HTN[0 : nlat - 1, :] + HTN[1:nlat, :])
    DXT[0, :] = 0.5 * (2 * HTN[0, :] - HTN[1, :] + HTN[0, :])

    DYT = np.empty((nlat, nlon))
    DYT[:, 1:nlon] = 0.5 * (HTE[:, 0 : nlon - 1] + HTE[:, 1:nlon])
    DYT[:, 0] = 0.5 * (HTE[:, nlon - 1] + HTE[:, 0])

    TAREA = DXT * DYT

    TLONG = np.where(TLONG < 0.0, TLONG + 2 * np.pi, TLONG)

    # vertical grid
    tmp = np.loadtxt(grid_attrs['vert_grid_file'])
    dz = tmp[:, 0]
    depth_edges = np.concatenate(([0.0], np.cumsum(dz)))
    z_w = depth_edges[0:-1]
    z_w_bot = depth_edges[1:]
    z_t = depth_edges[0:-1] + 0.5 * dz

    # read KMT
    kmt_flat = np.fromfile(grid_attrs['topography_fname'], dtype='>i4', count=-1)
    assert kmt_flat.shape[0] == (
        nlat * nlon
    ), f'unexpected dims in topography file: {grid_attrs["topography_fname"]}'
    assert kmt_flat.max() <= len(z_t), 'Max KMT > length z_t'
    KMT = kmt_flat.reshape(grid_attrs['lateral_dims']).astype(np.int32)

    # output dataset
    dso = xr.Dataset()
    dso['TLAT'] = xr.DataArray(
        TLAT / deg2rad,
        dims=('nlat', 'nlon'),
        attrs={'units': 'degrees_north', 'long_name': 'T-grid latitude'},
    )

    dso['TLONG'] = xr.DataArray(
        TLONG / deg2rad,
        dims=('nlat', 'nlon'),
        attrs={'units': 'degrees_east', 'long_name': 'T-grid longitude'},
    )

    dso['ULAT'] = xr.DataArray(
        ULAT / deg2rad,
        dims=('nlat', 'nlon'),
        attrs={'units': 'degrees_north', 'long_name': 'U-grid latitude'},
    )

    dso['ULONG'] = xr.DataArray(
        ULONG / deg2rad,
        dims=('nlat', 'nlon'),
        attrs={'units': 'degrees_east', 'long_name': 'U-grid longitude'},
    )

    dso['DXT'] = xr.DataArray(
        DXT,
        dims=('nlat', 'nlon'),
        attrs={
            'units': 'cm',
            'long_name': 'x-spacing centered at T points',
            'coordinates': 'TLONG TLAT',
        },
    )

    dso['DYT'] = xr.DataArray(
        DYT,
        dims=('nlat', 'nlon'),
        attrs={
            'units': 'cm',
            'long_name': 'y-spacing centered at T points',
            'coordinates': 'TLONG TLAT',
        },
    )

    dso['TAREA'] = xr.DataArray(
        TAREA / deg2rad,
        dims=('nlat', 'nlon'),
        attrs={'units': 'cm^2', 'long_name': 'area of T cells', 'coordinates': 'TLONG TLAT'},
    )

    dso['KMT'] = xr.DataArray(
        KMT,
        dims=('nlat', 'nlon'),
        attrs={'long_name': 'k Index of Deepest Grid Cell on T Grid', 'coordinates': 'TLONG TLAT'},
    )

    dso['z_t'] = xr.DataArray(
        z_t,
        dims=('z_t'),
        name='z_t',
        attrs={
            'units': 'cm',
            'long_name': 'depth from surface to midpoint of layer',
            'positive': 'down',
        },
    )

    dso['dz'] = xr.DataArray(
        dz,
        dims=('z_t'),
        coords={'z_t': dso.z_t},
        attrs={'units': 'cm', 'long_name': 'thickness of layer k'},
    )

    dso['z_w'] = xr.DataArray(
        z_w,
        dims=('z_w'),
        attrs={
            'units': 'cm',
            'positive': 'down',
            'long_name': 'depth from surface to top of layer',
        },
    )

    dso['z_w_bot'] = xr.DataArray(
        z_w_bot,
        dims=('z_w_bot'),
        attrs={
            'units': 'cm',
            'positive': 'down',
            'long_name': 'depth from surface to bottom of layer',
        },
    )

    dso.attrs = grid_attrs
    if scrip:
        raise NotImplementedError('SCRIP format not implemented')

    return dso


@jit(nopython=True, parallel=True)
def _compute_TLAT_TLONG(ULAT, ULONG, TLAT, TLONG, nlat, nlon):
    """Compute TLAT and TLONG from ULAT, ULONG"""

    for j in prange(1, nlat):
        jm1 = j - 1
        for i in prange(0, nlon):
            im1 = np.mod(i - 1 + nlon, nlon)

            tmp = np.cos(ULAT[jm1, im1])
            xsw = np.cos(ULONG[jm1, im1]) * tmp
            ysw = np.sin(ULONG[jm1, im1]) * tmp
            zsw = np.sin(ULAT[jm1, im1])

            tmp = np.cos(ULAT[jm1, i])
            xse = np.cos(ULONG[jm1, i]) * tmp
            yse = np.sin(ULONG[jm1, i]) * tmp
            zse = np.sin(ULAT[jm1, i])

            tmp = np.cos(ULAT[j, im1])
            xnw = np.cos(ULONG[j, im1]) * tmp
            ynw = np.sin(ULONG[j, im1]) * tmp
            znw = np.sin(ULAT[j, im1])

            tmp = np.cos(ULAT[j, i])
            xne = np.cos(ULONG[j, i]) * tmp
            yne = np.sin(ULONG[j, i]) * tmp
            zne = np.sin(ULAT[j, i])

            xc = 0.25 * (xsw + xse + xnw + xne)
            yc = 0.25 * (ysw + yse + ynw + yne)
            zc = 0.25 * (zsw + zse + znw + zne)

            r = np.sqrt(xc * xc + yc * yc + zc * zc)

            TLAT[j, i] = np.arcsin(zc / r)
            TLONG[j, i] = np.arctan2(yc, xc)

    # generate bottom row
    TLAT[0, :] = TLAT[1, :] - (TLAT[2, :] - TLAT[1, :])
    TLONG[0, :] = TLONG[1, :] - (TLONG[2, :] - TLONG[1, :])
