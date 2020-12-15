import os
from pathlib import Path

import numpy as np
import pkg_resources
import pooch
import xarray as xr
import yaml
from numba import jit, prange

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# On Cheyenne/Casper and/or CGD machines, use local inputdata directory
# See: https://github.com/NCAR/pop-tools/issues/24#issue-523701065
# The name of the environment variable that can overwrite the path argument
cesm_data_root_path = os.environ.get('CESMDATAROOT')

if cesm_data_root_path is not None and os.path.exists(cesm_data_root_path):
    INPUTDATA_DIR = cesm_data_root_path
else:
    # This is still the default in case the environment variable isn't defined
    INPUTDATA_DIR = ['~', '.pop_tools']


INPUTDATA = pooch.create(
    path=INPUTDATA_DIR,
    version_dev='master',
    base_url='https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/',
)


INPUTDATA.load_registry(pkg_resources.resource_stream('pop_tools', 'inputdata_registry.txt'))

if tqdm is not None:
    downloader = pooch.HTTPDownloader(progressbar=True, verify=False, allow_redirects=True)
else:
    downloader = pooch.HTTPDownloader(verify=False, allow_redirects=True)


grid_def_file = pkg_resources.resource_filename('pop_tools', 'pop_grid_definitions.yaml')
input_templates_dir = pkg_resources.resource_filename('pop_tools', 'input_templates')

with open(grid_def_file) as f:
    grid_defs = yaml.safe_load(f)


def fetch(self, fname, processor=None, downloader=None):

    """
    This is a modified version of Pooch.fetch() method. This modification is necessary
    due to the fact that on Cheyenne/Casper path to the local data storage folder points
    to a folder (CESMDATAROOT: /glade/p/cesmdata/cseg), and this is not a location that
    we have permissions to write to.

    Parameters
    ----------
    fname : str
        The file name (relative to the *base_url* of the remote data
        storage) to fetch from the local storage.
    processor : None or callable
        If not None, then a function (or callable object) that will be
        called before returning the full path and after the file has been
        downloaded (if required).
    downloader : None or callable
        If not None, then a function (or callable object) that will be
        called to download a given URL to a provided local file name. By
        default, downloads are done through HTTP without authentication
        using :class:`pooch.HTTPDownloader`.
    Returns
    -------
    full_path : str
        The absolute path (including the file name) of the file in the
        local storage.

    """

    self._assert_file_in_registry(fname)
    url = self.get_url(fname)
    full_path = self.abspath / fname
    known_hash = self.registry[fname]
    abspath = str(self.abspath)
    action, verb = pooch.core.download_action(full_path, known_hash)

    if action in ('download', 'update'):
        pooch.utils.get_logger().info("%s file '%s' from '%s' to '%s'.", verb, fname, url, abspath)
        if downloader is None:
            downloader = pooch.downloaders.choose_downloader(url)

        pooch.core.stream_download(url, full_path, known_hash, downloader, pooch=self)

    if processor is not None:
        return processor(str(full_path), action, self)

    return str(full_path)


# Override fetch method at instance level
# Reference: https://stackoverflow.com/a/46757134/7137180
# Replace fetch() with modified fetch() for this object only
INPUTDATA.fetch = fetch.__get__(INPUTDATA, pooch.Pooch)


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
        raise ValueError(
            f"""Unknown grid: {grid_name}
             Please select from the following: {list(grid_defs.keys())}"""
        )

    grid_attrs = grid_defs[grid_name]

    nlat = grid_attrs['lateral_dims'][0]
    nlon = grid_attrs['lateral_dims'][1]

    # read horizontal grid
    horiz_grid_fname = INPUTDATA.fetch(grid_attrs['horiz_grid_fname'], downloader=downloader)
    grid_file_data = np.fromfile(horiz_grid_fname, dtype='>f8', count=-1)
    grid_file_data = grid_file_data.reshape((7, nlat, nlon))

    ULAT = grid_file_data[0, :, :].astype(np.float)
    ULONG = grid_file_data[1, :, :].astype(np.float)
    HTN = grid_file_data[2, :, :].astype(np.float)
    HTE = grid_file_data[3, :, :].astype(np.float)

    # compute TLAT, TLONG
    TLAT = np.empty((nlat, nlon), dtype=np.float)
    TLONG = np.empty((nlat, nlon), dtype=np.float)
    _compute_TLAT_TLONG(ULAT, ULONG, TLAT, TLONG, nlat, nlon)

    # generate DXT, DYT
    # DXT[i,j] = (HTN[i,j] + HTN[i,j−1])/2
    DXT = np.empty((nlat, nlon))
    DXT[1:, :] = 0.5 * (HTN[: nlat - 1, :] + HTN[1:, :])
    # DXT[0, :] = 0.5 * (2 * HTN[0, :] - HTN[1, :] + HTN[0, :])
    DXT[0, :] = 0.5 * (HTN[0, :] + HTN[nlat - 1, :])

    # DYT[i,j] = (HTE[i,j] + HTE[i−1,j])/2
    DYT = np.empty((nlat, nlon))
    DYT[:, 1:] = 0.5 * (HTE[:, : nlon - 1] + HTE[:, 1:])
    DYT[:, 0] = 0.5 * (HTE[:, nlon - 1] + HTE[:, 0])

    # generate DXU, DYU
    # DXU[i,j] = (HTN[i,j] + HTN[i+1,j])/2
    DXU = np.empty((nlat, nlon))
    DXU[:, : nlon - 1] = 0.5 * (HTN[:, : nlon - 1] + HTN[:, 1:])
    DXU[:, nlon - 1] = 0.5 * (HTN[:, nlon - 1] + HTN[:, 0])

    # DYU[i,j] = (HTE[i,j] + HTE[i,j+1])/2
    DYU = np.empty((nlat, nlon))
    DYU[: nlat - 1, :] = 0.5 * (HTE[: nlat - 1, :] + HTE[1:, :])
    DYU[nlat - 1, :] = 0.5 * (HTE[nlat - 1, :] + HTE[0, :])

    # compute TAREA, UAREA
    TAREA = DXT * DYT
    UAREA = DXU * DYU

    # vertical grid
    vert_grid_fname = os.path.join(input_templates_dir, grid_attrs['vert_grid_file'])
    tmp = np.loadtxt(vert_grid_fname)
    dz = tmp[:, 0]
    depth_edges = np.concatenate(([0.0], np.cumsum(dz)))
    z_w = depth_edges[0:-1]
    z_w_bot = depth_edges[1:]
    z_t = depth_edges[0:-1] + 0.5 * dz

    # read KMT
    topography_fname = INPUTDATA.fetch(grid_attrs['topography_fname'], downloader=downloader)
    kmt_flat = np.fromfile(topography_fname, dtype='>i4', count=-1)
    assert kmt_flat.shape[0] == (
        nlat * nlon
    ), f'unexpected dims in topography file: {grid_attrs["topography_fname"]}'
    assert kmt_flat.max() <= len(z_t), 'Max KMT > length z_t'
    KMT = kmt_flat.reshape(grid_attrs['lateral_dims']).astype(np.int32)

    # read REGION_MASK
    region_mask_fname = INPUTDATA.fetch(grid_attrs['region_mask_fname'], downloader=downloader)
    region_mask_flat = np.fromfile(region_mask_fname, dtype='>i4', count=-1)
    assert region_mask_flat.shape[0] == (
        nlat * nlon
    ), f'unexpected dims in region_mask file: {grid_attrs["region_mask_fname"]}'
    REGION_MASK = region_mask_flat.reshape(grid_attrs['lateral_dims']).astype(np.int32)

    # output dataset
    dso = xr.Dataset()
    if scrip:
        corner_lat, corner_lon = _compute_corners(ULAT, ULONG)

        dso['grid_dims'] = xr.DataArray(np.array([nlon, nlat], dtype=np.int32), dims=('grid_rank',))
        dso.grid_dims.encoding = {'dtype': np.int32, '_FillValue': None}

        dso['grid_center_lat'] = xr.DataArray(
            np.rad2deg(TLAT.reshape((-1,))), dims=('grid_size'), attrs={'units': 'degrees'}
        )
        dso.grid_center_lat.encoding = {'dtype': np.float64, '_FillValue': None}

        dso['grid_center_lon'] = xr.DataArray(
            np.rad2deg(TLONG.reshape((-1,))), dims=('grid_size'), attrs={'units': 'degrees'}
        )
        dso.grid_center_lon.encoding = {'dtype': np.float64, '_FillValue': None}

        dso['grid_corner_lat'] = xr.DataArray(
            np.rad2deg(corner_lat.reshape((-1, 4))),
            dims=('grid_size', 'grid_corners'),
            attrs={'units': 'degrees'},
        )
        dso.grid_corner_lat.encoding = {'dtype': np.float64, '_FillValue': None}

        dso['grid_corner_lon'] = xr.DataArray(
            np.rad2deg(corner_lon.reshape((-1, 4))),
            dims=('grid_size', 'grid_corners'),
            attrs={'units': 'degrees'},
        )
        dso.grid_corner_lon.encoding = {'dtype': np.float64, '_FillValue': None}

        dso['grid_imask'] = xr.DataArray(
            np.where(KMT > 0, 1, 0).reshape((-1,)), dims=('grid_size'), attrs={'units': 'unitless'}
        )
        dso.grid_imask.encoding = {'dtype': np.int32, '_FillValue': None}

        grid_attrs.update({'conventions': 'SCRIP'})

    else:
        TLONG = np.where(TLONG < 0.0, TLONG + 2 * np.pi, TLONG)

        dso['TLAT'] = xr.DataArray(
            np.rad2deg(TLAT),
            dims=('nlat', 'nlon'),
            attrs={'units': 'degrees_north', 'long_name': 'T-grid latitude'},
        )

        dso['TLONG'] = xr.DataArray(
            np.rad2deg(TLONG),
            dims=('nlat', 'nlon'),
            attrs={'units': 'degrees_east', 'long_name': 'T-grid longitude'},
        )

        dso['ULAT'] = xr.DataArray(
            np.rad2deg(ULAT),
            dims=('nlat', 'nlon'),
            attrs={'units': 'degrees_north', 'long_name': 'U-grid latitude'},
        )

        dso['ULONG'] = xr.DataArray(
            np.rad2deg(ULONG),
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

        dso['DXU'] = xr.DataArray(
            DXU,
            dims=('nlat', 'nlon'),
            attrs={
                'units': 'cm',
                'long_name': 'x-spacing centered at U points',
                'coordinates': 'ULONG ULAT',
            },
        )

        dso['DYU'] = xr.DataArray(
            DYU,
            dims=('nlat', 'nlon'),
            attrs={
                'units': 'cm',
                'long_name': 'y-spacing centered at U points',
                'coordinates': 'ULONG ULAT',
            },
        )

        dso['TAREA'] = xr.DataArray(
            TAREA,
            dims=('nlat', 'nlon'),
            attrs={'units': 'cm^2', 'long_name': 'area of T cells', 'coordinates': 'TLONG TLAT'},
        )

        dso['UAREA'] = xr.DataArray(
            UAREA,
            dims=('nlat', 'nlon'),
            attrs={'units': 'cm^2', 'long_name': 'area of U cells', 'coordinates': 'ULONG ULAT'},
        )

        dso['KMT'] = xr.DataArray(
            KMT,
            dims=('nlat', 'nlon'),
            attrs={
                'long_name': 'k Index of Deepest Grid Cell on T Grid',
                'coordinates': 'TLONG TLAT',
            },
        )

        dso['REGION_MASK'] = xr.DataArray(
            REGION_MASK,
            dims=('nlat', 'nlon'),
            attrs={
                'long_name': 'basin index number (signed integers)',
                'coordinates': 'TLONG TLAT',
            },
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

    grid_attrs.update({'title': f'{grid_name} grid'})

    dso.attrs = grid_attrs

    # Remove region_mask_regions
    if 'region_mask_regions' in dso.attrs:
        regions = dso.attrs.pop('region_mask_regions')
        region_names, region_vals = list(zip(*regions.items()))
        region_coord = list(range(len(regions)))
        dso['region_name'] = xr.DataArray(list(region_names), coords=[region_coord], dims=['nreg'])
        dso['region_val'] = xr.DataArray(list(region_vals), coords=[region_coord], dims=['nreg'])
        dso['region_val'].attrs['coordinate'] = 'region_name'

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


def _compute_corners(ULAT, ULONG):
    """Compute grid corners."""

    nlat, nlon = ULAT.shape
    corner_lat = np.empty((nlat, nlon, 4), dtype=np.float)
    corner_lon = np.empty((nlat, nlon, 4), dtype=np.float)

    # NE corner
    corner_lat[:, :, 0] = ULAT
    corner_lon[:, :, 0] = ULONG

    # NW corner (copy from NE corner of column to the left, assume zonal periodic bc)
    corner_lat[:, :, 1] = np.roll(corner_lat[:, :, 0], 1, axis=1)
    corner_lon[:, :, 1] = np.roll(corner_lon[:, :, 0], 1, axis=1)

    # SW corner (copy from NW corner of row below, bottom row is extrapolated from 2 rows above)
    corner_lat[1:nlat, :, 2] = corner_lat[0 : nlat - 1, :, 1]
    corner_lon[1:nlat, :, 2] = corner_lon[0 : nlat - 1, :, 1]
    corner_lat[0, :, 2] = corner_lat[1, :, 2] - (corner_lat[2, :, 2] - corner_lat[1, :, 2])
    corner_lon[0, :, 2] = corner_lon[1, :, 2] - (corner_lon[2, :, 2] - corner_lon[1, :, 2])

    # SE corner (copy from NE corner of row below, bottom row is extrapolated from 2 rows above)
    corner_lat[1:nlat, :, 3] = corner_lat[0 : nlat - 1, :, 0]
    corner_lon[1:nlat, :, 3] = corner_lon[0 : nlat - 1, :, 0]
    corner_lat[0, :, 3] = corner_lat[1, :, 3] - (corner_lat[2, :, 3] - corner_lat[1, :, 3])
    corner_lon[0, :, 3] = corner_lon[1, :, 3] - (corner_lon[2, :, 3] - corner_lon[1, :, 3])

    return corner_lat, corner_lon
