"""
Functions to load sample data
"""

import os
from pathlib import Path

import pkg_resources
import pooch

DATASETS = pooch.create(
    path=['~', '.pop_tools', 'data'],
    version_dev='master',
    base_url='ftp://ftp.cgd.ucar.edu/archive/aletheia-data/cesm-data/ocn/',
    env='POP_TOOLS_DATA_DIR',
)
DATASETS.load_registry(pkg_resources.resource_stream('pop_tools', 'data_registry.txt'))


def fetch(pooch_instance, fname, processor=None, downloader=None):

    """
    This is a modified version of Pooch.fetch() method. This modification is necessary
    due to the fact that on Cheyenne/Casper path to the local data storage folder points
    to a folder (CESMDATAROOT: /glade/p/cesmdata/cseg), and this is not a location that
    we have permissions to write to.

    Parameters
    ----------
    pooch_instance : pooch.Pooch
        A Pooch instance to use.
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

    pooch_instance._assert_file_in_registry(fname)
    url = pooch_instance.get_url(fname)
    full_path = pooch_instance.abspath / fname
    known_hash = pooch_instance.registry[fname]
    abspath = str(pooch_instance.abspath)
    action, verb = pooch.core.download_action(full_path, known_hash)

    if action in ('download', 'update'):
        pooch.utils.get_logger().info("%s file '%s' from '%s' to '%s'.", verb, fname, url, abspath)
        if downloader is None:
            downloader = pooch.downloaders.choose_downloader(url)

        pooch.core.stream_download(url, full_path, known_hash, downloader, pooch=pooch_instance)

    if processor is not None:
        return processor(str(full_path), action, pooch_instance)

    return str(full_path)


class UnzipZarr(pooch.processors.Unzip):
    """
    Processor that unpacks a zarr store zip archive and
    returns the zarr store path.
    """

    def __call__(self, fname, action, pooch):
        """
        Extract all files from the given archive.
        Parameters
        ----------
        fname : str
            Full path of the zipped file in local storage.
        action : str
            Indicates what action was taken by :meth:`pooch.Pooch.fetch`:
            * ``"download"``: File didn't exist locally and was downloaded
            * ``"update"``: Local file was outdated and was re-download
            * ``"fetch"``: File exists and is updated so it wasn't downloaded
        pooch : :class:`pooch.Pooch`
            The instance of :class:`pooch.Pooch` that is calling this.
        Returns
        -------
        zarr_store : str
            A full path to a zarr store in the extracted archive.
        """
        extract_dir = fname + self.suffix
        if action in ('update', 'download') or not os.path.exists(extract_dir):
            # Make sure that the folder with the extracted files exists
            if not os.path.exists(extract_dir):
                os.makedirs(extract_dir)
            self._extract_file(fname, extract_dir)
        # Get a list of all file names (including subdirectories) in our folder
        # of unzipped files.
        fnames = [
            os.path.join(path, fname) for path, _, files in os.walk(extract_dir) for fname in files
        ]
        # Return the path of the zarr store
        return Path(sorted(fnames)[0]).parent.as_posix()
