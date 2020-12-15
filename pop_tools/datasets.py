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
