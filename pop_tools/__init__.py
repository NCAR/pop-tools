"""Top-level module for pop_tools"""

from importlib.metadata import version as _version

from .calc import cfc11sol, cfc12sol
from .datasets import DATASETS
from .eos import compute_pressure, eos
from .fill import lateral_fill, lateral_fill_np_array
from .grid import get_grid, grid_defs
from .region_masks import list_region_masks, region_mask_3d
from .xgcm_util import to_xgcm_grid_dataset

try:
    __version__ = _version("pop_tools")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "9999"
