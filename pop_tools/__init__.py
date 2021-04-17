"""Top-level module for pop_tools"""


from pkg_resources import DistributionNotFound, get_distribution

from .datasets import DATASETS
from .eos import compute_pressure, eos
from .calc import cfc12sol, cfc11sol
from .fill import lateral_fill, lateral_fill_np_array
from .grid import get_grid, grid_defs
from .region_masks import list_region_masks, region_mask_3d
from .xgcm_util import to_xgcm_grid_dataset

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
