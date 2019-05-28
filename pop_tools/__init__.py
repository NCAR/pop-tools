"""Top-level module for pop_tools"""


from pkg_resources import DistributionNotFound, get_distribution

from .config import grid_defs
from .eos import compute_pressure, eos
from .fill import lateral_fill, lateral_fill_np_array
from .grid import get_grid

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
