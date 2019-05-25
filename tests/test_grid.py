
import xarray as xr
import pop_tools


def test_template():
    print(pop_tools.grid_defs)


def test_get_grid():
    for grid in pop_tools.grid_defs.keys():
        print('-' * 80)
        print(grid)
        ds = pop_tools.get_grid(grid)
        ds.info()
        assert isinstance(ds, xr.Dataset)
        print()
