import os

import numpy as np
import pytest
import xarray as xr


def ds_compare(ds1, ds2, assertion='allequal', rtol=1e-5, atol=1e-8):
    """Compare two datasets."""
    assert assertion in ['allequal', 'allclose'], f'unknown assertion: {assertion}'

    compare_results = {}
    equal = []
    close = []
    for v in ds1.variables:
        if v not in ds2.variables:
            print(f'missing {v} in (2)')
        else:
            try:
                xr.testing.assert_identical(ds1[v], ds2[v])
                compare_results[v] = 'identical'
                equal.append(True)
                close.append(True)
            except BaseException:
                try:
                    xr.testing.assert_allclose(ds1[v], ds2[v], rtol=rtol, atol=atol)
                    compare_results[v] = 'close'
                    equal.append(False)
                    close.append(True)
                except BaseException:
                    compare_results[v] = 'different'
                    equal.append(False)
                    close.append(False)

    print(f'All equal: {all(equal)}')
    if not all(equal):
        print(f'All close: {all(close)}')
        for v, result in compare_results.items():
            print(f'{v}: {result}')

    if assertion == 'allequal':
        return all(equal)
    elif assertion == 'allclose':
        return all(close)
