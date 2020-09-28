import dask
import numpy as np
import xarray as xr
from numba import vectorize


@vectorize(['float64(float64)', 'float32(float32)'], nopython=True)
def compute_pressure(depth):
    """
    Convert depth in meters to pressure in bars.

    Parameters
    ----------
    depth : float
      Depth in meters

    Returns
    -------
    pressure : float
      Pressure in dbar
    """
    return (
        0.059808 * (np.exp(-0.025 * depth) - 1.0) + 0.100766 * depth + 2.28405e-7 * (depth ** 2.0)
    )


def eos(salt, temp, return_coefs=False, **kwargs):
    """
    Compute density as a function of salinity, temperature, and
    depth (or pressure).

    McDougall, T.J., D.R. Jackett, D.G. Wright, and R. Feistel, 2003:
    Accurate and Computationally Efficient Algorithms for Potential
    Temperature and Density of Seawater. J. Atmos. Oceanic Technol., 20,
    730–741, _`https://doi.org/10.1175/1520-0426(2003)20<730:AACEAF>2.0.CO;2`.

    test value:
        rho = 1033.213387 kg/m^3;
        S = 35.0 PSU, theta = 20.0 C, pressure = 2000.0 dbars

    Parameters
    ----------
    salt : float
        salinity, psu
    temp : float
        potential temperature, degree C
    return_coefs : boolean, optional [default=False]
        Logical, if true function returns 2 additional arguments:
        dRHOdS and dRHOdT
    depth : float, optional
        depth in meters, if not provided, pressure must be provided.
    pressure : float, optional
        depth in dbar

    Returns
    -------
    rho : float
        density kg/m^3
    dRHOdS : float, optional
        Derivative of rho with respect to salinity.
    dRHOdT : float, optional
        Derivative of rho with respect to temperature.
    """

    depth = kwargs.pop('depth', None)
    pressure = kwargs.pop('pressure', None)

    if kwargs:
        raise ValueError(f'unknown arguments: {kwargs}')

    if depth is None and pressure is None:
        raise ValueError('either depth or pressure must be supplied')
    else:
        d_or_p = depth if pressure is None else pressure

    use_xarray = False
    if any(isinstance(arg, xr.DataArray) for arg in [salt, temp, d_or_p]):
        if not all(isinstance(arg, xr.DataArray) for arg in [salt, temp, d_or_p]):
            raise ValueError('cannot operate on mixed types')
        use_xarray = True

    # compute pressure
    if pressure is None:
        if use_xarray:
            pressure = xr.full_like(depth, fill_value=np.nan)
            pressure[:] = 10.0 * compute_pressure(depth.data)  # dbar
        else:
            pressure = 10.0 * compute_pressure(depth)  # dbar

    # enforce min/max values
    tmin = -2.0
    tmax = 999.0
    smin = 0.0
    smax = 999.0

    if use_xarray:
        temp = temp.clip(tmin, tmax)
        salt = salt.clip(smin, smax)

        salt, temp, pressure = xr.broadcast(salt, temp, pressure)
        if isinstance(salt.data, dask.array.Array):
            pressure = pressure.chunk(salt.chunks)

        if return_coefs:
            RHO, dRHOdS, dRHOdT = _compute_eos_coeffs(salt, temp, pressure)

            dRHOdS.name = 'dRHOdS'
            dRHOdS.attrs['units'] = 'kg/m^3/degC'
            dRHOdS.attrs['long_name'] = 'Haline contraction coefficient'

            dRHOdT.name = 'dRHOdT'
            dRHOdT.attrs['units'] = 'kg/m^3/degC'
            dRHOdT.attrs['long_name'] = 'Thermal expansion coefficient'

        else:
            RHO = xr.apply_ufunc(
                _compute_eos,
                salt,
                temp,
                pressure,
                dask='parallelized',
                output_dtypes=[salt.dtype],
            )

        RHO.name = 'density'
        RHO.attrs['units'] = 'kg/m^3'
        RHO.attrs['long_name'] = 'Density'

    else:
        temp = np.clip(temp, tmin, tmax)
        salt = np.clip(salt, smin, smax)

        if return_coefs:
            RHO, dRHOdS, dRHOdT = _compute_eos_coeffs(salt, temp, pressure)
        else:
            RHO = _compute_eos(salt, temp, pressure)

    if return_coefs:
        return RHO, dRHOdS, dRHOdT
    else:
        return RHO


@vectorize(
    ['float64(float64, float64, float64)', 'float32(float32, float32, float32)'],
    nopython=True,
)
def _compute_eos(salt, temp, pressure):
    # MWJF EOS coefficients
    # *** these constants will be used to construct the numerator
    mwjfnp0s0t0 = 9.99843699e2
    mwjfnp0s0t1 = 7.35212840
    mwjfnp0s0t2 = -5.45928211e-2
    mwjfnp0s0t3 = 3.98476704e-4
    mwjfnp0s1t0 = 2.96938239
    mwjfnp0s1t1 = -7.23268813e-3
    mwjfnp0s2t0 = 2.12382341e-3
    mwjfnp1s0t0 = 1.04004591e-2
    mwjfnp1s0t2 = 1.03970529e-7
    mwjfnp1s1t0 = 5.18761880e-6
    mwjfnp2s0t0 = -3.24041825e-8
    mwjfnp2s0t2 = -1.23869360e-11

    # *** these constants will be used to construct the denominator
    mwjfdp0s0t0 = 1.0
    mwjfdp0s0t1 = 7.28606739e-3
    mwjfdp0s0t2 = -4.60835542e-5
    mwjfdp0s0t3 = 3.68390573e-7
    mwjfdp0s0t4 = 1.80809186e-10
    mwjfdp0s1t0 = 2.14691708e-3
    mwjfdp0s1t1 = -9.27062484e-6
    mwjfdp0s1t3 = -1.78343643e-10
    mwjfdp0sqt0 = 4.76534122e-6
    mwjfdp0sqt2 = 1.63410736e-9
    mwjfdp1s0t0 = 5.30848875e-6
    mwjfdp2s0t3 = -3.03175128e-16
    mwjfdp3s0t1 = -1.27934137e-17

    salt2 = salt ** 0.5

    # compute density
    # *** first calculate numerator of MWJF density [P_1(S,T,p)]
    mwjfnums0t0 = mwjfnp0s0t0 + pressure * (mwjfnp1s0t0 + pressure * mwjfnp2s0t0)
    mwjfnums0t1 = mwjfnp0s0t1
    mwjfnums0t2 = mwjfnp0s0t2 + pressure * (mwjfnp1s0t2 + pressure * mwjfnp2s0t2)
    mwjfnums0t3 = mwjfnp0s0t3
    mwjfnums1t0 = mwjfnp0s1t0 + pressure * mwjfnp1s1t0
    mwjfnums1t1 = mwjfnp0s1t1
    mwjfnums2t0 = mwjfnp0s2t0

    WORK1 = (
        mwjfnums0t0
        + temp * (mwjfnums0t1 + temp * (mwjfnums0t2 + mwjfnums0t3 * temp))
        + salt * (mwjfnums1t0 + mwjfnums1t1 * temp + mwjfnums2t0 * salt)
    )

    # *** now calculate denominator of MWJF density [P_2(S,T,p)]
    mwjfdens0t0 = mwjfdp0s0t0 + pressure * mwjfdp1s0t0
    mwjfdens0t1 = mwjfdp0s0t1 + (pressure ** 3) * mwjfdp3s0t1
    mwjfdens0t2 = mwjfdp0s0t2
    mwjfdens0t3 = mwjfdp0s0t3 + (pressure ** 2) * mwjfdp2s0t3
    mwjfdens0t4 = mwjfdp0s0t4
    mwjfdens1t0 = mwjfdp0s1t0
    mwjfdens1t1 = mwjfdp0s1t1
    mwjfdens1t3 = mwjfdp0s1t3
    mwjfdensqt0 = mwjfdp0sqt0
    mwjfdensqt2 = mwjfdp0sqt2

    WORK2 = (
        mwjfdens0t0
        + temp * (mwjfdens0t1 + temp * (mwjfdens0t2 + temp * (mwjfdens0t3 + mwjfdens0t4 * temp)))
        + salt
        * (
            mwjfdens1t0
            + temp * (mwjfdens1t1 + temp * temp * mwjfdens1t3)
            + salt2 * (mwjfdensqt0 + temp * temp * mwjfdensqt2)
        )
    )

    DENOMK = 1.0 / WORK2

    return WORK1 * DENOMK


# @jit(nopython=True)
def _compute_eos_coeffs(salt, temp, pressure):
    # MWJF EOS coefficients
    # *** these constants will be used to construct the numerator
    mwjfnp0s0t0 = 9.99843699e2
    mwjfnp0s0t1 = 7.35212840
    mwjfnp0s0t2 = -5.45928211e-2
    mwjfnp0s0t3 = 3.98476704e-4
    mwjfnp0s1t0 = 2.96938239
    mwjfnp0s1t1 = -7.23268813e-3
    mwjfnp0s2t0 = 2.12382341e-3
    mwjfnp1s0t0 = 1.04004591e-2
    mwjfnp1s0t2 = 1.03970529e-7
    mwjfnp1s1t0 = 5.18761880e-6
    mwjfnp2s0t0 = -3.24041825e-8
    mwjfnp2s0t2 = -1.23869360e-11

    # *** these constants will be used to construct the denominator
    mwjfdp0s0t0 = 1.0
    mwjfdp0s0t1 = 7.28606739e-3
    mwjfdp0s0t2 = -4.60835542e-5
    mwjfdp0s0t3 = 3.68390573e-7
    mwjfdp0s0t4 = 1.80809186e-10
    mwjfdp0s1t0 = 2.14691708e-3
    mwjfdp0s1t1 = -9.27062484e-6
    mwjfdp0s1t3 = -1.78343643e-10
    mwjfdp0sqt0 = 4.76534122e-6
    mwjfdp0sqt2 = 1.63410736e-9
    mwjfdp1s0t0 = 5.30848875e-6
    mwjfdp2s0t3 = -3.03175128e-16
    mwjfdp3s0t1 = -1.27934137e-17

    salt2 = salt ** 0.5

    # compute density
    # *** first calculate numerator of MWJF density [P_1(S,T,p)]
    mwjfnums0t0 = mwjfnp0s0t0 + pressure * (mwjfnp1s0t0 + pressure * mwjfnp2s0t0)
    mwjfnums0t1 = mwjfnp0s0t1
    mwjfnums0t2 = mwjfnp0s0t2 + pressure * (mwjfnp1s0t2 + pressure * mwjfnp2s0t2)
    mwjfnums0t3 = mwjfnp0s0t3
    mwjfnums1t0 = mwjfnp0s1t0 + pressure * mwjfnp1s1t0
    mwjfnums1t1 = mwjfnp0s1t1
    mwjfnums2t0 = mwjfnp0s2t0

    WORK1 = (
        mwjfnums0t0
        + temp * (mwjfnums0t1 + temp * (mwjfnums0t2 + mwjfnums0t3 * temp))
        + salt * (mwjfnums1t0 + mwjfnums1t1 * temp + mwjfnums2t0 * salt)
    )

    # *** now calculate denominator of MWJF density [P_2(S,T,p)]
    mwjfdens0t0 = mwjfdp0s0t0 + pressure * mwjfdp1s0t0
    mwjfdens0t1 = mwjfdp0s0t1 + (pressure ** 3) * mwjfdp3s0t1
    mwjfdens0t2 = mwjfdp0s0t2
    mwjfdens0t3 = mwjfdp0s0t3 + (pressure ** 2) * mwjfdp2s0t3
    mwjfdens0t4 = mwjfdp0s0t4
    mwjfdens1t0 = mwjfdp0s1t0
    mwjfdens1t1 = mwjfdp0s1t1
    mwjfdens1t3 = mwjfdp0s1t3
    mwjfdensqt0 = mwjfdp0sqt0
    mwjfdensqt2 = mwjfdp0sqt2

    WORK2 = (
        mwjfdens0t0
        + temp * (mwjfdens0t1 + temp * (mwjfdens0t2 + temp * (mwjfdens0t3 + mwjfdens0t4 * temp)))
        + salt
        * (
            mwjfdens1t0
            + temp * (mwjfdens1t1 + temp * temp * mwjfdens1t3)
            + salt2 * (mwjfdensqt0 + temp * temp * mwjfdensqt2)
        )
    )

    DENOMK = 1.0 / WORK2

    RHOFULL = WORK1 * DENOMK

    # dRHOdT
    WORK3 = mwjfnums0t1 + temp * (2.0 * mwjfnums0t2 + 3.0 * mwjfnums0t3 * temp) + mwjfnums1t1 * salt

    WORK4 = (
        mwjfdens0t1
        + salt * mwjfdens1t1
        + temp
        * (
            2.0 * (mwjfdens0t2 + salt * salt2 * mwjfdensqt2)
            + temp * (3.0 * (mwjfdens0t3 + salt * mwjfdens1t3) + temp * 4.0 * mwjfdens0t4)
        )
    )

    DRHODT = (WORK3 - WORK1 * DENOMK * WORK4) * DENOMK

    # dRHOdS
    WORK3 = mwjfnums1t0 + mwjfnums1t1 * temp + 2.0 * mwjfnums2t0 * salt

    WORK4 = (
        mwjfdens1t0
        + temp * (mwjfdens1t1 + temp * temp * mwjfdens1t3)
        + 1.5 * salt2 * (mwjfdensqt0 + temp * temp * mwjfdensqt2)
    )

    DRHODS = (WORK3 - WORK1 * DENOMK * WORK4) * DENOMK * 1000.0

    return RHOFULL, DRHODS, DRHODT
