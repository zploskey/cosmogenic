"""
Functions to calculation production rates
"""

from __future__ import division, print_function, unicode_literals

import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import dfitpack

from cosmogenic import muon
from cosmogenic import scaling as scal
from cosmogenic import util

LAMBDA_h = 160.0  # attenuation length of hadronic component in atm, g / cm2

DEFAULT_ALT = 0.0  # sea level
DEFAULT_LAT = 75.0  # high latitude


def P_sp(z, n, scaling=None, alt=None,
         lat=None, t=None, s=None, pressure=None):
    """
    Returns production rate due to spallation reactions (atoms/g/yr)

    ::math P_{sp} = f_{scaling} * P_0 * exp(-z / \Lambda)

    where f_scaling is a scaling factor. It currently scales for altitude
    and latitude after Stone (2000).

    Parameters
    ----------
    z : array_like
       depth in g/cm**2 (vector or scalar)
    n : nuclide object from cosmogenic.nuclide
        The nuclide to you want a production rate for. Can be a user-defined
        object as long as it satisfies the same interfaces as the objects
        in cosmogenic.nuclide.
    scaling : string, optional
              If "stone" applies Stone 2000 scaling factor
    alt : array_like
          site altitude in meters
    lat : array_like,
          site latitude (in degrees North)

    Returns
    -------
    p_sp : array_like
           production rate from spallation in atoms/g/year
    """

    if scaling is None and alt is None and lat is None and pressure is None:
        f_scaling = 1.0
        # we are not scaling, default to the LSD (Sato "Sa") prod. rate
        scaling = "Sa"
    elif scaling in ("stone", "St", None):
        # scale with Stone by default for now until LSD scaling is done
        f_scaling = scal.stone2000_sp(lat=lat, alt=alt, pressure=pressure)
        scaling = "St"
    elif scaling == "Sa":
        raise NotImplementedError("No Sato scaling implemented yet.")

    p_sp_ref = n.scaling_p_sp_ref[scaling]
    return f_scaling * p_sp_ref * np.exp(-z / LAMBDA_h)


def P_tot(z, n, scaling=None, alt=None, lat=None, s=None, pressure=None):
    """
    Total production rate of nuclide n in atoms / g of material / year

    Parameters
    ----------
    z : array_like
       depth in g/cm**2 (vector or scalar)
    alt : array_like
          site altitude in meters
    lat : array_like,
          site latitude (in degrees North)
    n : nuclide object from cosmogenic.nuclide
        The nuclide to you want a production rate for. Can be a user-defined
        object as long as it satisfies the same interfaces as the objects
        in cosmogenic.nuclide.
    scaling : string, optional
              If "stone" applies Stone 2000 scaling factor to spallation
              production rate.

    Returns
    -------
    p : array_like
           total production rate in atoms/g/year
    """
    production_rate = P_sp(z=z, n=n, scaling=scaling, alt=alt, lat=lat,
                           pressure=pressure)
    production_rate += muon.P_mu_total(z=z, n=n, h=alt)

    return production_rate


def interpolate_P_tot(
        max_depth, npts, n=None, scaling=None, alt=None, lat=None):
    """
    Interpolates the production rate function using a spline interpolation.
    Evaluated points are log base 2 spaced, with more points concentrated near
    the surface.

    Parameters
    ----------
    max_depth : float
                maximum depth to interpolate to in g/cm**2
    npts : int
           number of points to use in interpolation
    n : nuclide object
    scaling : string, optional
              If set to "stone", applies Stone 2000 scaling scheme to
              the spallation production rate.
    alt : float
          site altitude in meters
    lat : float
          site latitude (degrees)

    """
    zs = np.unique(np.logspace(0, np.log2(max_depth + 1), npts, base=2)) - 1
    prod_rates = P_tot(zs, n=n, scaling=scaling, alt=alt, lat=lat)
    p = ProductionSpline(zs, prod_rates)
    return p, zs, prod_rates


class ProductionSpline(InterpolatedUnivariateSpline):

    """
    One-dimensional interpolating spline for a given set of data points.

    Fits a spline y=s(x) of degree `k` to the provided `x`, `y` data. Spline
    function passes through all provided points. Equivalent to
    `UnivariateSpline` with s=0.

    Parameters
    ----------
    x : (N,) array_like
        production rate data points
    y : (N,) array_like
        depths in g/cm**2

    w : (N,) array_like, optional
    Weights for spline fitting. Must be positive. If None (default),
    weights are all equal.
    bbox : (2,) array_like, optional
    2-sequence specifying the boundary of the approximation interval. If
    None (default), bbox=[x[0],x[-1]].
    k : int, optional
    Degree of the smoothing spline. Must be 1 <= `k` <= 5.

    See Also
    --------
    InterpolatedUnivariateSpline : Superclass in NumPy
    Notes
    -----
    The number of data points must be larger than the spline degree `k`.
    """

    def __init__(self, x=None, y=None, w=None, bbox=[None] * 2, k=3, s=0,
                 filename=None):
        """

        Parameters
        ----------
        x,y : 1-d sequences of data points (z must be in strictly ascending
              order). x is production rate, y is corresponding depth

        Optional input:
          w          - positive 1-d sequence of weights
          bbox       - 2-sequence specifying the boundary of
                       the approximation interval.
                       By default, bbox=[p[0],p[-1]]
          k          - degree of the univariate spline (defaults to 3)
          s          - positive smoothing factor defined for
                       estimation condition:
                         sum((w[i]*(z[i]-s(p[i])))**2,axis=0) <= s
                       Default s=len(w) which should be a good value
                       if 1/w[i] is an estimate of the standard
                       deviation of y[i].
          filename   - file to load a saved spline from
        """

        if (x is None) or (y is None) and (filename is not None):
            self._data = util.unpickle(filename)
        else:
            self._data = dfitpack.fpcurf0(x, y, k, w=w,
                                          xb=bbox[0], xe=bbox[1], s=s)

        self._reset_class()

    def __call__(self, x, nu=0):
        res = super(ProductionSpline, self).__call__(x, nu)
        res_arr = np.atleast_1d(res)
        res_arr[res_arr < 0.0] = 0.0
        return res_arr[0] if res_arr.size == 1 else res_arr

    def save(self, filename):
        """
        What gets saved is pretty specific to the version of scipy that we are use
        so is important to test that this works with new versions of scipy. For
        now, the only important values for the spline interpolation are stored in
        the spline's _data member, but could change, as could the structure of
        _eval_args. To be safe, do not depend on this to load old interpolations.
        It is probably safest to create a new interpolation for a new scipy
        version.
        """
        util.pickle(self._data, filename)
