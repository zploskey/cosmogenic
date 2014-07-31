"""
Simulate geomorphic scenarios along with CN production.
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import scipy.integrate

from cosmogenic import production


def nexpose(P, nuclide, z, ti, tf=0, tol=1e-4, thickness=None):
    """
    Calculate concentrations for an arbitrary depth history z(t).

    :math:`\\int^t_0 P(z(t)) \\exp(-\\lambda t) dt`

    Parameters
    ----------
    P : function or callable
        P(z), production rate of nuclide in atoms/g/year as function of depth
        in g/cm**2.
    nuclide : cosmogenic.nuclide object
    z : function or callable
        z(t), Depth in g/cm**2 as a function of time t in years. Time decreases
        until the present.
    ti : float
        initial exposure age (years ago)
    tf : float
        time when exposure stopped (years ago)
    tol : float
          error tolerance for the integration
    thickness : float (optional)
        sample thickness in g/cm**2

    Returns
    -------
    (C, err) : tuple
               C is the concentration in atoms/g
               err is an estimate of the absolute error in C [atoms/g]

    """

    # define the integrand: instantaneous production and decay
    def p(t):
        return P(z(t)) * np.exp(-nuclide.LAMBDA * t)

    if thickness is None:
        res = scipy.integrate.quad(p, tf, ti, epsrel=tol)
    else:
        bottom_z = lambda t: z(t) + thickness
        p2d = lambda z, t: P(z) * np.exp(-nuclide.LAMBDA * t) / thickness
        res = scipy.integrate.dblquad(p2d, tf, ti, z, bottom_z, epsrel=tol)

    C = res[0]
    err = res[1]
    return C, err


def multiglaciate(dz, t_gl, t_intergl, t_postgl, z, n, p, n_gl=None,
                  postgl_shielding=0):
    """Find the resulting concentration profile for a glacial history and site.

    This function predicts the concentration profile for a glacial history. The
    glacial history of the site is described in such a way that the parameters
    are easy to vary for the Monte Carlo simulation--i.e. the times of
    glacial and interglacial periods are in lengths rather than absolute ages.
    Depths of the sample and the depths eroded during each glaciation are both
    in units of g/cm**2, avoiding tying our results to a rock density.

    Parameters
    ----------
    dz : vector of the depths eroded during each glaciation (g/cm2)
    t_gl : array_like or scalar
           array of lengths of time spent ice covered in each glaciation (yr)
    t_intergl : array_like or scalar
                vector, length of exposure periods (yr)
    t_postgl : float
               time the sample has been exposed since deglaciation (yr)
    z : array_like or scalar
        array of samples depths beneath the modern surface (g/cm**2)
    n : nuclide object
    p : function or callable
        production rate function p(z), should return a production rate in
        atoms/g/year at depth z (in g/cm*2).
    n_gl : int, optional
           If supplied, this is the number of glaciations to simulate
           assuming that t_gl and t_intergl are scalars, not vectors.
    """

    z = np.atleast_1d(z)

    if n_gl is not None:
        if n_gl > 1:
            ones = np.ones(n_gl)
            dz *= ones
            t_gl *= ones
            t_intergl *= ones
    else:
        n_gl = dz.size
        assert dz.size == t_gl.size == t_intergl.size

    # add the atoms created as we go back in time
    # recent interglacial first
    conc = simple_expose(z + postgl_shielding, t_postgl, n, p)
    z_cur = z.copy()  # start at current depths
    t_begint = t_postgl  # the time when the current interglacial began
    t_endint = 0.0  # time (now) when current interglacial ended
    for i in range(n_gl):
        z_cur += dz[i]  # go back to depth and time before glacial erosion
        t_endint = t_begint + t_gl[i]
        t_begint = t_endint + t_intergl[i]
        conc += expose(z_cur, t_begint, t_endint, n, p)
    return conc


def glacial_depth_v_time(gl, intergl, postgl, dz, n_gl=None):
    """ Returns a tuple of times and depths of a surface sample.

    Parameters
    ----------
    gl : array_like
         vector of lengths of each glaciation (yr)
    intergl: vector of lengths of interglacial periods (yr)
    postgl: time since last deglaciation (yr)
    dz: vector of glacial erosion depths during each glaciation

    Returns
    -------

    """
    gl = np.atleast_1d(gl)
    intergl = np.atleast_1d(intergl)
    dz = np.atleast_1d(dz)

    if n_gl is None:
        n_gl = max(gl.size, intergl.size, dz.size)

    # pad them all out to be the right size
    gl = gl * np.ones(n_gl)
    intergl = intergl * np.ones(n_gl)
    dz = dz * np.ones(n_gl)

    # interleave the two arrays
    tmp = np.column_stack((gl, intergl)).reshape(1, gl.size * 2).flatten()
    t = np.add.accumulate(np.concatenate(([0, postgl], tmp)))
    tmp = np.column_stack((dz, np.zeros(dz.size))).reshape(
        1, dz.size * 2).flatten()
    z = np.add.accumulate(np.concatenate(([0, 0], tmp)))
    return (t, z)


def expose(z, t_init, t_final, n, p):
    """
    Expose samples a depths z (g/cm**2) from time t_init until time t_final
    (both in years) at production rate p(z). Adjust their concentrations for
    radioactive decay since t_final. Return the concentration of nuclide n.
    """
    # See note in simple_expose for why we must assign this temporary.
    pofz = p(z)
    L = n.LAMBDA
    conc = (pofz / L) * (np.exp(-L * t_final) - np.exp(-L * t_init))
    return conc


def expose_from_site_data(z, t_init, t_final, n, h, lat):
    p = production.P_tot(z, h, lat, n)
    return (expose(z, t_init, t_final, n, p), p)


def simple_expose_slow(z, t_exp, n, h, lat):
    # calculate the production rate
    p = production.P_tot(z, h, lat, n)
    return (p / n.LAMBDA) * (1 - np.exp(-n.LAMBDA * t_exp))


def simple_expose(z, t_exp, n, p):
    """
    Expose samples at depths z (g/cm**2) for t_exp years, recording nuclide n
    with production rate p (atoms/g/yr).
    """

    # Note:
    # We must calculate the production rate at depths z first.
    # Otherwise, there are bizarre bugs when using the p = production.P_tot
    # The test case for this function fails if we don't assign a temporary.
    pofz = p(z)
    N = (pofz / n.LAMBDA) * (1 - np.exp(-n.LAMBDA * t_exp))
    return N


def fwd_profile(z0, z_removed, t, n, p):
    """
    Calculates the nuclide concentration profile resulting from repeated
    glaciation of a bedrock surface.

    In all parameters that reference time, time is zero starting at modern day
    and increases into the past.

    z0: modern depths at which we want predicted concentrations (g/cm2)
    z_removed: list of depths of rock removed in successive glaciations (g/cm2)
    t: ages of switching between glacial/interglacial (array of times in years)
    exposed to cosmic rays in the recent past (in years). The first element of
    this array should be the exposure time since deglaciation, increasing after.
    n: nuclide object
    p: production rate function of depth in g/cm2
    """
    L = n.LAMBDA  # decay constant
    N = np.zeros(z0.size)  # nuclide concentration
    t_beg = t[2::2]
    t_end = t[1::2]

    # Add nuclides formed postglacially
    N += simple_expose(z0, t[0], n, p)

    z_cur = z0.copy()
    for i, dz in enumerate(z_removed):
        z_cur += dz
        N += (p(z_cur) / L) * (np.exp(-L * t_end[i]) - np.exp(-L * t_beg[i]))

    return N


def fwd_profile_slow(z0, z_removed, t, n, h, lat):
    """
    Calculates the nuclide concentration profile resulting from repeated
    glaciation of a bedrock surface.

    In all parameters that reference time, time is zero starting at modern day
    and increases into the past.

    z0: modern depths at which we want predicted concentrations (g/cm2)
    z_removed: list of depths of rock removed in successive glaciations (g/cm2)
    t: ages of switching between glacial/interglacial (array of times in years)
    exposed to cosmic rays in the recent past (in years). The first element of
    this array should be the exposure time since deglaciation, increasing after.
    n: the nuclide being produced (nuclide object)
    h: elevation of the site (m)
    lat: latitude of the site (degrees)
    """
    L = n.LAMBDA
    N = np.zeros(z0.size)
    t_beg = t[2::2]
    t_end = t[1::2]

    # Add nuclides formed postglacially
    N += simple_expose_slow(z0, t[0], n, h, lat)

    z_cur = z0.copy()
    for i, dz in enumerate(z_removed):
        z_cur += dz
        p = production.P_tot(z_cur, h, lat, n)
        N += (p / L) * (np.exp(-L * t_end[i]) - np.exp(-L * t_beg[i]))

    return N


def rand_erosion_hist(avg, sigma, n):
    """
    Generates a sequence of n numbers randomly distributed about avg and
    standard deviation sigma.
    """
    return np.random.normal(avg, sigma, n)


def steady_erosion(P, z0, eros, nuc, T, T_stop=0):

    def int_eq(t):
        return P(z(t)) * np.exp(-nuc.LAMBDA * t)

    z0 = np.atleast_1d(z0)
    N = np.zeros_like(z0)
    for i, depth in enumerate(z0):
        z = lambda t: eros * t + depth
        res, _ = scipy.integrate.quad(int_eq, T_stop, T)
        N[i] = res

    return N
