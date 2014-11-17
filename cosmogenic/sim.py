"""
Simulate geomorphic scenarios along with CN production.
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import scipy.integrate

from cosmogenic import production


def nexpose(n, z, ti, tf=0, p=None, tol=1e-4, thickness=None):
    """
    Calculate concentrations for an arbitrary depth history z(t).

    :math:`\\int^t_0 P(z(t)) \\exp(-\\lambda t) dt`

    Parameters
    ----------
    n : cosmogenic.nuclide object
    z : function or callable
        z(t), Depth in g/cm**2 as a function of time t in years. Time decreases
        until the present.
    ti : float
        initial exposure age (years ago)
    tf : float
        time when exposure stopped (years ago)
    p : function or callable
        P(z), production rate of nuclide in atoms/g/year as function of depth
        in g/cm**2.
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
    if p is None:
        p = n.production_rate

    # define the integrand: instantaneous production and decay
    def P(t):
        return p(z(t)) * np.exp(-n.LAMBDA * t)

    if thickness is None:
        res = scipy.integrate.quad(P, tf, ti, epsrel=tol)
    else:
        bottom_z = lambda t: z(t) + thickness
        p2d = lambda z, t: p(z) * np.exp(-n.LAMBDA * t) / thickness
        res = scipy.integrate.dblquad(p2d, tf, ti, z, bottom_z, epsrel=tol)

    C = res[0]
    err = res[1]
    return C, err


def multiglaciate(dz, t_gl, t_intergl, t_postgl, z, n, p=None, n_gl=None,
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
    dz = np.atleast_1d(dz)
    t_gl = np.atleast_1d(t_gl)
    t_intergl = np.atleast_1d(t_intergl)
    t_postgl = np.atleast_1d(t_postgl)

    if p is None:
        p = n.production_rate

    if n_gl is None:
        n_gl = dz.size
    
    ones = np.ones(n_gl)
    dz = dz * ones if dz.size is not n_gl else dz
    t_gl = t_gl * ones if t_gl.size is not n_gl else t_gl
    t_intergl = (t_intergl * ones if t_intergl.size is not n_gl
                                  else intergl)
    assert dz.size == t_gl.size == t_intergl.size

    # add the atoms created as we go back in time
    # recent interglacial first
    conc = expose(n, z + postgl_shielding, t_postgl, p=p)
    z_cur = z.copy()  # start at current depths
    t_begint = t_postgl  # the time when the current interglacial began
    t_endint = 0.0  # time (now) when current interglacial ended
    for i in range(n_gl):
        z_cur += dz[i]  # go back to depth and time before glacial erosion
        t_endint = t_begint + t_gl[i]
        t_begint = t_endint + t_intergl[i]
        conc += expose(n, z_cur, t_begint, t_endint, p)
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


def expose(n, z, ti, tf=0, p=None):
    """
    Expose samples a depths z (g/cm**2) from time ti until time tf
    (both in years) at production rate p(z). Adjust their concentrations for
    radioactive decay since tf. Return the concentration of nuclide n.
    """
    if p is None:
        p = n.production_rate
    
    # See note in simple_expose for why we must assign this temporary.
    pofz = p(z)
    L = n.LAMBDA
    conc = (pofz / L) * (np.exp(-L * tf) - np.exp(-L * ti))
    return conc


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
