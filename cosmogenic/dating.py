"""
Functions for calculating exposure ages.

We generally follow the notation of Tibor Dunai (2010), Cosmogenic Nuclides:
Principles, Concepts and Application in the Earth Surface Sciences, Cambridge
University Press, Cambridge. pp. 187.
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import scipy.optimize as opt

from cosmogenic import production, sim


def exposure_age(C, P, nuclide, delC=None, delP=None, z0=0.0,
                 erosion_rate=None, z=None, thickness=None):
    """ Calculate an exposure age.

    Parameters:
    ----------
    C:  concentration of cosmogenic nuclide (CN) (atoms / g)
    P:  function
        local production rate as a function of depth in g/cm**2
        units: (atoms / g / yr)
    nuclide: a cosmogenic nuclide object
    delC: float
          absolute uncertainty in the concentration C (1-sigma)
    delP: float
          absolute uncertainty in production rate P (1-sigma)
          
    z0: modern depth (g / cm**2)
    erosion_rate (optional): assumed constant rate of erosion (g / cm**2 / yr),
                             default is no erosion. Ignored if z is supplied.
    z (optional): Function z(t) that returns depth in cm
    """

    # construct z(t) function if one was not supplied
    if z is None:
        if erosion_rate is None:
            z = lambda t: z0
        else:
            def z(t):
                return z0 + erosion_rate * t

    def residual(t):
        C_model, _ = sim.nexpose(n, z, t, p=P, thickness=thickness)
        return np.abs(C - C_model)

    # Go out to 7 half lives + 30%... This should definitely
    # be beyond saturation.
    upper_age_bound = 1.3 * (7 * np.log(2) / nuclide.LAMBDA)
    bounds = (0.0, upper_age_bound)

    res = opt.minimize_scalar(residual, bounds=bounds)
    t = res.x

    if delC is None:
        return t

    # TODO: calculate uncertainty in the exposure age
    delt = age_uncertainty(t, delC, nuclide, delP)
    return t, delt


def age_uncertainty(nuclide, t, delC, delP=None):
    """
    Calculate uncertainty in exposure age 
    """
    if delP is None:
        delP = 0.0

    raise NotImplementedError 
