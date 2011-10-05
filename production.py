#!/usr/bin/python

"""
Functions to calculation production rates
"""

import numpy as np

from scipy.interpolate import UnivariateSpline

import muon
import scaling

LAMBDA_h = 155.0 # attenuation length of hadronic component in atm, g / cm2
LAMBDA_fast = 4320.0 # after Heisinger 2002

def P_sp(z, alt, lat, n):
    """
    Production rate due to spallation reactions (atoms/g/yr)

    ::math P_{sp} = f_{scaling} * P_0 * exp(-z / \Lambda)

    where f_scaling is a scaling factor. It currently scales for altitude
    and latitude after Stone (2000).
    """
    f_scaling = scaling.stone2000_sp(lat, alt)
    return f_scaling * n.P0 * np.exp(-z / LAMBDA_h)

def P_tot(z, alt, lat, n):
    """
    Total production rate of nuclide n in atoms / g of material / year
    """
    return P_sp(z, alt, lat, n) + muon.P_mu_total(z, alt, n)

def interpolate_P_tot(max_depth, npts, alt, lat, n):
    zs = np.unique(np.logspace(0, np.log2(max_depth + 1), npts, base=2)) - 1
    prod_rates = P_tot(zs, alt, lat, n)
    p = UnivariateSpline(zs, prod_rates, k=3, s=0)
    return p
