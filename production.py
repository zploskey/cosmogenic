#!/usr/bin/python

"""
Functions to calculation production rates
"""

import numpy as np

import muon
import scaling

LAMBDA_h = 155.0 # attenuation length of hadronic component in atm, g / cm2

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
