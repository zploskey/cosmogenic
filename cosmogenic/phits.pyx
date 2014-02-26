"""
Implementation of PHITS, a set of analytical function fit to a Monte Carlo
simulation of the cosmic ray cascade of particles through the atmosphere.

Reference:
    Sato 2008
"""

import numpy as np

u_pmu = np.array([6.26e9, 0.00343, 1.01, 0.00418, 3.75e8])
u_nmu = np.array([5.82e9, 0.00362, 1.02, 0.00451, 3.20e8])

def flux_mu(d, charge='-'):
    """
    Positive muon flux at atmospheric depth d.
    """
    if charge == '-':
    	u = u_nmu
    elif charge == '+': 
	u = u_pmu
    else:
	raise ValueError("Unknown argument '%s'" % charge)
	
    return u[0] * (np.exp(-u[1] * d) - u[2] * np.exp(-u[3] * d)) + u[4]
