#!/usr/bin/python
"""
Scaling functions and associated helper functions
"""

#import tempfile

import numpy as np
from scipy.interpolate import UnivariateSpline

ELEVATION_LIMIT = 44330.76923076923 


def alt_to_p(z):
    """
    Convert elevation z in meters to pressure in hPa.
    Valid at midlatitudes and areas where pressure variation is not
    anomalous.

    Stone (2000) J. Geophys. Res., 105(B10), 23,753-23,759. Eq. 1
    """
    z = np.atleast_1d(z)

    if (z > ELEVATION_LIMIT).any():
        raise ValueError('Elevation too high for this approximation') 

    Ps = 1013.25 # sea level pressure (hPa)
    Ts = 288.15 # sea level temperature (K)
    dTdz = 0.0065 # adiabatic lapse rate (K/m)
    gMoverR = 0.03417 # combined constant gM/R (K/m)
    
    return Ps * np.exp((-gMoverR / dTdz) * (np.log(Ts) - np.log(Ts - dTdz * z)))


def stone2000_sp(lat, alt=None, pressure=None, interp='spline'):
    """
    Inputs:
    lat: sample latitude(s) in degrees, scalar
    alt: altitude of the sample (m), scalar
    P:   pressure(s) in hPa, scalar
    
    If both pressure and altitude are supplied we use pressure. If neither is
    supplied we default to sea level pressure.
    """
    P = pressure
    if P == None:
        if alt == None:
            P = 1013.25
        else:
            P = alt_to_p(alt)
    
    a = np.array([31.8518,    34.3699,    40.3153,    42.0983,    56.7733,    69.0720,    71.8733])
    b = np.array([250.3193,   258.4759,   308.9894,   512.6857,   649.1343,   832.4566,   863.1927])
    c = np.array([-0.083393,  -0.089807,  -0.106248,  -0.120551,  -0.160859,  -0.199252,  -0.207069])
    d = np.array([7.4260e-5,  7.9457e-5,  9.4508e-5,  1.1752e-4,  1.5463e-4,  1.9391e-4,  2.0127e-4])
    e = np.array([-2.2397e-8, -2.3697e-8, -2.8234e-8, -3.8809e-8, -5.0330e-8, -6.3653e-8, -6.6043e-8])

    # create index latitudes, 0 thru 90 at 10 degree intervals
    lat_interval = 10 # deg
    ilats = np.arange(0, 90 + lat_interval, lat_interval, dtype=int)
    
    # make sure we're dealing with positive numbers 
    lat = abs(lat)
    
    # number of reference latitudes
    n_ref = a.size
    n_lats = 10
    # calculate scaling factors for index latitudes at sea level
    S_lambda_idx = np.zeros(n_lats)
    S_lambda_idx[0:n_ref] = (a + b * np.exp(-P / 150.0) + c * P + d
						     * P**2 + e * P**3)
    S_lambda_idx[n_ref:n_lats] = S_lambda_idx[n_ref-1]

    # interpolate between the index latitude scaling factors
    # Here we set s=0 so that the spline interpolates exactly through all the
    # data points.
    S_lambda = np.zeros(n_lats)
    if interp == 'spline':
        S_lambda = UnivariateSpline(ilats, S_lambda_idx, s=0)
    elif interp == 'linear':
        S_lambda = UnivariateSpline(ilats, S_lambda_idx, k=1)
    else:
        raise Exception('Unknown Interpolation method')

    F_lambda = S_lambda(lat)
    
    return F_lambda

"""
Elsasser et al. 1956 via Dunai's book:
M = dipole moment [m2 / A]
mu0 = pi*4e-7 [N / A2]
c = 299,792,458 [m/s]
RE = radius of the earth: 
lambda = geomagnetic latitude [radians presumably]

cutoff rigidity Rc [V] = M * mu0 * c * cos^4(lambda) / (14 * pi * RE^2)

m N / s A3
"""

def stone2000Rcsp(h, Rc):
    """
    Cutoff-rigidity based scaling scheme based on Lal's spallation polynomials.
    
    h  = scalar atmospheric pressure (hPa)
    Rc = list of cutoff rigidities (GV)
    
    """
    
    raise NotImplementedError
