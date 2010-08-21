#!/usr/bin/python

import numpy as np
import scipy as sp
from numpy import exp, log, arange, array, abs
import np_util as util
import scipy.interpolate

@util.autovec
def alt_to_p(z):
    """
    Convert elevation z to pressure in hPa.
    Stone (2000) J. Geophys. Res., 105(B10), 23,753-23,759. Eq. 1
    """
    Ps = 1013.25 # sea level pressure (hPa)
    Ts = 288.15 # sea level temperature (K)
    dTdz = 0.0065 # adiabatic lapse rate (K/m)
    gMR = 0.03417 # combined constant gM/R (K/m)
    
    return Ps * exp((-gMR / dTdz) * (np.log(Ts) - np.log(Ts - dTdz * z)))

def stone2000(lat, P=None, Fsp=0.978, alt=None):
    """
    Inputs:
    lat: sample latitude(s) in degrees, as a scalar or an array
    P:   pressure(s) in hPa, scalar or array
    Fsp: fraction of spallation reaction, defaults to 0.978
    alt: altitude of the sample (m)
    
    If both pressure and altitude are supplied we use pressure. If neither is
    supplied we default to sea level pressure.
    """
    if P == None:
        if alt == None:
            P = 1013.25
        else:
            P = alt_to_p(alt)
        
    a = array([31.8518,    34.3699,    40.3153,    42.0983,    56.7733,    69.0720,    71.8733])
    b = array([250.3193,   258.4759,   308.9894,   512.6857,   649.1343,   832.4566,   863.1927])
    c = array([-0.083393,  -0.089807,  -0.106248,  -0.120551,  -0.160859,  -0.199252,  -0.207069])
    d = array([7.4260e-5,  7.9457e-5,  9.4508e-5,  1.1752e-4,  1.5463e-4,  1.9391e-4,  2.0127e-4])
    e = array([-2.2397e-8, -2.3697e-8, -2.8234e-8, -3.8809e-8, -5.0330e-8, -6.3653e-8, -6.6043e-8])

    # create index latitudes
    ilats = arange(0,70,10)
    
    # make sure we're dealing with positive numbers so the next part doesn't fail
    lat = abs(lat)
    # latitudes above 60 deg should be equivalent the scaling at 60, replace them
    if type(lat) == np.ndarray:
        lat = array([x if x < 60 else 60 for x in lat])
    else:
        if lat > 60: lat = 60.0
    lat = np.array(lat)
    
    # create ratios for 0 through 60 degrees by ten degree intervals
    n = range(len(ilats))
    # calculate scaling factors for index latitudes
    f_lat = [a[x] + b[x] * exp(-P/150.0) + c[x] * P + d[x] * P**2 + e[x] * P**3 for x in n]

    # for requests with multiple latitudes we need to transpose the result
    f_lat = np.transpose(f_lat)
    
    # interpolate between the index latitude scaling factors and evaluate
    # the interpolation function at each sample latitude
    S = np.zeros(len(lat))
    for i, ifactors in enumerate(f_lat):
        S[i] = sp.interpolate.interp1d(ilats, ifactors)(lat[i])
    
    # muon scaling
    mk = array([0.587, 0.6, 0.678, 0.833, 0.933, 1.0, 1.0])
    fm_lat = [mk_i * exp((1013.25 - P) / 242.0) for mk_i in mk]
    fm_lat = np.transpose(fm_lat)
    # get muon scaling factors
    
    M = np.zeros(len(lat))
    for i, ifactors in enumerate(fm_lat):
        M[i] = sp.interpolate.interp1d(ilats, ifactors)(lat[i])
    
    scalingfactors = S * Fsp + M * (1 - Fsp)
    
    return scalingfactors

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
    rad_lats = deg_lats * np.pi / 180.0
    return 42
