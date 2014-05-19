#!python
#cython: cdivision=True
#cython: wraparound=False
#cython: overflowcheck=False
#cython: profile=False
"""
Performs calculations for cosmogenic nuclide production rates from muons.
"""

from __future__ import division, print_function

import numpy as np

import scipy as sp
import scipy.misc
import scipy.integrate
import scipy.interpolate

cimport numpy as np

np.import_array()

cimport cython

from libc.math cimport log
from libc.math cimport exp

from cosmogenic import scaling

SEC_PER_YEAR = 3.15576e7  # seconds per year
ALPHA = 0.75  # empirical constant from Heisinger
SEA_LEVEL_PRESSURE = 1013.25  # hPa
F_NEGMU = 1 / (1.268 + 1)  # fraction of negative muons (from Heisinger 2002)
A_GRAVITY = 9.80665  # standard gravity
LIMIT = 2e5  # g cm-2, upper limit of muon flux integration
MAX_RANGE = 1e8  # g cm-2, maximum value of muon range

cpdef phi_sl(z):
    """
    Calculate the sea level flux of muons (cm**-2 s**-1) at depth z.
    Heisinger et al. 2002a eq 5.

    Parameters
    ----------
    z : array_like, depths beneath surface [g / cm**2]

    Returns
    -------
    phi_sl : ndarray
             sea level muon flux
    """
    return 2 * np.pi * phi_vert_sl(z) / (n_exponent(z) + 1)


cpdef p_fast_sl(z, n):
    """
    Heisinger 2002a eq 14, production rate of nuclides by fast muons @ sea level
    
    Parameters
    ----------
    z : array_like, depth [g/cm**2]
    n : nuclide object such as Be10Qtz

    Returns
    -------
    p_fast_sl : Fast muon production rate at depth z in units atoms / g / yr
    """
    return (n.sigma0 * beta(z) * phi_sl(z) * ebar(z) ** ALPHA * n.Natoms
            * SEC_PER_YEAR)


cpdef vert_flux_lt2k_hgcm2(z):
    """
    Heisinger 2002a equation 1
    z: depth in hg / cm^2
    """
    return 258.5 * np.exp(-5.5e-4 * z) / ((z + 210) * ((z + 10) ** 1.66 + 75))


cpdef vert_flux_gt2k_hgcm2(z):
    """
    Heisinger 2002a equation 2
    z: depth in hg / cm^2
    """
    return 1.82e-6 * (1211.0 / z) ** 2 * np.exp(-z / 1211.0) + 2.84e-13


cpdef phi_vert_sl(z):
    """
    Calculate the flux of vertically traveling muons as a function of depth
    at sea level and high latitude.

    Vertical muon flux equation from Heisinger et al. 2002a eq. 1 and 2

    z: depth in  g / cm**2 for surface at sea level / high latitude

    Returns flux in muons cm-2 sr-1 s-1
    """
    h = np.atleast_1d(z / 100.0)  # depth in hg/cm2
    lt = h < 2000
    gt = ~lt
    flux = np.empty_like(h)
    flux[lt] = vert_flux_lt2k_hgcm2(h[lt])
    flux[gt] = vert_flux_gt2k_hgcm2(h[gt])
    return flux


cpdef n_exponent(z):
    """
    Exponent for the muon flux at an angle
    Takes lithospheric depth z (g/cm**2)
    Heisinger et al. 2002a eq. 4 converted to units of g/cm**2
    """
    return 3.21 - 0.297 * np.log((z / 100.0) + 42) + 1.21e-5 * z


cpdef ebar(z):
    """
    Mean rate of change of energy with depth
    Heisinger et al. 2002a eq. 11
    """
    h = np.atleast_1d(z / 100.0)  # atmospheric depth in hg/cm2
    mean_energy = 7.6 + 321.7 * (1 - np.exp(-h * 8.059e-4))
    mean_energy += 50.7 * (1 - np.exp(-h * 5.05e-5))
    return mean_energy


cpdef beta(z):
    """
    Heisinger et al. 2002a approximation of the beta function (eq 16)
    """
    h = np.atleast_1d(z / 100.0)  # atmospheric depth in hg/cm2
    res = np.empty_like(h)

    gt = h > 1000
    res[gt] = 0.885

    lt = ~gt
    loghp1 = np.log(h[lt] + 1)
    res[lt] = 0.846 - 0.015 * loghp1 + 0.003139 * loghp1 ** 2

    return res


cpdef p_fast(z, flux, n):
    """
    Fast neutron production rate at sample site
    Takes:
    z: depth in g/cm2
    flux: muons flux in muons cm-2 yr-1
    n: nuclide object with properties sigma0 and Natoms
    """
    Beta = beta(z)
    Ebar = ebar(z)
    prod_rate = n.sigma0 * Beta * flux * Ebar ** ALPHA * n.Natoms
    return (prod_rate, Beta, Ebar)


cpdef R(z):
    """
    rate of stopped muons
    from heisinger 2002b eq 6
    """
    return -sp.misc.derivative(phi_sl, z, dx=0.1)


cpdef R_nmu(z):
    """
    rate of stopped negative muons
    heisinger 2002b eq 6
    """
    return F_NEGMU * R(z)


cdef double Rv0(double z):
    """
    Analytical solution for the stopping rate of the muon flux at sea
    level and high latitude for a depth (z) in g/cm2.
    in muons/g/s/sr
    """
    
    if z < 200000.0:
        a = exp(-5.5e-6 * z)
        b = z + 21000.0
        c = (z + 1000.0) ** 1.66 + 1.567e5
        dadz = -5.5e-6 * exp(-5.5e-6 * z)
        dbdz = 1.0
        dcdz = 1.66 * (z + 1000.0) ** 0.66
        stop_rate = (-5.401e7 * (b * c * dadz - a * (c * dbdz + b * dcdz))
                / (b ** 2.0 * c ** 2.0))
    else:
        f = (121100.0 / z) ** 2.0
        g = exp(-z / 121100.0)
        dfdz = -2.0 * 121100.0 ** 2.0 / z ** 3.0
        dgdz = -exp(-z / 121100.0) / 121100.0
        stop_rate = -1.82e-6 * (dfdz * g + dgdz * f)

    return stop_rate


cdef double momentum(double z):
    """
    Analytical function for muon momentum as a function of range were
    fit to Groom 2001 data (see calcs/fitgroom.ipynb).
    
    Calculates momentum (MeV/c) as a function of muon range (g/cm-2).
    
    Groom, D.E., Mokhov, N.V., and Striganov, S.I., 2001.
    Muon stopping power and range tables 10 MeV - 100 TeV.
    Atomic data and nuclear data tables, volume 78,
    number 2, pp. 183-356.
    """
    cdef:
        double lz
        double slope
        double mom_int
        double mom
        double a, b, c

    if z < 1.0:
        z = 1.0
    
    lz = log(z)
    
    if z < 5.7:
        # fitted slope and intercept in log-log space
        slope = 0.31183384474750259
        mom_int = 3.9010907766258351
        mom = exp(slope * lz + mom_int)
    else:
        a = 0.066278146047421563
        b = 0.01506720229624511
        c = 4.2081356458976158
        mom = exp(a * (lz + b) ** 2.0 + c)
    
    return mom


cdef double LZ(double z):
    """
    Effective atmospheric attentuation length for muons of range z in g cm-2.

    From Heisinger 2002b, p. 365.
    """
    cpdef double mom = momentum(z)
    cpdef double atten_len = 263.0 + (150.0 / 1000.0) * mom
    
    return atten_len


cpdef double Rv(double x, double dH):
    return Rv0(x) * exp(dH / LZ(x))


@cython.boundscheck(False)
cpdef P_mu_total(z, object n, h=None, is_alt=True, full_data=False):
    """
    Total production rate from muons

    Takes:
    z: a scalar or vector of sample depths in g/cm2
    n: a nuclide object containing nuclide specific information
    h: altitude in meters or the atmospheric pressure in hPa at surface, scalar
    is_alt (optional): makes h be treated as an altitude in meters

    Returns the total production rate from muons in atoms / g / yr
    """
    if h is None:
        h = 0.0
    
    cdef:
        np.ndarray zarr = np.atleast_1d(z).astype(np.double)
        long M = zarr.shape[0]
        long i
    
    # if h is an altitude instead of pressure, convert to pressure
    if is_alt:
        h = scaling.alt_to_p(h)

    # Calculate the atmospheric depth of each sample in g/cm2 between
    # the site and sea level
    cdef double deltaH = 10.0 * (SEA_LEVEL_PRESSURE - h) / A_GRAVITY
    # Calculate the stopping rate of vertically traveling muons as a function
    # of depth at sea level and high latitude, which is equivalent to the
    # range spectrum of vertically traveling muons at the surface.
    cdef:
        np.ndarray[double] R_v0 = np.empty_like(zarr)
        np.ndarray[double] L = np.empty_like(zarr)
        np.ndarray[double] R_v = np.empty_like(zarr)
    
    for i in range(M):
        R_v0[i] = Rv0(zarr[i])

        # Adjust range spectrum of vertically traveling muons for elevation
        L[i] = LZ(zarr[i])  # saved for full data report
        
        # vertical muons stopping rate at site
        R_v[i] = R_v0[i] * exp(deltaH / L[i])  

    # Integrate the stopping rate to get the vertical muon flux at depth z
    # at the sample site
    cdef np.ndarray[double, ndim=1] phi_v = np.zeros_like(zarr)
    cdef double tol = 1e-4  # relative error tolerance

    # we want to do the integrals in the order of decreasing depth
    # so we can accumulate the flux as we go up

    # indexes of the sorted, then reversed array
    cdef np.ndarray[long] sort_idxs
    
    with cython.wraparound(True):
        sort_idxs = np.argsort(zarr)[::-1]        
    
    cdef double[:] zsorted = zarr[sort_idxs]    
        
    # start with the flux below 2e5 g / cm2, assumed to be constant
    cdef:
        double a = 258.5 * 100 ** 2.66
        double b = 75 * 100 ** 1.66
        double lim = LIMIT
        double phi_200k
    phi_200k = ((a / ((lim + 21000.0) * (((lim + 1000.0) ** 1.66) + b)))
            * exp(-5.5e-6 * lim))
    phi_v += phi_200k
    
    # keep track of the vertical flux at the previous depth
    cdef:
        double prev_phi_v = phi_200k
        cdef double prev_z = LIMIT
        cdef tuple argstup = (deltaH,)
        cdef double zi
        cdef long idx
    
    i = 0
    for i in range(M):
        zi = zsorted[i]
        idx = sort_idxs[i]
        if zi < LIMIT:
            phi_v[idx], _ = scipy.integrate.quad(
                        Rv, zi, prev_z, args=argstup, epsrel=tol, epsabs=0)
            phi_v[idx] += prev_phi_v
            prev_phi_v = phi_v[idx]
            prev_z = zi

    cdef:
        np.ndarray[double] nofz
        np.ndarray[double] dndz
        np.ndarray[double] phi
        np.ndarray[double] R
        np.ndarray[double] P_fast
        np.ndarray[double] Beta
        np.ndarray[double] Ebar
        np.ndarray[double] P_neg
        np.ndarray[double] P_mu_tot

    # exponent for total depth (atmosphere + rock)
    nofz = n_exponent(zarr + deltaH)  
    # d(n(z))/dz
    dndz = (-0.297 / 100.0) / ((zarr + deltaH) / 100.0 + 42) + 1.21e-5

    # find total flux of muons at the site
    phi = 2 * np.pi * phi_v / (nofz + 1)  # muons/cm2/s
    phi *= SEC_PER_YEAR  # convert to muons/cm2/yr

    # find stopping rate of negative muons/g/s
    # R = fraction_of_negative_muons * derivative(tot_muon_flux(z+deltaH))
    R = F_NEGMU * 2 * np.pi * ((R_v / (nofz + 1)) + (phi_v * dndz
            / (nofz + 1) ** 2))
    R *= SEC_PER_YEAR  # convert to negative muons/g/yr

    # get nuclide production rates
    (P_fast, Beta, Ebar) = p_fast(zarr, phi, n)  # for fast muons
    P_neg = R * n.k_neg  # and negative muons
    P_mu_tot = P_fast + P_neg  # total production from muons, atoms/g/yr

    cdef np.ndarray[double] phi_v0
    
    if not full_data:
        return P_mu_tot if P_mu_tot.size != 1 else P_mu_tot[0]
    else:
        # Calculate the flux of vertically traveling muons as a function of depth
        # at sea level and high latitude
        phi_v0 = np.empty_like(zarr)
        for i in range(M):
            phi_v0[i] = phi_vert_sl(zarr[i])
        
        return {'phi_v0': phi_v0,
                'R_v0': R_v0,
                'phi_v': phi_v,
                'R_v': R_v,
                'phi': phi,
                'R': R,
                'P_fast': P_fast,
                'Beta': Beta,
                'Ebar': Ebar,
                'P_fast': P_fast,
                'P_neg': P_neg,
                'LZ': L,
                'P_mu_tot': P_mu_tot,
                'deltaH': deltaH,
                }

