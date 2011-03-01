#!/usr/bin/python

from __future__ import division

import numpy as np
import scipy.integrate
import scipy.interpolate

import scaling

SEC_PER_YEAR = 3.15576 * 10 ** 7 # seconds per year
ALPHA = 0.75 # empirical constant from Heisinger
SEA_LEVEL_PRESSURE = 1013.25 # hPa
F_NEGMU = 1 / (1.268 + 1) # fraction of negative muons (from Heisinger 2002)

# GENERAL MUONS SECTION

def phi_vert_slhl(h):
    """
    Vertical muon flux (Heisinger et al. 2002a eq. 1) at depth z (g / cm2)
    at sea level / high latitude in cm-2 sr-1 s-1
    """
    def flux_lt2k(h):
        return 258.5 * np.exp(-5.5e-4 * h) / ((h + 210) * ((h + 10)**1.66 + 75))

    def flux_gt2k(h):
        return 1.82e-6 * (1211.0 / h)**2 * np.exp(-h / 1211.0) + 2.84e-13        
    h = np.atleast_1d(h) / 100.0 # depth in hg/cm2

    # calculate the flux in units cm-2 sr-1 s-1
    flux = np.zeros(h.size)
    i_lt_2k = np.where(h < 2000)
    flux[i_lt_2k] = flux_lt2k(h[i_lt_2k])
    i_gt_2k = np.where(h >= 2000)
    flux[i_gt_2k] = flux_gt2k(h[i_gt_2k])

    return flux

def n(z):
    """
    Exponent for the muon flux at an angle
    Takes z (g/cm2)
    Heisinger et al. 2002a eq. 4
    """
    return 3.21 - 0.297 * np.log((z / 100.0) + 42)  + 1.21e-5 * z

def phi_slhl(z):
    """
    Heisinger et al. 2002a eq 5
    Total muon flux in muons cm-2 yr-1
    """
    return 2 * np.pi * phi_vert_slhl(z) / (n(z) + 1)

# @memory.cache
def ebar(z):
    """
    Mean rate of change of energy with depth
    Heisinger et al. 2002a eq. 11
    """
    h = z / 100.0 # atmospheric depth in hg/cm2
    mean_energy = 7.6 + 321.7 * (1 - np.exp(-h * 8.059e-4))
    mean_energy += 50.7 * (1 - np.exp(-h * 5.05e-5))
    return mean_energy

# @memory.cache
def beta(z):
    """
    Heisinger et al. 2002a approximation of the beta function (eq 16)
    """
    z = np.atleast_1d(z)
    h = z / 100.0

    loghp1 = np.log(h + 1)
    b = 0.846 - 0.015 * loghp1 + 0.003139 * loghp1**2
    b[h >= 1000] = 0.885

    return b

def p_fast_slhl(z, n):
    """
    Heisinger 2002a eq 14, production rate of nuclides by fast muons
    z is depth in g/cm2
    n is the a nuclide object such as Be10Qtz
    """
    return n.sigma0 * beta(z) * phi_slhl(z) * ebar(z)**ALPHA * n.Natoms

def p_fast(z, flux, n):
    """
    Fast neutron production rate at sample site
    Takes:
    z: depth in g/cm2
    flux: muons flux in muons cm-2 yr-1
    n: nuclide object with properties sigma0 and Natoms
    """
    Beta = beta(z)
    Ebar = ebar(z)
    prod_rate = n.sigma0 * Beta * flux * Ebar**ALPHA * n.Natoms
    return (prod_rate, Beta, Ebar)

def R(z):
    """
    rate of stopped muons
    from heisinger 2002b eq 6 
    """
    return -scipy.derivative(phi_slhl, z, dx=0.1) #@UndefinedVariable

# @memory.cache
def Rv0(z):
    """
    Analytical solution for the stopping rate of the muon flux at sea
    level and high latitude for a depth (z) in g/cm2.
    in muons/g/s/sr
    """
    z = np.atleast_1d(z)
    stop_rate = np.zeros(z.size)

    lt = z < 200000    
    zl = z[lt]
    a = np.exp(-5.5e-6 * zl)
    b = zl + 21000
    c = (zl + 1000)**1.66 + 1.567e5
    dadz = -5.5e-6 * np.exp(-5.5e-6 * zl)
    dbdz = 1
    dcdz = 1.66 *(zl + 1000)**0.66
    stop_rate[lt] = -5.401e7 * (b * c * dadz - a * (c * dbdz + b * dcdz)) / (b**2 * c**2)
    
    gt = z >= 200000
    zg = z[gt]
    f = (121100.0 / zg)**2
    g = np.exp(-zg / 121100.0)
    dfdz = -2 * (121100)**2 / zg**3
    dgdz = -np.exp(-zg / 121100.0) / 121100.0
    stop_rate[gt] = -1.82e-6 * (dfdz * g + dgdz * f)
    
    return stop_rate

# PRODUCTION FROM NEGATIVE MUONS

def R_nmu(z):
    """
    rate of stopped negative muons
    heisinger 2002b eq 6 
    """
    return F_NEGMU * R(z)

# GENERAL MUONS
momentums = np.array([47.04, 56.16, 68.02, 85.1, 100, 152.7, 176.4, 221.8,
                      286.8, 391.7, 494.5, 899.5, 1101, 1502, 2103, 3104, 4104,
                      8105, 10110, 14110, 20110, 30110, 40110, 80110, 100100,
                      140100,200100,300100,400100,800100])
ranges = np.array([0.8516, 1.542, 2.866, 5.70, 9.15, 26.76, 36.96, 58.79, 93.32,
                   152.4, 211.5, 441.8, 553.4, 771.2, 1088, 1599, 2095, 3998,   
                   4920, 6724, 9360, 13620, 17760, 33430, 40840, 54950, 74590,
                   104000, 130200, 212900])
# interpolate the log range momentum date
log_LZ_interp = scipy.interpolate.interp1d(np.log(ranges), np.log(momentums))

# @memory.cache
def LZ(z):
    """
    Converts muon range to momentum
    Effective atmospheric attentuaton length for muons of range z
    
    From Heisinger 2002
    """
    z = np.atleast_1d(z)
    
    z[z < 1] = 1 # make sure we don't take the log of anything < 1
    
    P_MeVc = np.exp(log_LZ_interp(np.log(z)))

    atten_len = 263.0 + 150.0 * (P_MeVc / 1000.0)
    return atten_len

# @memory.cache
def P_mu_total(z, h, nuc, is_alt=True, full_data=False):
    """
    Total production rate from muons
    
    Takes:
    z: a vector of depths
    h: altitude in meters or the atmospheric pressure in hPa at surface
    n: a nuclide object containing nuclide specific information
    is_alt (optional): makes h be treated as an altitude in meters
    
    Returns the total production rate from muons in atoms
    """
    z = np.atleast_1d(z)
    h = np.atleast_1d(h)

    if z.size != h.size and h.size != 1:
        raise ValueError("z and h must be arrays of the same length or h must "
                         " be a scalar")

    # if h is an altitude instead of pressure, convert to pressure
    if is_alt:
        h = scaling.alt_to_p(h)
    
    # calculate the atmospheric depth of the sample
    H = 1.019716 * (SEA_LEVEL_PRESSURE - h) # g/cm2
    
    # find the stopping rate of vertical muons at SLHL
    R_v0 = Rv0(z)
    
    # calculate vertical muon stopping rate at the site
    L = LZ(z) # save this for later calculations
    R_v = R_v0 * np.exp(H / L) # vertical muons stopping rate at site

    phi_v0 = phi_vert_slhl(z)

    # integrate the stopping rate to get the vertical muon flux at depth z
    # at the sample site
    phi_v = np.zeros(z.size)
    int_err = phi_v.copy()
    tol = phi_v0 * 1e-4 # absolute error tolerance
    lim = 2e5+1 # limit of our integration
    for i, zi in enumerate(z):
        if H.size == 1:
            pressure = H
        else:
            pressure = H[i]
        
        def Rv(z):
            return Rv0(z) * np.exp(pressure / L[i])
        
        phi_v[i], int_err[i] = scipy.integrate.quad(Rv, zi, lim, epsabs=tol[i])
    
    # add in the flux below 2e5 g / cm2, assumed to be constant
    phi_v += phi_vert_slhl(2e5)
    
    nofz = n(z + H) # calculate exponent for total depth (atmosphere + rock)
    dndz = (-0.297 / 100.0) / ((z + H) / 100.0 + 42) + 1.21e-5 # d(n(z))/dz
    
    # find total flux of muons at the site
    phi = 2 * np.pi * phi_v / (nofz + 1) # muons/cm2/s
    phi *= SEC_PER_YEAR # convert to muons/cm2/yr
    
    # find total muon stopping rate of negative muons/g/s
    # R = fraction_of_negative_muons * derivative(tot_muon_flux(z+H))
    R = F_NEGMU * 2 * np.pi * (R_v / (nofz + 1) + phi_v * (nofz + 1)**-2 * dndz)
    R *= SEC_PER_YEAR # convert to negative muons/g/yr
    
    # get nuclide production rates
    (P_fast, Beta, Ebar) = p_fast(z, phi, nuc) # for fast muons
    P_neg = R * nuc.k_neg # and negative muons
    P_tot = P_fast + P_neg # total production from muons, atoms/g/yr
    
    if not full_data:
        return P_tot
    else:
        # flux of vertical muons at sea level/high latitude
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
                'P_tot': P_tot,}
                
if __name__ == '__main__':
    import nuclide
    import pprint
    be10 = nuclide.Be10Qtz()
    z = 0
    h = 0
    p_mu = P_mu_total(z, h, nuc=be10, full_data=True)
    pp = pprint.PrettyPrinter()
    pp.pprint(p_mu)
