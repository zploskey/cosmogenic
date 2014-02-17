"""
Performs calculations for cosmogenic nuclide production rates from muons.
"""

from __future__ import division, print_function, unicode_literals

import numpy as np
import scipy as sp
import scipy.misc
import scipy.integrate
import scipy.interpolate

from cosmogenic import scaling

SEC_PER_YEAR = 3.15576e7  # seconds per year
ALPHA = 0.75  # empirical constant from Heisinger
SEA_LEVEL_PRESSURE = 1013.25  # hPa
F_NEGMU = 1 / (1.268 + 1)  # fraction of negative muons (from Heisinger 2002)
A_GRAVITY = 9.80665  # standard gravity
LIMIT = 2e5  # g cm-2, upper limit of muon flux integration
MAX_RANGE = 1e8  # g cm-2, maximum value of muon range

def phi_sl(z):
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


def p_fast_sl(z, n):
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


def vert_flux_lt2k_hgcm2(z):
    """
    Heisinger 2002a equation 1
    z: depth in hg / cm^2
    """
    return 258.5 * np.exp(-5.5e-4 * z) / ((z + 210) * ((z + 10) ** 1.66 + 75))


def vert_flux_gt2k_hgcm2(z):
    """
    Heisinger 2002a equation 2
    z: depth in hg / cm^2
    """
    return 1.82e-6 * (1211.0 / z) ** 2 * np.exp(-z / 1211.0) + 2.84e-13


def phi_vert_sl(z):
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


def n_exponent(z):
    """
    Exponent for the muon flux at an angle
    Takes lithospheric depth z (g/cm**2)
    Heisinger et al. 2002a eq. 4 converted to units of g/cm**2
    """
    return 3.21 - 0.297 * np.log((z / 100.0) + 42) + 1.21e-5 * z


def ebar(z):
    """
    Mean rate of change of energy with depth
    Heisinger et al. 2002a eq. 11
    """
    h = np.atleast_1d(z / 100.0)  # atmospheric depth in hg/cm2
    mean_energy = 7.6 + 321.7 * (1 - np.exp(-h * 8.059e-4))
    mean_energy += 50.7 * (1 - np.exp(-h * 5.05e-5))
    return mean_energy


def beta(z):
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
    prod_rate = n.sigma0 * Beta * flux * Ebar ** ALPHA * n.Natoms
    return (prod_rate, Beta, Ebar)


def R(z):
    """
    rate of stopped muons
    from heisinger 2002b eq 6
    """
    return -sp.misc.derivative(phi_sl, z, dx=0.1)


def R_nmu(z):
    """
    rate of stopped negative muons
    heisinger 2002b eq 6
    """
    return F_NEGMU * R(z)


def Rv0(z):
    """
    Analytical solution for the stopping rate of the muon flux at sea
    level and high latitude for a depth (z) in g/cm2.
    in muons/g/s/sr
    """
    z = np.atleast_1d(z)
    stop_rate = np.zeros(z.size)

    lt = z < 200000.0
    zl = z[lt]
    a = np.exp(-5.5e-6 * zl)
    b = zl + 21000.0
    c = (zl + 1000.0) ** 1.66 + 1.567e5
    dadz = -5.5e-6 * np.exp(-5.5e-6 * zl)
    dbdz = 1.0
    dcdz = 1.66 * (zl + 1000.0) ** 0.66
    stop_rate[lt] = (-5.401e7 * (b * c * dadz - a * (c * dbdz + b * dcdz))
            / (b ** 2 * c ** 2))

    gt = ~lt
    zg = z[gt]
    f = (121100.0 / zg) ** 2
    g = np.exp(-zg / 121100.0)
    dfdz = -2.0 * (121100.0) ** 2 / zg ** 3
    dgdz = -np.exp(-zg / 121100.0) / 121100.0
    stop_rate[gt] = -1.82e-6 * (dfdz * g + dgdz * f)

    return stop_rate

# PRODUCTION FROM NEGATIVE MUONS

def interpolate_momentum_range():
    """
    Interpolate muon momentum (MeV/c) as a function of range in g cm-2.
    
    Uses data from:
    
    Groom, D.E., Mokhov, N.V., and Striganov, S.I., 2001.
    Muon stopping power and range tables 10 MeV - 100 TeV.
    Atomic data and nuclear data tables, volume 78,
    number 2, pp. 183-356.
    """

    momentum_data = np.array([47.04, 56.16, 68.02, 85.1, 100, 152.7, 176.4, 221.8,
                        286.8, 391.7, 494.5, 899.5, 1101, 1502, 2103, 3104, 4104,
                        8105, 10110, 14110, 20110, 30110, 40110, 80110, 100100,
                        140100, 200100, 300100, 400100, 800100,
                        2.453e5,
                        2.990e5,
                        3.616e5,
                        4.384e5,
                        4.957e5,
                        6.400e5,
                        6.877e5,
                        7.603e5,
                        8.379e5,
                        9.264e5,
                        9.894e5,
                        1.141e6,
                        1.189e6])

    range_data = np.array([0.8516, 1.542, 2.866, 5.70, 9.15, 26.76, 36.96, 58.79, 93.32,
                    152.4, 211.5, 441.8, 553.4, 771.2, 1088, 1599, 2095, 3998,
                    4920, 6724, 9360, 13620, 17760, 33430, 40840, 54950, 74590,
                    104000, 130200, 212900,
                    1e6,
                    1.4e6,
                    2e6,
                    3e6,
                    4e6,
                    8e6,
                    1e7,
                    1.4e7,
                    2e7,
                    3e7,
                    4e7,
                    8e7,
                    1e8])

    logp = sp.interpolate.interp1d(np.log(range_data), np.log(momentum_data))
    momentum_function = lambda z: np.exp(logp(np.log(z)))
    
    return momentum_function


momentum = interpolate_momentum_range()


def LZ(z):
    """
    Effective atmospheric attentuation length for muons of range z

    From Heisinger 2002b, p. 365.
    """
    zin = np.atleast_1d(z).astype(np.double)
    toolow_idxs = zin < 1.0
    zin[toolow_idxs] = 1.0
    zin[zin > MAX_RANGE] = np.nan
    P_MeVc = momentum(zin)
    atten_len = 263.0 + 150.0 * (P_MeVc / 1000.0)
    return atten_len if zin.size > 1 else atten_len[0]


def P_mu_total(z, n, h=0.0, is_alt=True, full_data=False):
    """
    Total production rate from muons

    Takes:
    z: a scalar or vector of sample depths in g/cm2
    n: a nuclide object containing nuclide specific information
    h: altitude in meters or the atmospheric pressure in hPa at surface, scalar
    is_alt (optional): makes h be treated as an altitude in meters

    Returns the total production rate from muons in atoms / g / yr
    """
    z = np.atleast_1d(z).astype(np.double)
    # if h is an altitude instead of pressure, convert to pressure
    if is_alt:
        h = scaling.alt_to_p(h)

    # Calculate the flux of vertically traveling muons as a function of depth
    # at sea level and high latitude
    phi_v0 = phi_vert_sl(z)

    # Calculate the stopping rate of vertically traveling muons as a function
    # of depth at sea level and high latitude, which is equivalent to the
    # range spectrum of vertically traveling muons at the surface.
    R_v0 = Rv0(z)

    # 3. Adjust range spectrum of vertically traveling muons for elevation

    # Calculate the atmospheric depth of each sample in g/cm2 between
    # the site and sea level
    deltaH = 10.0 * (SEA_LEVEL_PRESSURE - h) / A_GRAVITY

    # calculate vertical muon stopping rate at the site
    L = LZ(z)  # saved for full data report
    R_v = R_v0 * np.exp(deltaH / L)  # vertical muons stopping rate at site

    def Rv(z):
        return Rv0(z) * np.exp(deltaH / LZ(z))

    # Integrate the stopping rate to get the vertical muon flux at depth z
    # at the sample site
    phi_v = np.zeros_like(z)
    tol = 1e-6  # relative error tolerance

    # we want to do the integrals in the order of decreasing depth
    # so we can accumulate the flux as we go up

    # indexes of the sorted, then reversed array
    rev_sort_idxs = np.argsort(z)[::-1]
    
    # reorder z to be in increasing order
    zsorted = z[rev_sort_idxs]
    
    # start with the flux below 2e5 g / cm2, assumed to be constant
    a = 258.5 * 100 ** 2.66
    b = 75 * 100 ** 1.66
    phi_200k = ((a / ((LIMIT + 21000.0) * (((LIMIT + 1000.0) ** 1.66) + b)))
            * np.exp(-5.5e-6 * LIMIT))
    phi_v += phi_200k
    
    # keep track of the vertical flux at the previous depth
    prev_phi_v = phi_200k
    prev_z = LIMIT
    for i, zi in enumerate(zsorted):
        idx = rev_sort_idxs[i]

        if zi > LIMIT:
            continue

        phi_v[idx], _ = scipy.integrate.quad(
                    Rv, zi, prev_z, epsrel=tol, epsabs=0)
        phi_v[idx] += prev_phi_v
        prev_phi_v = phi_v[idx]
        prev_z = zi

    nofz = n_exponent(z + deltaH)  # exponent for total depth (atmosphere + rock)
    # d(n(z))/dz
    dndz = (-0.297 / 100.0) / ((z + deltaH) / 100.0 + 42) + 1.21e-5

    # find total flux of muons at the site
    phi = 2 * np.pi * phi_v / (nofz + 1)  # muons/cm2/s
    phi *= SEC_PER_YEAR  # convert to muons/cm2/yr

    # find stopping rate of negative muons/g/s
    # R = fraction_of_negative_muons * derivative(tot_muon_flux(z+deltaH))
    R = F_NEGMU * 2 * np.pi * ((R_v / (nofz + 1)) + (phi_v * dndz
            / (nofz + 1) ** 2))
    R *= SEC_PER_YEAR  # convert to negative muons/g/yr

    # get nuclide production rates
    (P_fast, Beta, Ebar) = p_fast(z, phi, n)  # for fast muons
    P_neg = R * n.k_neg  # and negative muons
    P_mu_tot = P_fast + P_neg  # total production from muons, atoms/g/yr

    if not full_data:
        return P_mu_tot
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
                'P_mu_tot': P_mu_tot,
                'deltaH': deltaH,
                }

