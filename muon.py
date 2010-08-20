import np_util as util
import numpy as np
from numpy import array, abs, pi
import scipy as sp
import scipy.integrate as integrate
import scipy.interpolate as interp
import scaling

SEC_PER_YEAR = 3.15576 * 10 ** 7
ALPHA = 0.75
SEA_LEVEL_PRESSURE = 1013.25 # hPa

F_NEGMU = 1 / (1.268 + 1) # fraction of negative muons (from Heisinger 2002)


# GENERAL MUONS SECTION

@util.autovec
def phi_vert_slhl(z):
    """
    Vertical muon flux (Heisinger et al. 2002a eq. 1) at depth z (g / cm2)
    at sea level / high latitude in cm-2 sr-1 yr-1
    """
    h = z / 100.0 # depth in hg/cm2
    
    # calculate the flux in units cm-2 sr-1 s-1 
    if h < 2000:
        flux = 258.5 * np.exp(-5.5e-4 * h) / ((h + 210) * ((h + 10)**1.66 + 75))
    else:
        flux = 1.82e-6 * (1211 / h)**2 * np.exp(-h / 1211) + 2.84e-13
    flux *= SEC_PER_YEAR # convert to cm-2 sr-1 yr-1
    
    return flux

@util.autovec
def n(z):
    """
    Exponent for the muon flux at an angle 
    Heisinger et al. 2002a eq. 4
    """
    h = z / 100.0
    return 3.21 - 0.297 * np.log(h + 42) + 1.21e-3 * h

@util.autovec
def phi_slhl(z):
    """
    Heisinger et al. 2002a eq 5
    Total muon flux in muons cm-2 yr-1
    """
    return 2 * pi * phi_vert_slhl(z) / (n(z) + 1)

@util.autovec
def ebar(z):
    """
    Mean rate of change of energy with depth
    Heisinger et al. 2002a eq. 11
    """
    h = z / 100.0 # atmospheric depth in hg/cm2
    mean_energy = 7.6 + 321.7 * (1 - np.exp(-h * 8.059e-4))
    mean_energy += 50.7 * (1 - np.exp(-h * 5.05e-5))
    return mean_energy

@util.autovec
def beta(z):
    """
    Heisinger et al. 2002a approximation of the beta function (eq 16)
    """
    h = z / 100.0
    if h >= 1000:
        return 0.885
    return 0.846 - 0.015 * np.log(h + 1) + 0.003139 * (np.log(h + 1))**2

def p_fast_slhl(z, n):
    """
    Heisinger 2002a eq 14, production rate of nuclides by fast muons
    z is depth in g/cm2
    n is the a nuclide object such as Be10Qtz
    """
    return nuc.sigma0 * nuc.Natoms * beta(z) * phi_slhl(z) * ebar(z) ** ALPHA

def p_fast(z, flux, nuc):
    """
    Fast neutron production rate at sample site
    Takes:
    flux: muons flux in muons cm-2 yr-1
    """
    return nuc.sigma0 * nuc.Natoms * flux * beta(z) * ebar(z) ** ALPHA

def R(z):
    """
    rate of stopped muons
    from heisinger 2002b eq 6 
    """
    return -sp.derivative(phi_slhl, z, dx=0.1)

@util.autovec
def Rv0(z):
    """
    Analytical solution for the stopping rate of the muon flux at sea
    level and high latitude for a depth (z) in g/cm2. Derivative was 
    calculated with Mathematica 7.
    """
    if z < 200000:
        stop_rate = -(np.exp(-5.5e-6 * z) \
        * (-6.82489e13 - 1.36111e13 * (1000 + z)**0.66 - 4.35546e8 * (1000 + z)**1.66 \
        + z * (-3.36503e8 - 6.48146e8 * (1000 + z)**0.66 - 2147.47 * (1000 + z)**1.66))) \
        / ((21000 + z)**2 * (75 + 0.00047863 * (1000 + z)**1.66)**2)
    else:
        f = (121100.0 / z)**2
        g = np.exp(-z / 121100.0)
        dfdz = -2.0 * (121100)**2
        dgdz = -np.exp(-z/121100.0) / 121100.0
        stop_rate = -1.82e-6 * (dfdz * g + dgdz * f)
    return stop_rate 

# PRODUCTION FROM NEGATIVE MUONS

def R_nmu(z):
    """
    rate of stopped negative muons
    heisinger 2002b eq 6 
    """
    return F_NEGMU * R(z)

# GENERAL MUONS

def LZ(z):
	"""
	Converts muon range to momentum
	Effective atmospheric attentuaton length for muons of range Z
	
	From Heisinger 2002
	"""
	p = array([47.04,56.16,68.02,85.1,100,152.7,176.4,221.8,286.8,391.7,494.5,899.5,1101,1502,2103,3104,4104,8105,10110,14110,20110,30110,40110,80110,100100,140100,200100,300100,400100,800100])
	range = array([0.8516,1.542,2.866,5.70,9.15,26.76,36.96,58.79,93.32,152.4,211.5,441.8,553.4,771.2,1088,1599,2095,3998,4920,6724,9360,13620,17760,33430,40840,54950,74590,104000,130200,212900])
	# interpolate the log range momentum date
	ifnc = interp.interp1d(np.log(range), np.log(p))
	P_MeVc = np.exp(ifnc(np.log(z)))
	return 263 + 150 * (P_MeVc / 1000.0)

def P_mu_total(z, h, nuc, is_alt=False):
    """
    Total production rate from muons
    
    Takes:
    z: a vector of depths
    h: atmospheric pressure in hPa at surface (or altitude in m, see below)
    n: a nuclide object containing nuclide specific information
    is_alt (optional): makes h be treated as 
    """
    t = type(z)
    if (t == 'numpy.ndarray' or t == 'list') and len(z) != len(h):
        raise ValueError("z and h must be arrays of the same length")
    
    if is_alt:
         h = scaling.alt_to_p(h)
    
    # calculate the atmospheric depth of the sample
    H = 1.019716 * (SEA_LEVEL_PRESSURE - h) 
    
    # flux of vertical muons at sea level/high latitude
    phi_v0 = phi_vert_slhl(z) # save for later?

    # find the stopping rate of vertical muons at SLHL
    R_v0 = Rv0(z)
    
    # calculate vertical muon stopping rate at the site
    L = LZ(z)
    R_v = R_v0 * np.exp(H / L)
    
    # integrate the stopping rate to get the vertical muon flux at depth z
    # at the sample site
    z = np.array(z)
    phi_v = np.zeros(len(z))
    i = 0
    for zi in z:
        phi_v_tmp = integrate.quad(lambda x: Rv0(x) * np.exp(H[i]/L[i]), zi, 2e5)
        phi_v[i] = phi_v_tmp[0]
        i += 1
    
    # add in the flux below 2e5 g / cm2, assumed to be constant
    phi_v += phi_vert_slhl(2e5)
    
    # find total flux of muons
    phi = 2 * pi * phi_v / (n(z + H) + 1)
    
    # calculate total muon stopping rate at depth z + H
    # R = derivative(tot_muon_flux(z+H))
    nofz = n(z + h)
    dndz = -0.297 / ((z + H) + 4200) + 1.21e-5 # derivative of 
    R = 2 * pi * (R_v / (nofz + 1) - phi_v * nofz**-2 * dndz)
    
    R_neg = F_NEGMU * R # stopping rate of negative muons
    
    # get nuclide production rates
    P_fast = p_fast(z, phi, nuc) # for fast muons
    P_neg = R_neg * nuc.k_neg # and negative muons
    P_tot = P_fast + P_neg
    
    return {'P_tot': P_tot, 'P_fast': P_fast, 'P_neg': P_neg, 'L': L, 'R': R, \
            'phi': phi, 'H': h, 'phi_v': phi_v, 'R_v': R_v, 'phi_v0': phi_v0, 'R_v0': R_v0}
    
    
    
    