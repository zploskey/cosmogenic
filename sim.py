#!/usr/bin/python

import numpy as np
from numpy import exp, log, array
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import np_util as util
import scipy.interpolate as itp

# sample specific stuff to tweak:

density = 2.67 # specific gravity
latitude = 49 # degrees

shielding = 1.0
scaling = 1.0

# constants
lambda_be10 = 4.998*10**-7 # half life = 1.387 myr from Chmeleff 2010
lambda_al26 = 9.832*10**-7
lambda_cl36 = 2.303*10**-6

# Stone production rate normalized to 07KNSTD, Balco AlBe_changes_v22
p10_St = 4.49 # +/- 0.39
sec_per_year = 3.15576 * 10 ** 7
f_mu_neg = 1 / (1.268 + 1) # fraction of negative muons (from Heisinger 2002)

LAMBDA_h = 160 # attenuation length of hadronic component in atm, g / cm2

Noxygen = 2.005 * 10**22 # atoms of O / g quartz, from John's program
Nsi = 1.0025*10**22 # atoms Si / g quartz for Al-26
Nca = 1.0739 * 10**20 # atoms Ca / g / % CaO
Nk = 1.2786 * 10**20 # atoms K / g / % K2O

# percentage CaO and K2O for Cl-36 production
pctCaO = 56.0 
pctK2O = 13.0

alpha = 0.75 # Heisinger exponent constant

# Old value from Heisinger
# be10sigma190 = 0.094*10**-27

# from Balco AlBe_changes_v221
be10sigma190 = 8.6*10**-29 

be10sigma0 = be10sigma190 / 190**alpha
al_be_ratio = 6.75
al26sigma0 = 2 * al_be_ratio * be10sigma0

# This gives a surface production rate of 2.9 atom/g/yr in K2O, as given in
# Heisinger 2002, paper 2
cl36sigma0k = 89.79 * 10**-30

cl36sigma0ca = 27.36 * 10**-30

# probability factors from Heisinger et al. 2002b
# for Be-10 production in O in quartz:
fC10 = 0.704
fD10 = 0.1828
fstar10 = 0.0043
# be10stopped_mu_yield = fC10 * fD10 * fstar10
# superseded value for be10 yield, is ~ 0.000553
be10stopped_mu_yield = 5.05*10**-4 # from Balco AlBe_changes_v221

# for Al-26 in quartz
fC26 = 0.296
fD26 = 0.6559
fstar26 = 2.2

# for Cl-36 in K-feldspar
fCk = 0.755
fDk = 0.8020
fstark = 3.5

# for Cl-36 in Ca-carbonate
fCca = 0.361
fDca = 0.8486
fstarca = 4.5

@util.autovec
def neg_mu_stoprate(z):
    """
    Heisinger et al. 2002b eq. 1
    stopping rate of negative muons
    """
    return f_mu_neg * mu_flux(z)

@util.autovec
def vert_mu_flux(z):
    """
    Vertical muon flux (Heisinger et al. 2002a eq. 1) at depth z (g / cm2)
    """
    h = z / 100.0 # depth in hg/cm2
    # calculate the flux in units cm-2 sr-1 s-1 
    if h < 2000:
        flux = 258.5 * exp(-5.5*10**-4 * h) / ((h + 210) * ((h + 10)**1.66 + 75))
    else:
        flux = 1.82*10**-6 * (1211 / h)**2 * exp(-h / 1211) + 2.84*10**-13
    flux *= sec_per_year # convert to cm-2 sr-1 yr-1
    return flux


@util.autovec
def angled_mu_flux(z, angle):
    """
    Muon flux (cm-2 sr-1 s-1) at a given angle from the zenith in radians
    Heisinger et al. 2002a eq. 3
    """
    return vert_mu_flux(z) * np.cos(angle) ** n(z)

@util.autovec
def n(z):
    """
    Exponent for the muon flux at an angle 
    Heisinger et al. 2002a eq. 4
    """
    h = z / 100.0
    return 3.21 - 0.297 * log(h + 42) + 1.21*10**-3 * h


@util.autovec
def fast_mu_flux(z):
    """
    Heisinger et al. 2002a eq 5
    muon flux in muons cm-2 yr-1
    """
    return 2 * np.pi * vert_mu_flux(z) / (n(z) + 1)

@util.autovec
def neg_mu_flux(z):
    """
    flux of negative muons in cm-2 yr-1
    """
    return f_mu_neg * fast_mu_flux(z)

@util.autovec
def R_mu_neg(z):
    """
    rate of stopped negative muons
    heisinger 2002b eq 6 
    """
    return np.abs(f_mu_neg * sp.derivative(fast_mu_flux, z, dx=0.1))

@util.autovec
def Ebar(z):
    """
    Mean rate of change of energy with depth
    Heisinger et al. 2002a eq. 11
    """
    h = z / 100.0 # atmospheric depth in hg/cm2
    mean_energy = 7.6 + 321.7 * (1 - exp(-h * 8.059*10**-4))
    mean_energy += 50.7 * (1 - exp(-h * 5.05*10**-5))
    return mean_energy

@util.autovec
def beta_ratio(z):
    """
    Heisinger et al. 2002a approximation of the beta function (eq 16)
    """
    h = z / 100.0
    if h >= 1000:
        return 0.885
    return 0.846 - 0.015 * log(h + 1) + 0.003139 * (log(h + 1))**2

@util.autovec
def beta_ratio_hgcm2(h):
    """
    Heisinger et al. 2002a approximation of the beta function (eq 16)
    """
    if h >= 1000:
        return 0.885
    return 0.846 - 0.015 * log(h + 1) + 0.003139 * (log(h + 1))**2

@util.autovec
def P10_mu_f(z):
    """
    Heisinger 2002a eq 14, production rate of nuclides by fast muons
    """
    return be10sigma0 * Noxygen * beta_ratio(z) * fast_mu_flux(z) * Ebar(z) ** 0.75

@util.autovec
def P26_mu_f(z):
    """
    Production rate of 26Al by fast muons at depth z
    """
    return al26sigma0 * Nsi * beta_ratio(z) * fast_mu_flux(z) * Ebar(z) ** 0.75

@util.autovec
def P36k_mu_f(z):
    return cl36sigma0k * Nk * pctK2O * beta_ratio(z) * fast_mu_flux(z) * Ebar(z) ** 0.75

@util.autovec
def P36ca_mu_f(z):
    return cl36sigm0ca * Nca * pctCaO * beta_ratio(z) * fast_mu_flux(z) * Ebar(z) ** 0.75

@util.autovec
def P10_n_mu_f(z):
    """
    Rate of neutron production by fast muons
    Heisinger et al. 2002a, eq. 21
    """
    return 4.8*10**-6 * beta_ratio(z) * fast_mu_flux(z) * Ebar(z)**alpha

@util.autovec
def P10_mu_neg(z):
    """
    Production rate of 10Be by negative muons
    """
    return R_mu_neg(z) * be10stopped_mu_yield

@util.autovec
def P26_mu_neg(z):
    """
    Production rate of 26Al by negative muons
    """
    return R_mu_neg(z) * fC26 * fD26 * fstar26

@util.autovec
def P36k_mu_neg(z):
    """
    Production rate of 36Cl in potassium by negative muons
    """
    return R_mu_neg(z) * fCk * fDk * fstark

@util.autovec
def P36ca_mu_neg(z):
    """
    Production rate of 36Cl in Ca by negative muons
    """
    return R_mu_neg(z) * fCca * fDca * fstarca

def P36_mu_neg(z):
    """
    Total production rate of 36Cl by negative muons
    """
    return P36k_mu_neg(z) + P36ca_mu_neg(z)

@util.autovec
def P_n_h(z):
    """
    The rate of thermalized neutrons at depth z
    Heisinger et al. 2002b, eq. 15
    """
    return 2525.0 * exp(-z / LAMBDA_h)

@util.autovec
def P36_n_mu_neg(z):
	"""
	Neutron production rate by negative muon capture
	Heisinger et al. 2002b eq. 16
	This function is currently non-functioning until I can find a source
	for neutron yields.
	"""
	return Noxygen * fc * fD * fn * R_mu_neg(z)
	
fnorm = 1 # for now, not sure what this should be

@util.autovec
def P36_n(z):
    """
    The production rate of 10Be by neutron capture
    Heisinger et al. 2002b, eq. 18
    """
    return (P36_n_h(z) * fnorm + P36_n_mu_f(z) + P36_n_UTh) * f36Be

@util.autovec
def P10_mu(z):
	"""
	Production rate of Be10 by muons at depth z.
	"""
	return P10_mu_f(z) + P10_mu_neg(z)

@util.autovec
def P10_h(z):
	"""
	Production rate of Be10 by hadrons (spallation reactions) at depth z.
	See Gosse and Phillips (2000)
	We subtract the fast muon flux from 
	"""
	return (p10_St * scaling - P10_mu_f(z)) * exp(-z / LAMBDA_h)

@util.autovec
def P10(z):
	"""
	The production rate of Be10 in quartz at depth z is the sum of
	production rates from the spallation rxns as well as fast and negative
	muons.
	"""
	return P10_h(z) + P10_mu(z)

def stone2000(lat, P, Fsp=0.978):
	a = array([31.8518,  34.3699,  40.3153,  42.0983,  56.7733, 69.0720, 71.8733])
	b = array([250.3193, 258.4759, 308.9894, 512.6857, 649.1343, 832.4566, 863.1927])
	c = array([-0.083393, -0.089807, -0.106248, -0.120551, -0.160859, -0.199252, -0.207069])
	d = array([7.4260e-5, 7.9457e-5, 9.4508e-5, 1.1752e-4, 1.5463e-4, 1.9391e-4, 2.0127e-4])
	e = array([-2.2397e-8, -2.3697e-8, -2.8234e-8, -3.8809e-8, -5.0330e-8, -6.3653e-8, -6.6043e-8])

	# index latitudes
	ilats = np.arange(0,70,10)
	
	# make sure we're dealing with an array so the next part doesn't fail
	lat = np.abs(array(lat))
	lat = array([x if x < 60 else 60 for x in lat])
	
	# create ratios for 0 through 60 degrees by ten degree intervals
	n = range(len(ilats))
	f_lat = [a[x] + b[x] * exp(-P/150.0) + c[x] * P + d[x] * P**2 + e[x] * P**3 for x in n]
	
	#S = array([])
	#for f in f_lat:
	#	S.append(interp1d(ilats, f))
	
	# interpolate the 
	S = itp.interp1d(ilats, f_lat)(lat)
	
# 	lat0 = a[0] + b[0] * exp(-P/150.0) + c(0) * P + d(0) * P**2 + e(0) * P**3
# 	lat10 = a[1] + b(1 exp(-P/150.0) + c[1] * P + d[1] * P**2 + e[1] * P**3
# 	lat20 = a[2] + b[2] * exp(-P/150.0) + c[2] * P + d[2] * P**2 + e[2] * P**3
# 	lat30 = a[3] + b[3] * exp(-P/150.0) + c[3] * P + d[3] * P**2 + e[3] * P**3
# 	lat40 = a[4] + b[4] * exp(-P/150.0) + c[4] * P + d[4] * P**2 + e[4] * P**3
# 	lat50 = a[5] + b[5] * exp(-P/150.0) + c[5] * P + d[5] * P**2 + e[5] * P**3
# 	lat60 = a[6] + b[6] * exp(-P/150.0) + c[6] * P + d[6] * P**2 + e[6] * P**3
	
	# production by muons
	
	mk = array([0.587, 0.6, 0.678, 0.833, 0.933, 1.0, 1.0])
	fm_lat = mk * exp((1013.25 - P) / 242.0)
	
	#M = array([])
	#for fm in fm_lat:
	#	M.append(interp1d(ilats, fm_lat))
	
	M = itp.interp1d(ilats, fm_lat)(lat)
	
	Fm = 1 - Fsp
	scalingfactor = S * Fsp + M * Fm
	
	return scalingfactor
	
	

def stone2000Rcsp(h, Rc):
	"""
	Cutoff-rigidity based scaling scheme based on Lal's spallation polynomials.
	
	h  = scalar atmospheric pressure (hPa)
	Rc = list of cutoff rigidities (GV)
	
	"""
	rad_lats = deg_lats * np.pi / 180.0
