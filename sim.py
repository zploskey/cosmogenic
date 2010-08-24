#!/usr/bin/python

import numpy as np
from numpy import exp, array, abs
import scipy as sp
import np_util as util
import production as prod

## constants
#LAMBDA_BE10 = 4.998e-7 # half life = 1.387 myr from Chmeleff 2010
#LAMBDA_AL26 = 9.832e-7
#LAMBDA_CL36 = 2.303e-6

## Stone production rate normalized to 07KNSTD, Balco AlBe_changes_v22
#p10_St = 4.49 # +/- 0.39


#LAMBDA_h = 160 # attenuation length of hadronic component in atm, g / cm2

#Noxygen = 2.005e22 # atoms of O / g quartz, from John's program
#Nsi = 1.0025e22 # atoms Si / g quartz for Al-26
#Nca = 1.0739e20 # atoms Ca / g / % CaO
#Nk = 1.2786e20 # atoms K / g / % K2O

## percentage CaO and K2O for Cl-36 production
#pctCaO = 56.0 
#pctK2O = 13.0

#alpha = 0.75 # Heisinger exponent constant

## Old value from Heisinger
## be10sigma190 = 0.094*10**-27

## from Balco AlBe_changes_v221
#be10sigma190 = 8.6e-29 


#al_be_ratio = 6.75
#al26sigma0 = 2 * al_be_ratio * be10sigma0

## This gives a surface production rate of 2.9 atom/g/yr in K2O, as given in
## Heisinger 2002, paper 2
#cl36sigma0k = 89.79e-30

#cl36sigma0ca = 27.36e-30

## for Al-26 in quartz
#fC26 = 0.296
#fD26 = 0.6559
#fstar26 = 2.2

## for Cl-36 in K-feldspar
#fCk = 0.755
#fDk = 0.8020
#fstark = 3.5

## for Cl-36 in Ca-carbonate
#fCca = 0.361
#fDca = 0.8486
#fstarca = 4.5

def fwd_profile(z0, z_removed, t, n, h, lat):
    """
    Calculates the nuclide concentration profile resulting from repeated
    glaciation of a bedrock surface.

    In all parameters that reference time, time is zero starting at modern day
    and increases into the past.

    z0: modern depths at which we want predicted concentrations (g/cm2)
    z_removed: list of depths of rock removed in successive glaciations (g/cm2)
    t: ages of switching between glacial/interglacial (array of times in years)
    exposed to cosmic rays in the recent past (in years)
    n: the nuclide being produced (nuclide object)
    h: elevation of the site (m)
    lat: latitude of the site (degrees) 
    """
    L = n.LAMBDA
    N_postgl = simple_expose(z0, t[0], n, h, lat)

    n_samples = len(z0)
    N = np.zeros(n_samples)
    z_cur = z0.copy()
    for i, dz in enumerate(z_removed):
        z_cur += dz
        p = prod.P_tot(z_cur, h, lat, n)
        N += (p / L) * (exp(-L * t[2*i]) + exp(-L * t[2*i+1]))

    N += N_postgl
    return N


def simple_expose(z, t_exp, n, h, lat):
    # calculate the production rate
    p = prod.P_tot(z, h, lat, n)
    return (p / n.LAMBDA) * (1 - exp(-n.LAMBDA * t_exp))

# Don't mind the functions below, they have been superseded by muon.py and may
# not actually work as advertised.

@util.autovec
def neg_mu_flux(z):
    """
    flux of negative muons in cm-2 yr-1
    """
    return f_mu_neg * fast_mu_flux(z)

@util.autovec
def P26_mu_f(z):
    """
    Production rate of 26Al by fast muons at depth z
    """
    return al26sigma0 * Nsi * beta_ratio(z) * fast_mu_flux(z) * ebar(z) ** 0.75

@util.autovec
def P36k_mu_f(z):
    return cl36sigma0k * Nk * pctK2O * beta_ratio(z) * fast_mu_flux(z) * ebar(z) ** 0.75

@util.autovec
def P36ca_mu_f(z):
    return cl36sigm0ca * Nca * pctCaO * beta_ratio(z) * fast_mu_flux(z) * ebar(z) ** 0.75

@util.autovec
def P10_n_mu_f(z):
    """
    Rate of neutron production by fast muons
    Heisinger et al. 2002a, eq. 21
    """
    return 4.8*10**-6 * beta_ratio(z) * fast_mu_flux(z) * ebar(z)**alpha

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

def P_mu_total(z, h, consts):
	"""
	Calculate production rate of Al-26 or Be-10 from muons as a function of
	depth below the surface z (g/cm2) and site pressure h (hPa).
	"""
	
	# get atmospheric depth in g/cm2
	H = (1013.25 - h) * 1.019716
	


	
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
    The production rate of 36Cl by neutron capture
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
    See Gosse and Phillips (2001)
    We subtract the fast muon flux from 
    """
    return (p10_St * f_scaling - P10_mu_f(z)) * exp(-z / LAMBDA_h)

@util.autovec
def P10(z):
    """
    The production rate of Be10 in quartz at depth z is the sum of
    production rates from the spallation rxns as well as fast and negative
    muons.
    """
    return P10_h(z) + P10_mu(z)
