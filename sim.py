#!/usr/bin/python

import numpy as np

import production

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

def multiglaciate(dz, t_gl, t_intergl, t_postgl, z, n, h_surface, lat):
    """Find the resulting concentration profile for a glacial history and site.
    
    This function predicts the concentration profile for a glacial history. The
    glacial history of the site is described in such a way that the parameters
    are easy to vary for the Monte Carlo simulation--i.e. the times of
    glacial and interglacial periods are in lengths rather than absolute ages.
    Depths of the sample and the depths eroded during each glaciation are both
    in units of g/cm2, avoiding tying our results to a rock density. Presumably 
    this could be adapted to handle variable densities but that might get
    really complicated.
    
    Parameters
    
    z: vector of samples depths beneath the modern surface (g/cm2)
    t_gl: vector of lengths of time spent ice covered in each glaciation (yr)
    t_intergl: vector, time spent exposed for each interglaciation  period (yr)
    dz: vector of the depths eroded during each glaciation (g/cm2)
    t_postgl: time the sample has been exposed since deglaciation (yr)
    h_surface: elevation of the modern day surface (m)
    n: nuclide object
    """
    L = n.LAMBDA
    # add the atoms created as we go back in time
    conc = simple_expose(z, t_postgl, n, h_surface, lat)
    z_cur = z.copy()    # start at current depths
    t_begint = t_postgl # the age when the current interglacial began
    t_endint = 0.0      # age when current interglacial ended
    for i, z_rem in enumerate(dz):
        z_cur += z_rem # go back to depth and time before glacial erosion
        t_endint = t_begint + t_gl[i]
        t_begint = t_endint + t_intergl[i]
        
        conc += expose(z_cur, t_begint, t_endint, n, h_surface, lat)
        # p = production.P_tot(z_cur, h, lat, n)
        # add atoms made during this interglacial period that haven't decayed
        # conc += (p / L) * (np.exp(-L * t_endint) - np.exp(-L * t_begint))
    
    return conc

def depth_v_time(gl, intergl, postgl, dz):
    """ Returns a tuple of times and depths of a surface sample.
    
    gl: vector of lengths of each glaciation (yr)
    intergl: vector of lengths of interglacial periods (yr)
    postgl: time since last deglaciation (yr)
    dz: vector of glacial erosion depths during each glaciation 
    
    """
    assert gl.size == intergl.size == dz.size
    # interleave the two arrays
    tmp = np.column_stack((gl, intergl)).reshape(1, gl.size * 2).flatten()
    t = np.add.accumulate(np.concatenate(([0, postgl], tmp)))
    tmp = np.column_stack((dz, np.zeros(dz.size))).reshape(1, dz.size * 2).flatten()
    z = np.add.accumulate(np.concatenate(([0, 0], tmp)))
    return (t, z)

def expose(z, t_init, t_final, n, h, lat):
    L = n.LAMBDA
    p = production.P_tot(z, h, lat, n)
    conc = (p / L) * (np.exp(-L * t_final) - np.exp(-L * t_init))
    return conc

def simple_expose(z, t_exp, n, h, lat):
    # calculate the production rate
    p = production.P_tot(z, h, lat, n)
    return (p / n.LAMBDA) * (1 - np.exp(-n.LAMBDA * t_exp))

def fwd_profile(z0, z_removed, t, n, h, lat):
    """
    Calculates the nuclide concentration profile resulting from repeated
    glaciation of a bedrock surface.

    In all parameters that reference time, time is zero starting at modern day
    and increases into the past.

    z0: modern depths at which we want predicted concentrations (g/cm2)
    z_removed: list of depths of rock removed in successive glaciations (g/cm2)
    t: ages of switching between glacial/interglacial (array of times in years)
    exposed to cosmic rays in the recent past (in years). The first element of
    this array should be the exposure time since deglaciation, increasing after.
    n: the nuclide being produced (nuclide object)
    h: elevation of the site (m)
    lat: latitude of the site (degrees) 
    """
    L = n.LAMBDA
    N = np.zeros(len(z0))
    t_beg = t[2::2]
    t_end = t[1::2]

    # Add nuclides formed postglacially
    N += simple_expose(z0, t[0], n, h, lat)
    
    z_cur = z0.copy()
    for i, dz in enumerate(z_removed):
        z_cur += dz
        p = production.P_tot(z_cur, h, lat, n)
        N += (p / L) * (np.exp(-L * t_end[i]) - np.exp(-L * t_beg[i]))
        
    return N

def fwd_profile_steps(z0, z_removed, t, n, h, lat):
    """
    The profile at end of each interglacial period.
    """
    L = n.LAMBDA
    N = np.zeros(len(z0))
    Ns = np.zeros( (len(t) + 1, len(z0)) )
    t_beg = t[2::2]
    t_end = t[1::2]

    # Add nuclides formed postglacially
    z_cur = z0.copy()
    for i, dz in enumerate(z_removed):
        z_cur += dz
        p = production.P_tot(z_cur, h, lat, n)
        buildup =  (p / L) * (np.exp(-L * t_end[i]) - np.exp(-L * t_beg[i]))
        N += buildup
        Ns[i,:] = N.copy().T
    Nhol = simple_expose(z0, t[0], n, h, lat)
    N += Nhol
    Ns[-1,:] = N
    
    return Ns

def rand_erosion_hist(avg, sigma, n):
    """
    Generates a sequence of n numbers randomly distributed about avg_dz and
    standard deviation approximately equal to sigma.
    """
    return np.random.normal(avg, sigma, n)
