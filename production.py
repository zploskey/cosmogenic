import numpy as np
import scipy as sp
import nuclide
import muon
import scaling

LAMBDA_h = 160 # attenuation length of hadronic component in atm, g / cm2

def P_sp(z, rho, alt, lat, n):
    """
    Production rate due to spallation reactions
    """
    f_scaling = scaling.stone2000(lat=lat, alt=alt, Fsp=1)
    return  f_scaling * n.P0 * np.exp(-z * rho / LAMBDA_h)

def P_tot(z, rho, alt, lat, n):
    return P_sp(z, rho, alt, lat, n) + muon.P_mu_total(z, alt, n)['P_tot']
