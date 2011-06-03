#!/usr/bin/python
"""
Cosmogenic nuclide classes to hold important constants
"""

import muon

ALPHA = 0.75

class Be10Qtz():
    """
    Data for the radioisotope Beryllium-10
    """
    def __init__(self):
        self.Natoms = 2.005e22 # atoms of O / g quartz, from John's program
        self.LAMBDA = 4.998e-7
        
        # probability factors from Heisinger et al. 2002b
        # for Be-10 production in O in quartz:
        self.fC = 0.704
        self.fD = 0.1828
        self.fstar = 0.0043
        
        # be10stopped_mu_yield = fC10 * fD10 * fstar10
        # superseded value for be10 yield, is ~ 0.000553
        self.k_neg = (self.fC * self.fD * self.fstar) / 1.096 
        # 1.096 factor is normalized to 07KNSTD from Balco's AlBe_changes_v221
        self.sigma190 = 8.6e-29 # from Balco AlBe_changes_v221
        self.sigma0 = self.sigma190 / 190 ** ALPHA

        # Production rate in atoms / g / yr from Stone, adjusted for 07KNSTD ala
        # Balco's 2008 paper. This number apparently includes production from
        # fast muons, so I have opted to subtract them here.
        self.P0 = 4.49 - muon.p_fast_slhl(0, self)
    
    def relative_error(self, concentration):
        return 0.9 * concentration**(-0.29402281)
    
    def measurement_error(self, concentration):
        """ Approximate measurement error for the concentration.
        
        Approximate measurement error for the concentration given in
        atoms per gram per year. Estimated from a plot from Balco.
        See balco_relerr_nov2.jpg or 
        http://cosmognosis.wordpress.com/2010/11/03/exotic-burial-dating-methods
        """
        return concentration * self.relative_error(concentration)
    
class Al26Qtz():
    """
    Aluminum-26 data
    """
    def __init__(self):
        self.Natoms = 1.0025e22 # atoms Si / g quartz
        self.k_neg = 0.296 * 0.6559 * 0.022
        self.delk_neg = 0.296 * 0.6559 * 0.002
        self.sigma190 = 1.41e-27
        self.delsigma190 = 0.17e-27
        self.sigma0 = self.sigma190 / 190 ** ALPHA

