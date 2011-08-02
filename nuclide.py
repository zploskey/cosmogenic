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

    def __init__(self, constants='stone'):
        self.Natoms = 2.005e22 # atoms of O / g quartz, from John's program
        self.LAMBDA = 4.998e-7
        
        # probability factors from Heisinger et al. 2002b
        # for Be-10 production in quartz:
        self.fC = 0.704
        self.fD = 0.1828

        if constants == 'heisinger':
            # be10stopped_mu_yield = fC10 * fD10 * fstar10
            # superseded value for be10 yield, is ~ 0.000553
            self.fstar = 0.0043 / 1.096
            # 1.096 factor is to 07KNSTD from Balco's AlBe_changes_v221
            self.sigma190 = 8.6e-29 # from Balco AlBe_changes_v221
            self.sigma0 = self.sigma190 / 190 ** ALPHA
        elif constants == 'stone':
            # John Stone, pers. communication
            self.fstar = 0.0011 # already adjusted for 07KNSTD
            self.sigma0 = 8.0707e-31
            # for consistency calculate what sigma190 should be
            self.sigma190 = self.sigma0 * 190 ** ALPHA
        else:
            raise Exception('Unknown constants: %s' % constants)
        
        self.k_neg = self.fC * self.fD * self.fstar

        # Production rate in atoms / g / yr from Stone, adjusted for 07KNSTD ala
        # Balco's 2008 paper. This number apparently includes production from
        # fast muons, so I have opted to subtract them here.
        p_mu_dict = muon.P_mu_total(z=0, h=0, nuc=self, full_data=True)
        print 'p_mu_dict[P_fast] =', p_mu_dict['P_fast']
        self.P0 = 4.49 - p_mu_dict['P_fast']
        print 'P0 =', self.P0
    
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

