"""
Cosmogenic nuclide classes to hold important constants
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import math

from . import muon

ALPHA = 0.75
P10_REF_ST = 4.49 # atoms / g / yr
P36_REF_ST = 22.5 # atoms / g pure Kfeldspar / yr
P36_REF_ST_PUREKFELD = 160.0 # roughly, atoms / g K / yr

class Be10Qtz():
    """
    Data for the radioisotope Beryllium-10
    """

    def __init__(self, constants='stone'):
        self.Natoms = 2.0046e22 # atoms of O / g quartz
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
        
        # stopped/negative muon yield
        self.k_neg = self.fC * self.fD * self.fstar
        #self.delk_neg = self.fC * self.fD * self.delfstar
        
        # Production rate in atoms / g / yr from Stone, adjusted for 07KNSTD ala
        # Balco's 2008 paper. This number apparently includes production from
        # fast muons, so I have opted to subtract them here.
        p_mu_tot = muon.P_mu_total(z=0, h=0, nuc=self) # full_data=True)
        self.P0 = P10_REF_ST - p_mu_tot
    
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
    def __init__(self, constants='stone'):
        self.Natoms = 1.0025e22 # atoms Si / g quartz
        self.halflife = 7.17e5 # years
        self.LAMBDA = np.log(2) / 7.17e5 # years ** -1

        # modifying factors from Heisinger 2002b
        self.fC = 0.296  # chemical compound factor 
        self.fD = 0.6559 # probability that the muon does not decay
        
        if constants == 'heisinger':
            self.fstar = 0.022 # probability to emit Al-26
            self.delfstar = 0.002
            # sigma 190 from Heisinger 2002a, converted: 1 mb = 10**-27 cm**2
            self.sigma190 = 1.41e-27 # cm**2
            self.delsigma190 = 0.15e-27 # cm**2
            # the fast muon cross-section at 1 GeV
            self.sigma0 = self.sigma190 / 190 ** ALPHA
            self.delsigma0 = 3.3e-27 # cm**2
        elif constants == 'stone':
            # from second round NSF proposal, John Ston pers. comm.
            self.fstar = 0.0084 # probability to emit Al-26
            self.sigma0 =  13.6e-30 # cm**2, 1 microbarn = 1e-30 cm**2
            self.sigma190 = self.sigma0 * 190 ** ALPHA
        else:
            raise Exception('Unknown constants: %s' % constants)
        
        self.k_neg = self.fC * self.fD * self.fstar
        # self.delk_neg = self.fC * self.fD * self.delfstar
        p_mu_tot = muon.P_mu_total(z=0, h=0, nuc=self)
        self.R2610 = 6.02 * 1.106 # ratio of Al26 to Be10 production at surface
        self.P0 = (P10_REF_ST * self.R2610) - p_mu_tot
        
    def relative_error(self, concentration):
        """ Approximate fractional error for the concentration.
        
        Approximate measurement error for the concentration given in
        atoms per gram per year. Estimated from a plot from Balco.
        See balco_relerr_nov2.jpg or 
        http://cosmognosis.wordpress.com/2010/11/03/exotic-burial-dating-methods
        """
        return 4.40538328952 * concentration**(-0.32879674)
    
    def measurement_error(self, concentration):
        return concentration * self.relative_error(concentration)

class Cl36Kfeld():
    """
    Data for the radioisotope Chlorine-36 produced in K-feldspar.
    
    This can be used to predict the production rate in pure K-feldspar.
    """

    def __init__(self, constants='stone'):
        self.Natoms = 2.164e21 # atoms K / g Kfeldspar, from John's program
        self.LAMBDA = math.log(2) / 3.01e5 # decay constant, 1/yr
        
        # probability factors 
        self.fC = 0.12
        self.fD = 0.8020
        
        if constants == 'heisinger':
            raise NotImplementedError()
        elif constants == 'stone':
            # John Stone, pers. communication
            self.fstar = 0.0568
            self.sigma0 = 9.40e-30
            # for consistency calculate what sigma190 should be
            self.sigma190 = self.sigma0 * 190 ** ALPHA
        else:
            raise Exception('Unknown constants: %s' % constants)
        
        # stopped/negative muon yield
        self.k_neg = self.fC * self.fD * self.fstar
        #self.delk_neg = self.fC * self.fD * self.delfstar
        p_mu_tot = muon.P_mu_total(z=0, h=0, nuc=self)[0] 
        self.P0 = P36_REF_ST - p_mu_tot
    
    def relative_error(self, concentration):
        # for now assume same statistics as Al-26
        return 4.40538328952 * concentration**(-0.32879674)
    
    def measurement_error(self, concentration):
        return concentration * self.relative_error(concentration)
