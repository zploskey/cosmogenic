"""
Cosmogenic nuclide classes to hold important constants.

Nuclides in this class contain reference production rates of the cosmogenic
nuclide from spallation for each scaling scheme as a dictionary called
"scaling_p_sp_ref". Production rate values are from Borchers et al. (2014)
unless otherwise noted.

Reference:
    Borchers et al. (2014) Geological calibration of spallation production
    rates in the CRONUS-Earth Project. Quaternary Geochronology, in press.

Production rates from muons are calculated using functions and
constants from Heisinger (2002a,b), except when using a revised
set of constants derived from fitting Beacon Valley core data
by John Stone and contained in the "stone" set of constants.

In general, classes in this module contain the data and functions
needed to describe the production pathway from target element to
the resulting nuclide.  They are named as:
    "name" + "mass number" + "target material"
where target material is typically a mineral, or in some cases,
a specific element.

Some base classes may omit the target material, but are
not intended to be directly used.

Each complete production pathway should contain the following
parameters:
    fC: chemical compound factor
    fD: probability that the muon does not decay in the k-shell
        before being captured by the nucleus
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import math

from . import production

ALPHA = 0.75  # constant from Heisinger 2002


class Nuclide(object):
    """Base object for all nuclides."""   

    def __init__(self):
        self.production_rate = self.get_production_rate()

    def get_production_rate(self, *args, **kwargs):
        """ Production rate"""
        
        def p(z, *args, **kwargs):
            return production.P_tot(z, n=self, *args, **kwargs)

        return p


class Be10Qtz(Nuclide):
    """
    Data for the radio-isotope Beryllium-10

    Parameters
    ----------
    constants : string, optional
                Muon constants to use. Accepts "heisinger" for the values
                from Heisinger 2002. Using "stone" (default) will use
                unpublished muon constants derived from a fit to a
                continously exposed and slowly eroding surface in
                Antarctica.
    """
    # map scaling scheme codes to reference spallation production rates
    scaling_p_sp_ref = {
        "Sa": 3.92,
        "St": 4.01,
        "Sf": 4.09,
        "Lm": 4.00,
        "De": 3.69,
        "Du": 3.70,
        "Li": 4.06,
    }

    def __init__(self, constants='stone'):
        self.Natoms = 2.0046e22  # atoms of O / g quartz
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
            self.sigma190 = 8.6e-29  # from Balco AlBe_changes_v221
            self.sigma0 = self.sigma190 / 190 ** ALPHA
        elif constants == 'stone':
            # John Stone, pers. communication
            self.fstar = 0.0011  # already adjusted for 07KNSTD
            self.sigma0 = 8.0707e-31
            # for consistency calculate what sigma190 should be
            self.sigma190 = self.sigma0 * 190 ** ALPHA
        else:
            raise Exception('Unknown constants: %s' % constants)

        # stopped/negative muon yield
        self.k_neg = self.fC * self.fD * self.fstar
        #self.delk_neg = self.fC * self.fD * self.delfstar
        super(Be10Qtz, self).__init__()

    def relative_error(self, concentration):
        """ Approximate relative error for concentration. """
        return 0.9 * concentration ** (-0.29402281)

    def measurement_error(self, concentration):
        """ Approximate measurement error for the concentration.

        Approximate measurement error for the concentration given in
        atoms per gram per year. Estimated from a plot from Balco.
        See balco_relerr_nov2.jpg or
        http://cosmognosis.wordpress.com/2010/11/03/exotic-burial-dating-methods
        """
        return concentration * self.relative_error(concentration)


class Al26Qtz(Nuclide):
    """
    Aluminum-26 data
    """

    # map scaling scheme codes to reference spallation production rates
    scaling_p_sp_ref = {
        "Sa": 28.54,
        "St": 27.93,  # sea level high latitude prod'n minus muon
        "Sf": 28.61,
        "Lm": 27.93,
        "De": 26.26,
        "Du": 26.29,
        "Li": 28.72,
    }

    def __init__(self, constants='stone'):
        self.Natoms = 1.0025e22  # atoms Si / g quartz
        self.halflife = 7.17e5  # years
        self.LAMBDA = np.log(2) / 7.17e5  # years ** -1

        # modifying factors from Heisinger 2002b
        self.fC = 0.296  # chemical compound factor
        self.fD = 0.6559  # probability that the muon does not decay

        if constants == 'heisinger':
            self.fstar = 0.022  # probability to emit Al-26
            self.delfstar = 0.002
            # sigma 190 from Heisinger 2002a, converted: 1 mb = 10**-27 cm**2
            self.sigma190 = 1.41e-27  # cm**2
            self.delsigma190 = 0.15e-27  # cm**2
            # the fast muon cross-section at 1 GeV
            self.sigma0 = self.sigma190 / 190 ** ALPHA
            self.delsigma0 = 3.3e-27  # cm**2
        elif constants == 'stone':
            # from second round NSF proposal, John Ston pers. comm.
            self.fstar = 0.0084  # probability to emit Al-26
            self.sigma0 = 13.6e-30  # cm**2, 1 microbarn = 1e-30 cm**2
            self.sigma190 = self.sigma0 * 190 ** ALPHA
        else:
            raise Exception('Unknown constants: %s' % constants)

        self.k_neg = self.fC * self.fD * self.fstar
        # self.delk_neg = self.fC * self.fD * self.delfstar
        super(Al26Qtz, self).__init__()

    def relative_error(self, concentration):
        """ Approximate fractional error for the concentration.

        Approximate measurement error for the concentration given in
        atoms per gram per year. Estimated from a plot from Balco.
        See balco_relerr_nov2.jpg or
        http://cosmognosis.wordpress.com/2010/11/03/exotic-burial-dating-methods
        """
        return 4.40538328952 * concentration ** (-0.32879674)

    def measurement_error(self, concentration):
        """ Approximate measurement error for the concentration. """
        return concentration * self.relative_error(concentration)


class Cl36(Nuclide):
    LAMBDA = math.log(2) / 3.01e5  # decay constant, 1/yr


class Cl36CaCO3(Cl36):
    """
    Chlorine-36 in calcium.
    """

    # factors from Heisinger 2002b rel. to slow negative muon production
    fC = 0.361
    fD = 0.8486
    
    # number of atoms of Ca per g of CaCO3
    Natoms = 6.017e21

    def __init__(self, constants='heisinger', fCa=0.4004):
        # map scaling scheme codes to reference spallation production rates
        self.scaling_p_sp_ref = {
            "Sa": 56.27 * fCa,
            "St": 52.34 * fCa,  # sea level high latitude prod'n minus muon
            "Sf": 56.61 * fCa,
            "Lm": 51.83 * fCa,
            "De": 55.90 * fCa,
            "Du": 55.27 * fCa,
            "Li": 60.66 * fCa,
        }
        
        if constants == 'heisinger':
            self.fstar = 0.045
            self.delfstar = 0.005
            self.sigma0 = 27.4e-30
            self.delsigma0 = 5.9e-30
        elif constants == 'braucher':
            self.fstar = 1.362e-2
            self.sigma0 = 8.262e-30
        else:
            raise NotImplementedError(
                    'Only "heisinger" constants are currently included.')


class Cl36K(Cl36):
    """
    Chlorine-36 production in K (potassium).
    """

    # map scaling scheme codes to reference spallation production rates
    scaling_p_sp_ref = {
        "Sa": 156.09,
        "St": 150.72,  # sea level high latitude prod'n minus muon
        "De": 128.25,
        "Du": 128.89,
        "Li": 142.24,
    }
    
    # factors from Heisinger 2002b rel. to slow negative muon production
    # these are for K20 so must be wrong for pure K
    # TODO: FIXME
    fC = 0.755  # chemical compound factor
    fD = 0.8020

    def __init__(self, constants='braucher'):
        self.fstar = 5.857e-2
        self.sigma0 = 9.214e-30
        raise NotImplementedError(
                'Pure K production not implemented because fC is not known')


class Cl36Kfeld(Cl36):
    """Represent the radioisotope Chlorine-36 production in K-feldspar.
    
    For pure orthoclase by default.  Can be adjusted for K fraction by
    mass (fK).
    """
    
    def __init__(self, fK=0.1405, constants='braucher'):
        # map scaling scheme codes to reference spallation production rates
        # We multiply by 0.1405, the fraction by weight of K in pure orthoclase
        self.scaling_p_sp_ref = {
            "Sa": 156.09 * fK,
            "St": 150.72 * fK,  # sea level high latitude prod'n minus muon
            "De": 128.25 * fK,
            "Du": 128.89 * fK,
            "Li": 142.24 * fK,
        }
        
        self.Natoms = 2.164e21  # atoms K / g Kfeldspar

        # probability factors
        self.fC = 0.12
        self.fD = 0.8020

        if constants == 'heisinger':
            self.fstar = 0.035
            self.delfstar = 0.005
            # TODO: still need to verify proper sigma values
            raise NotImplementedError(
                    "Heisinger 36-Cl in K-feldspar not yet implemented.")
        elif constants == 'stone':
            # John Stone, pers. communication
            self.fstar = 0.0568
            self.sigma0 = 9.40e-30
        elif constants == 'braucher':
            self.fstar = 5.857e-2
            self.sigma0 = 9.214e-30
        else:
            raise Exception('Unknown constants: %s' % constants)
        
        # for consistency calculate what sigma190 should be
        self.sigma190 = self.sigma0 * 190 ** ALPHA
        # stopped/negative muon yield
        self.k_neg = self.fC * self.fD * self.fstar
        #self.delk_neg = self.fC * self.fD * self.delfstar
        super(Cl36Kfeld, self).__init__()

    def relative_error(self, concentration):
        """Approximate relative error for concentration.

        For now, assumes same statistics as Al-26.
        """
        return 4.40538328952 * concentration ** (-0.32879674)

    def measurement_error(self, concentration):
        """Approximate measurement error for concentration.

        For now, assumes same statistics as Al-26.
        """
        return concentration * self.relative_error(concentration)

# TODO: Cl36 other pathways, Helium, neon, carbon
