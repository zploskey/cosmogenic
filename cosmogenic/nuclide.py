"""
Cosmogenic nuclide classes to hold important constants.

Nuclides in this class contain reference production rates of the cosmogenic
nuclide from spallation for each scaling scheme as a dictionary called
"scaling_p_sp_ref". Production rate values are from Borchers et al. (2014)
unless otherwise noted.

Reference:
    Borchers et al. (2014) Geological calibration of spallation production
    rates in the CRONUS-Earth Project. Quaternary Geochronology, in press.
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import math

ALPHA = 0.75  # constant from Heisinger 2002
P36KFELD_SLHL = 22.5  # atoms / g pure Kfeldspar / yr
P36K_SLHL = 160.0  # roughly, atoms / g K / yr


class Be10Qtz(object):

    """
    Data for the radioisotope Beryllium-10

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
        "Sa":  3.92,
        "St":  4.01,
        "Sf":  4.09,
        "Lm":  4.00,
        "De":  3.69,
        "Du":  3.70,
        "Li":  4.06,
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


class Al26Qtz(object):

    """
    Aluminum-26 data
    """

    # map scaling scheme codes to reference spallation production rates
    scaling_p_sp_ref = {
        "Sa":  28.54,
        "St":  27.93,  # sea level high latitude prod'n minus muon
        "Sf":  28.61,
        "Lm":  27.93,
        "De":  26.26,
        "Du":  26.29,
        "Li":  28.72,
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


class Cl36Ca(object):

    """
    Chlorine-36 in calcium.
    """
    # map scaling scheme codes to reference spallation production rates
    scaling_p_sp_ref = {
        "Sa":  56.27,
        "St":  52.34,  # sea level high latitude prod'n minus muon
        "Sf":  56.61,
        "Lm":  51.83,
        "De":  55.90,
        "Du":  55.27,
        "Li":  60.66,
    }

    def __init__(self, constants='stone'):
        raise NotImplementedError("Cl-36 in Ca is not implemented yet.")


class Cl36K(object):

    """
    Chlorine-36 production in K (potassium).
    """

    # map scaling scheme codes to reference spallation production rates
    scaling_p_sp_ref = {
        "Sa":  156.09,
        "St":  150.72,  # sea level high latitude prod'n minus muon
        "De":  128.25,
        "Du":  128.89,
        "Li":  142.24,
    }

    def __init__(self, constants='stone'):
        raise NotImplementedError("Cl-36 in Ca is not implemented yet.")


class Cl36Kfeld(object):

    """
    Data for the radioisotope Chlorine-36 produced in K-feldspar.

    This can be used to predict the production rate in pure K-feldspar.
    """

    def __init__(self, constants='stone'):
        self.Natoms = 2.164e21  # atoms K / g Kfeldspar
        self.LAMBDA = math.log(2) / 3.01e5  # decay constant, 1/yr

        self.P36_slhl = P36KFELD_SLHL

        # probability factors
        self.fC = 0.12
        self.fD = 0.8020

        if constants == 'heisinger':
            raise NotImplementedError("Heisinger 36-Cl not yet implemented.")
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

    def relative_error(self, concentration):
        """
        Approximate relative error for concentration.

        For now, assumes same statistics as Al-26.
        """
        return 4.40538328952 * concentration ** (-0.32879674)

    def measurement_error(self, concentration):
        """ Approximate measurement error for concentration.

        For now, assumes same statistics as Al-26.
        """
        return concentration * self.relative_error(concentration)

# TODO: Helium and carbon
