"""
====================================
PARMA (:mod:`cosmogenic.parma`)
====================================

.. currentmodule:: cosmogenic.parma

Implementation of PARMA, a set of analytical function fit to PHITS,
a Monte Carlo simulation of the cosmic ray cascade of particles through
the atmosphere.

Reference:
Tatsuhiko Sato, Hiroshi Yasuda, Koji Niita, Akira Endo, and Lembit Sihver.
(2008) Development of PARMA: PHITS-based Analytical Radiation Model in the
Atmosphere. Radiation Research, 170(2), 244--259.

    Sato 2008
"""
from __future__ import division, print_function, unicode_literals

cimport numpy as np
import numpy as np


class Particle(object):
    
    def __init__(self):
        pass


class PrimaryParticle(Particle):
    
    def __init__(self):
        super(PrimaryParticle, self).__init__()
        self.a = None
    
    def flux(self, s, d, E):
        """
        s: force field potential
        d: atmospheric depth
        E: kinetic energy per nucleon
        """
        a = self.a
        flux = self.flux_TOA(s, E + a[0]) * (a[1] * np.exp(-a[2] * d) 
                                        + (1 - a[1]) * np.exp(-a[3] * d))
        
        return flux
        
    def flux_TOA(self, s, E):
        a = self.a
        E_LIS = E + s * self.Z / self.A
        f = self.C(E_LIS) * Beta(E_LIS) ** a[4]
        f /= R(E_LIS) ** a[5]
        f *= (R(E) / R(E_LIS)) ** 2
        return f

    def C(self, E):
        a = self.a
        C = a[6] + a[7] / (1 - np.exp((E - a[8]) / a[9]))
        return C


class Proton(PrimaryParticle):
    a = np.array([
        2.12,  # cm2 g-1 MeV / nucleon
        0.445,
        0.0101,  # cm2 g-1
        0.396,  # cm2 g-1
        2.924,  
        2.708,
        1.27e4,  # s-1 m-2 sr-1 GV-1
        4.83e3,
        3.28e4,
        7.44e3,
        3.46,
        1.68,
        1.37,
        2.07,
        108,
        2.3e3])
    
    Z = 1
    
    def __init__(self):
        super(Proton, self).__init__()

class Alpha(PrimaryParticle):
    a = np.array([
        17.6,
        0.438,
        0.0121,
        0.0434,
        1.841,
        2.646,
        2.36e3,
        432,
        6.06e3,
        2.41e3,
        3.33,
        11.7,
        0.967,
        3.2,
        15.0,
        853])
        
    Z = 2
    
    def __init__(self):
        super(Alpha, self).__init__()

class Muon(Particle):
    
    u_pmu = np.array([6.26e9, 0.00343, 1.01, 0.00418, 3.75e8])
    u_nmu = np.array([5.82e9, 0.00362, 1.02, 0.00451, 3.20e8])

    def __init__(self):
        super(Muon, self).__init__()
    
    def flux(self, d, charge='-'):
        """
        Negative or positive muon flux at atmospheric depth d.

        If charge is not specified, the default is negative.
        """
        if charge == '-':
            u = self.u_nmu
        elif charge == '+': 
            u = self.u_pmu
        else:
            raise ValueError("Unknown argument '%s'" % charge)
        
        return u[0] * (np.exp(-u[1] * d) - u[2] * np.exp(-u[3] * d)) + u[4]


def R(energy):
    raise NotImplementedError()


def Beta(energy):
    raise NotImplementedError()
    


