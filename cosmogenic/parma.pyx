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

"""
from __future__ import division, print_function, unicode_literals

cimport numpy as np
import numpy as np

from scipy.constants import physical_constants as const


class Particle(object):

    # mass of the particle
    A = None

    # atomic number
    Z = None


class PrimaryParticle(Particle):

    a = None

    # particle rest mass, MeV
    Em = None

    def __init__(self):
        super(PrimaryParticle, self).__init__()

    def flux_pri(self, s, d, E):
        """
        Primary particle flux at depth d in the atmosphere.
        Units: cm-2 s-1 (MeV/Nucleon)-1
        Sato 2008, eq. 1.
        s: force field potential
        d: atmospheric depth
        E: kinetic energy per nucleon
        """
        a = self.a
        flux = self.flux_TOA(s, E + a[0] * d) * (a[1] * np.exp(
            -a[2] * d) + (1 - a[1]) * np.exp(-a[3] * d))
        return flux

    def flux_TOA(self, s, E):
        """
        Primary particle flux at the top of the atmosphere.
        Sato 2008, eq. 2
        Units: cm-2 s-1 (MeV/nucleon)-1
        """
        a = self.a
        R = self.R
        E_LIS = E + s * self.Z / self.A
        f = self.C(E_LIS) * self.Beta(E_LIS) ** a[4]
        f /= R(E_LIS) ** a[5]
        f *= (R(E) / R(E_LIS)) ** 2
        # convert to cm-2 s-1 (MeV/nucleon)-1 from s-1 m-2 sr-1 GV-1
        f *= 1.675 * np.pi
        return f

    def C(self, E):
        """
        s-1 m-2 sr-1 GV-1
        """
        a = self.a
        C = a[6] + a[7] / (1 - np.exp((E - a[8]) / a[9]))
        return C

    def R(self, E):
        """
        Partical rigidity in GV, as a function of energy E in MeV/nucleon.
        Sato 2008, p. 249
        """
        R = 0.001
        R *= np.sqrt((self.A * E) ** 2 + 2 * self.A * self.Em * E) / self.Z
        return R

    def Beta(self, E):
        """
        Speed of particle relative to light.
        Beta = v / c
        E: kinetic energy in MeV / nucleon
        """
        return np.sqrt(1 - np.sqrt(self.Em / E))

    def f1(self, rc, d):
        raise NotImplementedError()

    def f2(self, rc, d):
        raise NotImplementedError()

    def f3(self, rc, d):
        raise NotImplementedError()

    def Phi_N(self, s, rc, d):
        return self.f1(rc, d) + self.f2(rc, d) * s ** self.f3(rc, d)

    def b(self, i, d):
        return np.polyval(self.bc[i, ::-1], d)

    def flux_sec(self, s, rc, d, E):
        PhiN = self.Phi_N(s, rc, d)
        flux_sec = PhiN * self.b(0, d) * E ** self.b(
            1, d) / (1 + self.b(2, d) * E ** self.b(3, d))
        return flux_sec

    def Ec(self, rc):
        num = (np.sqrt((1000 * rc * self.Z) ** 2 + self.Em ** 2) - self.Em
        return num / self.A

    def Es(self, rc, d):
        a = self.a
        return a[12] * (self.Ec(rc) - a[13] * d)

    def Es1(self, rc, d):
        a = self.a
        res = self.Es(rc, d)
        an = a[14]
        res[res < an] = an
        return res

    def Es2(self, rc, d):
        a = self.a
        res = self.Es(rc, d)
        an = a[15]
        res[res < an] = an
        return res

    def flux(self, s, rc, d, E):
        """
        Total flux
        """
        a = self.a
        flux = self.flux_pri(s, d, E)
        flux *= (np.tanh(a[10] * (E / self.Es1(rc, d) - 1)) + 1) / 2
        flux += self.flux_sec(s, rc, d, E) * (
            np.tanh(a[11] * (1 - E / self.Es2(rc, d)))) / 2


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

    bc = np.array(
        [[1.26, 0.00323, 4.81e-6, 2.28e-9],
         [0.438, -5.58e-4, 7.84e-7, -3.87e-10],
         [1.81e-4, -5.18e-7, 7.59e-10, -3.82e-13],
         [1.71, 7.16e-4, -9.32e-7, 5.27e-10]])

    A = 1  # mass
    Z = 1
    Em = const['proton mass energy equivalent in MeV'][0]

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

    bc = np.array(
        [[1.00, 0, 0, 0],
         [0.881, 0, 0, 0],
         [1.8e-4, 0, 0, 0],
         [4.77, 0, 0, 0]])

    A = 4
    Z = 2
    Em = const['alpha particle mass energy equivalent in MeV'][0]

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
