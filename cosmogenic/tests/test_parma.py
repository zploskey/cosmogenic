import unittest

import numpy as np

from cosmogenic import parma
from TestBase import TestBase


class TestParma(TestBase):

    def setUp(self):
        self.proton = parma.Proton()
        self.alpha = parma.Alpha()

        # force field potential, MV
        self.s = 1200
        
        # atmospheric depths g/cm2
        self.depths = np.linspace(0, 1000, 20)

        # energy in MeV
        self.E = 1000
       
    def test_proton_flux(self):
        pf = self.proton.flux(self.s, self.depths, self.E)
        print pf
        self.assertTrue(pf is not None)


    def test_flux(self):
        af = self.alpha.flux(self.s, self.depths, self.E)
        print af
        self.assertTrue(af is not None)


if __name__ == "__main__":
    unittest.main()
