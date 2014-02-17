import unittest

import numpy as np

from cosmogenic import production
from cosmogenic import nuclide
from TestBase import TestBase

class TestProduction(TestBase):

    def setUp(self):
        self.z = np.linspace(0, 2.2e5, 500)
        self.n = nuclide.Be10Qtz()
    
    def test_P_sp(self):
        pofz = production.P_sp(self.z, self.n)
        self.monotonically_decreasing(pofz)

    def test_P_tot(self):
        pofz = production.P_tot(self.z, n=self.n, alt=0.0, lat=75.0)
        self.monotonically_decreasing(pofz)


if __name__ == "__main__":
    unittest.main()
