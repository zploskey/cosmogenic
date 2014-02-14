from __future__ import print_function, division

import unittest

import numpy as np

from cosmogenic import muon
from cosmogenic import nuclide
from cosmogenic.tests.TestBase import TestBase

class TestMuon(TestBase):

    def setUp(self):
        self.z = np.linspace(0, 1.1775e6, 10)
        self.n = nuclide.Be10Qtz()
        self.alt = 0.0

    def test_phi_sl(self):
        res = muon.phi_sl(self.z)
        self.assertTrue(res.size == self.z.size)
        self.monotonically_decreasing(res)

    def test_P_mu_total(self):
        res = muon.P_mu_total(z=self.z, n=self.n, h=self.alt)
        self.assertTrue(res.size == self.z.size)
        self.monotonically_decreasing(res)

if __name__ == "__main__":
    unittest.main()
