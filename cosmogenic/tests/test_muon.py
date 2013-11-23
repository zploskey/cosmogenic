from __future__ import print_function, division

import unittest

import numpy as np

import cosmogenic.muon as muon
import cosmogenic.nuclide as nuclide
from cosmogenic.tests.TestBase import TestBase

class TestMuon(TestBase):

    def setUp(self):
        self.z = np.linspace(0, 1.1775e6, 10)
        self.n = nuclide.Be10Qtz()
        self.alt = 0.0

    def test_phi_sl(self):
        res = muon.phi_sl(self.z)
        self.assertTrue(res.size == self.z.size)
        self.decreases_with_depth(res)

    def test_P_mu_total(self):
        res = muon.P_mu_total(self.z, self.alt, self.n)
        self.assertTrue(res.size == self.z.size)
        self.decreases_with_depth(res)

if __name__ == "__main__":
    unittest.main()
