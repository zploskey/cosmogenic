from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from cosmogenic import sim
from cosmogenic import nuclide
from cosmogenic import production as prod
from .TestBase import TestBase

class TestSim(TestBase):

    def setUp(self):
        self.alt = 250.0
        self.lat = 65.0
        self.dz_scalar = 200.0
        self.t_gl = 15000.0
        self.t_intergl = 85000.0
        self.z = np.linspace(0, 1e4, 3)
        self.n = nuclide.Be10Qtz()
        self.p = lambda x: prod.P_tot(z=x, n=self.n, alt=self.alt, lat=self.lat)
        self.t_postgl = 10000.0 
        self.pgl_shield = 13.0
        self.n_gl = 3

    def test_simple_expose(self):
        N = sim.simple_expose(self.z, self.t_postgl, self.n, self.p)
        Nmoretime = sim.simple_expose(self.z, self.t_postgl + 1000.0, self.n,
                self.p)
        Nmoredepth = sim.simple_expose(self.z + 500.0, self.t_postgl, self.n,
                self.p)
        self.assertTrue((N < Nmoretime).all())
        self.assertTrue((N > Nmoredepth).all())
    
    def test_expose(self):
        ti = 95000.0
        tf = self.t_postgl
        N = sim.expose(self.z, ti, tf, self.n, self.p)
        Nmoretime = sim.expose(self.z, ti + 500.0, tf - 500.0, self.n, self.p)
        Nmoredepth = sim.expose(self.z + 500.0, ti, tf, self.n, self.p)
        self.assertTrue((N < Nmoretime).all())
        self.assertTrue((N > Nmoredepth).all())

    def test_multiglaciate(self):
        
        dz_scalar_low = self.dz_scalar - 15.0
        
        self.assertTrue(dz_scalar_low >= 0.0)

        N = sim.multiglaciate(self.dz_scalar, 
                              self.t_gl,
                              self.t_intergl,
                              self.t_postgl,
                              self.z,
                              self.n,
                              self.p,
                              postgl_shielding=self.pgl_shield,
                              n_gl=self.n_gl)

        N_deep = sim.multiglaciate(self.dz_scalar, 
                                   self.t_gl,
                                   self.t_intergl,
                                   self.t_postgl,
                                   self.z + 1000.0,
                                   self.n,
                                   self.p,
                                   postgl_shielding=self.pgl_shield,
                                   n_gl=self.n_gl)

        N_hi = sim.multiglaciate(self.dz_scalar - 15.0, 
                                 self.t_gl,
                                 self.t_intergl,
                                 self.t_postgl,
                                 self.z,
                                 self.n,
                                 self.p,
                                 postgl_shielding=self.pgl_shield,
                                 n_gl=self.n_gl)
        
        hi_gt_low = (N_hi > N).all()
        self.assertTrue(hi_gt_low)

        deep_lt_not = (N_deep < N).all()
        self.assertTrue(deep_lt_not)


if __name__ == "__main__":
    unittest.main()
