from __future__ import print_function, division, absolute_import

import numpy as np

import cosmogenic.sim as sim
import cosmogenic.nuclide as nuclide
import cosmogenic.production as prod

def test_multiglaciate():
    alt = 250.0
    lat = 65.0
    n_cycles = 3
    dz =  np.ones(n_cycles, dtype=np.double) * 100.0
    t_gl = np.ones(n_cycles, dtype=np.double) * 1.5e4
    t_intergl = np.ones(n_cycles, dtype=np.double) * 8.5e4
    t_postgl = 1.5e4
    z = np.linspace(0, 1e3, 10)
    n = nuclide.Be10Qtz()
    p = lambda z: prod.P_tot(z, alt, lat, n)
    postgl_shielding = 13.0

    N = sim.multiglaciate(dz, t_gl, t_intergl, t_postgl, z, n, p,
                          postgl_shielding=postgl_shielding)
    
    return True

if __name__ == "__main__":
    print("Pass") if test_multiglaciate() else print("Fail")
