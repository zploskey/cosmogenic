import numpy as np
import nose

import cosmogenic.muon as muon
import cosmogenic.nuclide as nuclide

def test_phi_sl():
    z = np.linspace(0, 15000.0, 10)
    res = muon.phi_sl(z)
    assert res.size == z.size, "Input and output sizes are inconsistent"
    pass

def test_P_mu_total():
    be10 = nuclide.Be10Qtz()
    z = np.linspace(0, 1000, 20)
    res = muon.P_mu_total(z, 126.0, be10)
    print res
    assert res.size == z.size, "Input and output sizes are inconsistent"
    pass
