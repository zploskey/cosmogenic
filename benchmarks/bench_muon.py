from __future__ import print_function, division

import numpy as np

from cosmogenic import nuclide
from cosmogenic import muon

n = nuclide.Be10Qtz()
z = np.linspace(0, 3000, 15)
p_mu = muon.P_mu_total(z=z, n=n)

#print(p_mu)
