
from cosmogenic import nuclide, muon

n = nuclide.Be10Qtz()
z = np.linspace(0, 3000, 15)
p_mu = muon.P_mu_total(z=z, n=n)
