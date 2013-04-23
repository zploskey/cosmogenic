import numpy as np
import matplotlib.pyplot as plt

import cosmogenic.nuclide as nuclide
import cosmogenic.sim as sim
import cosmogenic.production as production

nuc = nuclide.Be10Qtz()
p = lambda z: production.P_tot(z, 0.0, 75.0, nuc)

z = np.linspace(0, 1300, 20)
eros = 2.0e-13
t_exp = 1e6

N = sim.steady_erosion(p, z, eros, nuc, t_exp)

plt.plot(z, N, 'o')
plt.xlabel('Depth (g/cm2)')
plt.ylabel('Be-10 Concentration (atoms/g)')
plt.show()


