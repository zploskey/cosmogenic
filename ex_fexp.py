import matplotlib.pyplot as plt
import numpy as np

import nuclide
import production
import sim

rho = 2.67
h = 1.0 # elevation in meters
lat = 65.0 # sample latitude in degrees
n = 20 # number of glaciations
z0 = np.linspace(0, 800 * rho)
be10 = nuclide.Be10Qtz()

glen = np.arange(10000,90000,20000) # meter depths eroded per glaciation
z_eroded = np.array(300 * rho * np.ones(n))

tH = 17000.0

for i,  in enumerate(glen):
    t[:,i] = np.add.accumulate([0] + [20000, 80000] * n) + tH

Nhol = sim.simple_expose(z0, tH, be10, h, lat)
N = np.zeros( (len(depths), len(z0)) )
for i, zm in enumerate(depths):
    N[i,:] = sim.fwd_profile(z0, z_eroded[:,i], t, be10, h, lat)

z = z0 / 100.0 / rho

fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogx(Nhol, z, 'r--', lw=2)
leg = ['17 kyr']
for i, zm in enumerate(depths):
    plt.semilogx(N[i,:], z, lw=2)
    leg.append(str(zm) + '0 m/Myr')
ax.invert_yaxis()
ax.set_xlim((900,3e5))
plt.xlabel('[Be-10] (atoms / g quartz)')
plt.ylabel('Depth (m)')
plt.legend(tuple(leg), loc='lower right')
plt.show()
