import matplotlib.pyplot as plt
import numpy as np

import nuclide
import sim

rho = 2.67
n = 20
z_eroded = 500 * rho * np.ones(n) # erode 500 cm per glaciation
z0 = np.linspace(0, 800 * rho)
h = 1.0 # elevation in meters
lat = 65.0 # sample latitude in degrees

shielding = 1.0
be10 = nuclide.Be10Qtz()

tH = 17000
# 10kyr glaciation
t10 = np.add.accumulate([0] + [10000, 90000] * n) + tH
# 50kyr
t50 = np.add.accumulate([0] + [50000] * 2 * n) + tH
# 90kyr
t90 = np.add.accumulate([0] + [90000, 10000] * n) + tH

Nhol = sim.simple_expose(z0, tH, be10, h, lat)
N10  = sim.fwd_profile(z0, z_eroded, t10, be10, h, lat)
N50  = sim.fwd_profile(z0, z_eroded, t50, be10, h, lat)
N90  = sim.fwd_profile(z0, z_eroded, t90, be10, h, lat)

z = z0 / 100.0 / rho # meters

fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.semilogx(Nhol, z, 'k', lw=2)
plt.semilogx(N10, z, N50, z, N90, z, lw=2)
ax.invert_yaxis()
plt.xlabel('[Be-10] (atoms / g quartz)')
plt.ylabel('Depth (m)')
plt.legend(('Simple exposure', 'f_exp = 0.9', 'f_exp = 0.5', 'f_exp = 0.1'),
    loc='upper left')

ax2 = fig.add_subplot(121)

z_eroded_b = 100 * rho * np.ones(n)
N10b = sim.fwd_profile(z0, z_eroded_b, t10, be10, h, lat)
N50b = sim.fwd_profile(z0, z_eroded_b, t50, be10, h, lat)
N90b = sim.fwd_profile(z0, z_eroded_b, t90, be10, h, lat)

plt.semilogx(Nhol, z, 'k', lw=2)
plt.semilogx(N10b, z, N50b, z, N90b, z, lw=2)
ax2.invert_yaxis()
plt.show()
