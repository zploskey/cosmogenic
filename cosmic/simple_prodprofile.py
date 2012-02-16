#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import nuclide
import production

rho = 2.67
h = 220 # elevation in meters
lat = 44.544 # sample latitude in degrees
z0 = np.linspace(0, 1000 * rho)
be10 = nuclide.Be10Qtz()
Ptot = production.P_tot(z0, h, lat, be10)
z = z0 / rho / 100.0

fig = plt.figure(figsize=(4.5,7))
ax = fig.add_subplot(111)
ax.semilogx(Ptot, z, 'b', lw=2)
ax.invert_yaxis()
ax.set_xlabel(r'$^{10}$Be Production Rate (atoms / g quartz / yr)', fontsize=13)
ax.set_ylabel('Depth (m)', fontsize=13)
ax.set_title(r'$^{10}$Be Production Rate Profile', fontsize=16)
ax.set_xlim((0.01, 10))
plt.show()
plt.savefig('jay_prod_profile.png')
