#!/usr/bin/python

import time

import numpy as np
from numpy import array, arange, linspace, ones
import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib import rc

import nuclide
import sim
import production
import muon

# s = sample.Sample(rho=2.67, h=1000, lat=65, shielding=1.0, z=1)
# print "Total muon flux: %f " % muon.tot_mu_flux(s)
rho = 2.67
h = 1 # elevation in meters
lat = 65 # sample latitude in degrees
z0 = linspace(0, 800 * rho)
shielding = 1.0
be10 = nuclide.Be10Qtz()

# get production curves first
Psp = production.P_sp(z0, h, lat, be10)
muondata = muon.P_mu_total(z0, h, be10, full_data=True)
Pfmu = muondata['P_fast']
Pnmu = muondata['P_neg']
Pmu = Pfmu + Pnmu
Ptot = Psp + Pmu

z = z0 / rho / 100.0
plt.semilogx(Psp, z, 'b', Pmu, z, 'burlywood',  Ptot, z, 'r--', lw=2)
ax = plt.gca()
ax.invert_yaxis()
plt.xlabel('Be-10 Production Rate (atoms / g quartz / yr)')
plt.ylabel('Depth (m)')
plt.legend(('Spallation', 'Muons', 'Total'), loc='lower right')
plt.xlim((10e-4, 10))
mpl.axis.YAxis(ax).set_label_position('right')
# plt.subplots_adjust(bottom=0.15)
plt.show()
