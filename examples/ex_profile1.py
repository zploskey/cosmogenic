#!/usr/bin/python

import time

import numpy as np
import matplotlib.pyplot as plt

import nuclide
import sim

# s = sample.Sample(rho=2.67, h=1000, lat=65, shielding=1.0, z=1)
# print "Total muon flux: %f " % muon.tot_mu_flux(s)
rho = 2.67
start = time.time()

z0 = np.linspace(0, 800 * rho)
h = 1.0 # elevation in meters
lat = 65.0 # sample latitude in degrees
# lat = array([54,32,34,32,12,45,65,76,43,78]) # latitude array for testing
shielding = 1.0
be10 = nuclide.Be10Qtz()

# get production curves first
#Psp = production.P_sp(z0, be10, 'stone', h, lat)
#muondata = muon.P_mu_total(z0, h, be10, full_data=True)
#Pfmu = muondata['P_fast']
#Pnmu = muondata['P_neg']
#Pmu = Pfmu + Pnmu
#Ptot = Psp + Pmu

# z_removed = 267 * ones(nt / 2)
# t = np.linspace(0,1e6,33)
z_removed = np.genfromtxt("erosion_depths.txt") # in meters
z_removed *= 100 * rho # in g/cm^2
t = np.add.accumulate(np.genfromtxt("t_periods.txt")) * 1000.0

N = sim.fwd_profile(z0, z_removed, t, be10, h, lat)
Nhol = sim.simple_expose(z0, t_exp=t[0], n=be10, h=h, lat=lat)
sat = sim.simple_expose(z0, t_exp=1e8, n=be10, h=h, lat=lat)
same = sim.simple_expose(z0, t_exp=t[-1], n=be10, h=h, lat=lat)

end = time.time()
elapsed = end - start
print "Calculations took", elapsed, "seconds to run."

print "N =", N
z = z0 / (rho * 100.0)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.semilogx(Nhol, z, 'r--', same, z, 'g:', N, z, 'k', sat, z, 'b-.', lw=2)
leg = ax.legend(('Simple exposure', \
                 'Continuous exposure', \
                 'Periodic glaciation', \
                 'Saturated profile'), 'upper left', shadow=True)

# matplotlib.text.Text instances
for txt in leg.get_texts():
    txt.set_fontsize('small')

ax.invert_yaxis()
plt.xlabel('Beryllium-10 Concentration [atoms / g quartz]')
plt.ylabel('Depth [m]')
plt.subplots_adjust(bottom=0.1)
#plt.semilogx(Ptot, z, 'b', Psp, z, 'g--',  Pmu, z, 'k-.')
#plt.xlabel('Beryllium-10 Production Rate [atoms / g quartz / yr]')

#plt.subplots_adjust(bottom=0.15)
plt.show()
