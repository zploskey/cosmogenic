#!/usr/bin/python

"""
Draws some curves of the N vs Z diagram at different stages in its
development.
"""


import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

import sim
import nuclide

DIRECTORY = 'test'

rho = 2.67
n = 20
z0 = np.linspace(0, 800 * rho) # sample depths
z_removed = np.ones(n) * 200 * rho # 2 m / glacial cycle
idepth = np.add.accumulate(z_removed)
tH = 17000
t_steps = np.array([0] + [20000, 80000] * n)
t = np.add.accumulate(t_steps) + tH
be10 = nuclide.Be10Qtz()
h = 1 # elevation (m)
lat = 65

# z = z0 / rho / 100

# Ns = sim.fwd_profile_steps(z0, z_removed, t, be10, h, lat)
ti = 80000
tg = 20000

# prep a figure
fig = plt.figure()
ax = fig.add_subplot(111)

lines = []
npts = 500 # number of points to use on each line
N = np.zeros(npts)
#leg = []
for i in range(1, 4):

    # begin interglacial period
    depths = np.linspace(0, z0[-1] + idepth[-i], npts)
    N *= np.exp(-be10.LAMBDA * tg) # radioactive decay
    Nexp = sim.simple_expose(depths, ti, be10, h, lat) # exposure
    N += Nexp
    # fnc = sp.interpolate.interp1d(depths, N)
    
    depthsm = depths / rho / 100.0
    ln = plt.semilogx(N, depthsm, lw=2, color='lightslategray')
#    leg.append('Ig ' + str(i))
    lines = np.append(lines, ln)

    # begin glaciation
    N *= np.exp(-be10.LAMBDA * tg) # isotopic decay
    # erode the top of the profile away
    depths -= z_removed[-i]
    nofz = interpolate.interp1d(depths, N) # interpolate conc. curve
    depths = np.linspace(0, depths[-1], 500)
    N = nofz(depths)

    depthsm = depths / rho / 100.0
    ln = plt.semilogx(N, depthsm, color='lightslategray', lw=2)
    lines = np.append(lines, ln)
#    leg.append('Gl ' + str(i))

# account for recent cosmic ray exposure
Nhol = sim.simple_expose(depths, tH, be10, h, lat) 
N *= np.exp(-be10.LAMBDA * tH)
N += Nhol
ln = plt.semilogx(N, depthsm, color='r', lw=2)
lines = np.append(lines, ln)

#leg = tuple(leg)
plt.ylim((0, 8))
plt.xlim((2000, 6e5))
ax = plt.gca()
ax.invert_yaxis()
plt.xlabel('[Be-10] (atoms / g quartz)')
plt.ylabel('Depth (m)')


def show_only(n, lines):
    """
    Helper function to set all lines but the one at index n to hidden
    """
    plt.setp(lines, alpha=0)
    plt.setp(lines[n], alpha=1)

# for i, ln in enumerate(lines):
#   show_only(i, lines)
#    filename = DIRECTORY + '/stepwise' + str(i)
#    plt.savefig(filename, transparent=True)

plt.legend((lines[-1], lines[-2]), \
    ('Current profile','Previous profiles'), \
    loc='center right')

plt.setp(lines, alpha=0) # hide all lines
for i, ln in enumerate(lines):
    plt.setp(lines[0:i], color='lightslategray', alpha=1)
#    if i == 0:
#        plt.setp(lines[i+1], color='lightslategray', alpha=1)
    plt.setp(ln, color='red', alpha=1)

#    if i == 0:
#        plt.setp(lines[i+1], alpha=0)
    filename = DIRECTORY + '/step' + str(i)
    plt.savefig(filename, transparent=False)

plt.show()

    
