#!/usr/bin/python

"""
Draws some curves of the N vs Z diagram at different stages in its
development.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline as USpline

import sim
import nuclide
import production

DIRECTORY = 'test'

rho = 2.7
n = 4
npts = 200
z0 = np.linspace(0, 800 * rho, npts) # sample depths
z_removed = np.ones(n) * 200 * rho # 2 m / glacial cycle
idepth = np.add.accumulate(z_removed)
tH = 15500
ti = 85000
tg = 15000
t_steps = np.array([0] + [tg, ti] * n)
t = np.add.accumulate(t_steps) + tH
be10 = nuclide.Be10Qtz()
L = be10.LAMBDA
alt = 220 # elevation (m)
lat = 44.54

z = z0 / rho / 100

lines = []
leg = []

N = np.zeros(z0.size)
Ns = np.zeros((n + 1, z0.size))
# zeroth glaciation
# deal with any previous nuclides
N *= np.exp(-L * tg)
for i in range(z_removed.size):
    depths = np.linspace(0, z0[-1] + idepth[-i], npts)
    
    # begin interglacial period
    N *= np.exp(-L * ti)
    Nexp = sim.simple_expose_slow(depths, ti, be10, alt, lat)
    N += Nexp

    depthsm = depths / rho / 100.0
    Ns[i] = N

    # begin glaciation
    N *= np.exp(-L * tg) # isotopic decay
    # erode the top of the profile away
    depths -= z_removed[-i]
    nofz = USpline(depths, N, k=3, s=0)
    depths = np.linspace(0, depths[-1], npts)
    N = nofz(depths)

    #depthsm = depths / rho / 100.0

# account for recent cosmic ray exposure
N *= np.exp(-L * tH)
Nhol = sim.simple_expose_slow(depths, tH, be10, alt, lat) 
N += Nhol
Ns[-1] = N

fig_height = 5 # in.
fig_width = 3.5 # in.

fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.add_subplot(111)
ax.set_xlim((1000, 1e6))
ax.set_title('Concentration Profile', fontsize=12)
ax.set_xlabel('[$^{10}$Be] (atoms/g)', fontsize=10)
ax.set_ylabel('Depth (m)', fontsize=10)
#canvas = plt.FigureCanvas(conc_fig)

for i in range(n+1):
    if i != 0:
        prev_line = ax.lines[-1]
        prev_line.set_color('lightslategray')
    ax.semilogx(Ns[i], z0 / rho / 100, 'b')
    if i == 0:
        ax.invert_yaxis()
    fig.subplots_adjust(top=0.95, bottom=0.13)
    filename = DIRECTORY + '/step' + str(i)
    plt.savefig(filename + '.png', transparent=False, dpi=500)
#plt.show()
#canvas.print_figure('target_conc.png', dpi=500)

#for i in range(1, n):

#    # begin interglacial period
#    depths = np.linspace(0, z0[-1] + idepth[-i], npts)
#    N *= np.exp(-be10.LAMBDA * tg) # radioactive decay
#    Nexp = sim.simple_expose(depths, ti, be10, p)
#    N += Nexp
#    # fnc = sp.interpolate.interp1d(depths, N)
#    
#    depthsm = depths / rho / 100.0
#    ln = plt.semilogx(N, depthsm, lw=2, color='lightslategray')
##    leg.append('Ig ' + str(i))
#    lines = np.append(lines, ln)

#    # begin glaciation
#    N *= np.exp(-be10.LAMBDA * tg) # isotopic decay
#    # erode the top of the profile away
#    depths -= z_removed[-i]
#    nofz = interpolate.interp1d(depths, N) # interpolate conc. curve
#    depths = np.linspace(0, depths[-1], 500)
#    N = nofz(depths)

#    depthsm = depths / rho / 100.0
#    ln = plt.semilogx(N, depthsm, color='lightslategray', lw=2)
#    lines = np.append(lines, ln)
##    leg.append('Gl ' + str(i))

## account for recent cosmic ray exposure
#Nhol = sim.simple_expose(depths, tH, be10, alt, lat) 
#N *= np.exp(-be10.LAMBDA * tH)
#N += Nhol
#ln = plt.semilogx(N, depthsm, color='r', lw=2)
#lines = np.append(lines, ln)

    
