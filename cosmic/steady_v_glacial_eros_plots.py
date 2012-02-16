from __future__ import division

import numpy as np

from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.font_manager 

import sim
import production
import nuclide
import muon

rho = 2.7
z_max = 1000 # cm
z = np.linspace(0, rho * z_max)
z_m = z / 100 / rho
alt = 0
lat = 75
nuc = nuclide.Be10Qtz()
t_exp = 5e6 # go for 5 million years
t = t_exp # show us the end product

# let erosion rate be 20 m / Myr
eros_rate_mMyr = 20 # m / Myr
eros_rate = eros_rate_mMyr * 1e-4 * rho # gcm2/yr

p_sp = production.P_sp(z, alt, lat, nuc)
mu = muon.P_mu_total(z, alt, nuc, full_data=True)
C_steady = sim.steady_erosion(z, p_sp, mu['P_fast'], mu['P_neg'], eros_rate,
                              nuc, t, t_exp)

t_gl = 15000
t_int = 85000
t_cycle = t_gl + t_int
n_gl = int(np.floor(t_exp / t_cycle))
dz_scalar = eros_rate * t_cycle
dz = np.ones(n_gl) * dz_scalar

# interpolate a really awesome production rate function
max_possible_depth = n_gl * dz_scalar + z_max
npts = 500 # points to interpolate through
p = production.interpolate_P_tot(max_possible_depth, npts, alt, lat, nuc)

C_gl = sim.multiglaciate(dz, t_gl, t_int, 0, z, nuc, p, n_gl=n_gl)
t_mid_postgl = 15000 # yr
C_post15k = sim.simple_expose(z, t_mid_postgl, nuc, p)

# make some figures!!
fig_height = 6 # in.
fig_width = 5 # in.
fig = Figure(figsize=(fig_width, fig_height))
fig_dpi = 500
canvas = FigureCanvas(fig)
ax = fig.add_subplot(111)
# ax.set_title('Steady v.s. glacial erosion', fontsize=14)
ax.semilogx(C_steady, z_m, 'k--', label='Steady erosion')
leg = ax.legend(loc='lower right', prop=leg_prop)
ax.invert_yaxis()
leg_prop = matplotlib.font_manager.FontProperties(size=10)
ax.set_xlabel('[$^{10}$Be] (atoms/g)', fontsize=12)
ax.set_ylabel('Depth (m)', fontsize=12)
ax.grid(linestyle='--', color='0.75')
fig.subplots_adjust(left=0.15)
canvas.print_figure('steady_eros.png', dpi=fig_dpi)
ax.semilogx(C_gl, z_m, 'b', label='Glacial erosion')
leg = ax.legend(loc='lower right', prop=leg_prop)
canvas.print_figure('steady_plus_glacial_eros.png', dpi=fig_dpi)
