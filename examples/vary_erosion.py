from __future__ import division

import numpy as np
import matplotlib as mpl
import matplotlib.font_manager 

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import sim
import production
import nuclide
import muon

rho = 2.67
z_max_m = 10
z_max = z_max_m * 100 * rho # g / cm**2
z = np.linspace(0, z_max)
z_m = z / 100 / rho
alt = 220
lat = 44.544
nuc = nuclide.Be10Qtz()
t_exp = 2e6 # go for 10 million years

# let erosion rate be 20 m / Myr
eros_rates_mMyr = np.array([10, 30, 60, 100, 200]) # m / Myr
eros_rates = eros_rates_mMyr * 1e-4 * rho # gcm2/yr

t_gl = 15000
t_int = 85000
t_postgl = 15500
postgl_shielding = 85 # g / cm**2
t_cycle = t_gl + t_int
n_gl = int(np.floor(t_exp / t_cycle))
dz_scalar = eros_rates * t_cycle

# interpolate a really awesome production rate function
max_possible_depth = n_gl * np.max(dz_scalar) + z_max
p = production.interpolate_P_tot(max_possible_depth, npts=500, alt=alt, lat=lat,
                                 n=nuc)

# postglacial isotope production
C_postgl = sim.simple_expose(z + postgl_shielding, t_postgl, nuc, p)

# production during glacial erosion plus postglacial accum.
C = np.zeros((eros_rates.size, z.size))
for i in range(eros_rates.size):
    C[i] = sim.multiglaciate(dz_scalar[i], t_gl, t_int, t_postgl, z, nuc, p, 
                             n_gl, postgl_shielding)

fig_height = 6 # in.
fig_width = 4.5 # in.

fig = Figure(figsize=(fig_width, fig_height))
canvas = FigureCanvas(fig)
ax = fig.add_subplot(111)
ax.set_title('Effect of erosion depth', fontsize=14)
ax.plot(C_postgl, z_m, 'b', label='15.5 kyr')
ax.set_xscale('log')
for i in range(eros_rates.size):
    ax.semilogx(C[i], z_m, 'r--', label=str(eros_rates_mMyr[i]) + ' m')

ax.invert_yaxis()
#leg_prop = matplotlib.font_manager.FontProperties(size=10) 
#leg = ax.legend(loc='lower right', prop=leg_prop)
ax.set_xlabel('[$^{10}$Be] (atoms/g)', fontsize=12)
ax.set_ylabel('Depth (m)', fontsize=12)
ax.grid(linestyle='-', color='0.75')
fig.subplots_adjust(left=0.15)
canvas.print_figure('vary_erosion.png', dpi=500)

# put on our measurements
z_samp_m = np.array([
1,
14.25,
19,
34,
113,
208,
367]) / 100.0

C_meas = np.array([
57287.0,
47274.1,
42731.2,
32661.6,
12838.1,
8803.6,
5615.4])

#ax.errorbar(C_meas, z_samp_m, xerr=nuc.measurement_error(C_meas), fmt='k.')
ax.plot(C_meas, z_samp_m, 'k.')
canvas.print_figure('vary_eros_w_meq_data.png', dpi=500)
canvas.print_figure('vary_eros_w_meq_data.eps', dpi=500)

