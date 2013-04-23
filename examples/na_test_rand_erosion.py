from __future__ import division

import numpy as np
import joblib
import matplotlib as mpl
import matplotlib.font_manager 

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import sim
import nuclide
import na
import production

constraints = {
    'n_samples':      10,
    'bottom_depth_m': 9.5,
    'alt':            220,   # surface elevation, assumed constant, m
    'lat':            44.544,
    'rho':            2.67,  # g/cm2
    # define max allowable parameter values to normalize to
    'max_dz_m':       12,    # max m of rock eroded each glaciation
    'min_dz_m':       0.01,  # m
    't_gl':           15000,
    't_int':          85000,
    't_postgl':       15500,
    'postgl_shielding': 80, # postglacial shielding correction (for snow/till cover)
    'nuclide':        nuclide.Be10Qtz(),
    'n_gl':           20,     # number of glaciations
    # define our parameters for the Neighborhood Algorithm
    'ns':             100, # number of samples each iteration
    'nr':             50,  # number of voronoi cells that we explore in each iteration
    'ensemble_size':  100,
    'n_prod_interp_pts': 500,
}
con = constraints
con['bottom_depth'] = con['bottom_depth_m'] * 100 * con['rho']
con['max_dz'] = con['max_dz_m'] * 100 * con['rho']
con['min_dz'] = con['min_dz_m'] * 100 * con['rho']
# logarithmically place our samples
con['sample_depths_m'] = np.logspace(0, np.log(con['bottom_depth_m'] + 1), con['n_samples'], base=np.e) - 1
con['sample_depths'] = con['sample_depths_m'] * 100 * con['rho']

# Construct the target erosion history
# made using sim.rand_erosion_hist(6, 3, 20)

con['dz_m'] = np.array([  9,   2,   4,  2,
        0.2,   0.3,   0.1,   0.3,
        0.2,   3.4,   8 ,   7,
        2,   3.6,   8.3,   6,
        5 ,   1.8,  7.5,   4])
con['dz'] = con['dz_m'] * 100 * con['rho']
avg_dz_m = con['dz_m'].mean()

t, z_targ = sim.depth_v_time(con['t_gl'], con['t_int'], con['t_postgl'], 
                             con['dz_m'], n_gl=con['n_gl'])
_, z_const = sim.depth_v_time(con['t_gl'], con['t_int'], con['t_postgl'], 
                              avg_dz_m, n_gl=con['n_gl'])

fig_height = 3.5 # in.
fig_width = 5 # in.

fig = Figure(figsize=(fig_width, fig_height))
canvas = FigureCanvas(fig)
ax = fig.add_subplot(111)
ax.set_title('Varied Erosion History', fontsize=12)
ax.invert_xaxis()
ax.invert_yaxis()
var_line, = ax.plot(t * 1e-6, z_targ, 'r')
ax.set_xlabel('Time Before Present (Myr)', fontsize=10)
ax.set_ylabel('Depth (m)', fontsize=10)
ax.grid(linestyle='-', color='0.75')
fig.subplots_adjust(bottom=0.13)
ax.set_xlim(left=2)
ax.set_ylim(bottom=75)
canvas.print_figure('var_hist.png', dpi=500)
ax.set_title('Steady Erosion History', fontsize=12)
const_line, = ax.plot(t * 1e-6, z_const, 'b')
ax.lines.remove(var_line)
canvas.print_figure('const_hist.png', dpi=500)

fig_height = 5 # in.
fig_width = 3.5 # in.

# interpolate a production function
max_possible_depth = con['n_gl'] * con['max_dz'] + con['bottom_depth']
p = production.interpolate_P_tot(max_possible_depth, 
                                 npts=con['n_prod_interp_pts'], alt=con['alt'],
                                 lat=con['lat'], n=con['nuclide'])

# make target concentration graph
z_target = np.linspace(0, con['bottom_depth'])

C_target = sim.multiglaciate(con['dz'], con['t_gl'], con['t_int'], 
                             con['t_postgl'], z_target, con['nuclide'], p,
                             con['n_gl'], 
                             postgl_shielding=con['postgl_shielding'])

conc_fig = Figure(figsize=(fig_width, fig_height))
conc_canvas = FigureCanvas(conc_fig)
ax = conc_fig.add_subplot(111)
ax.set_title('Target Concentration Profile', fontsize=12)
ax.semilogx(C_target, z_target / con['rho'] / 100)
ax.invert_yaxis()
ax.set_xlabel('[$^{10}$Be] (atoms/g)', fontsize=10)
ax.set_ylabel('Depth (m)', fontsize=10)
conc_fig.subplots_adjust(left=0.15, top=0.95)
conc_canvas.print_figure('target_conc.png', dpi=500)

C_true = sim.multiglaciate(con['dz'], con['t_gl'], con['t_int'], 
                             con['t_postgl'], con['sample_depths'], 
                             con['nuclide'], p, con['n_gl'], 
                             postgl_shielding=con['postgl_shielding'])
sigma = con['nuclide'].measurement_error(C_true)
C_meas = np.random.normal(loc=C_true, scale=sigma)
C_meas_err = con['nuclide'].measurement_error(C_meas)
con['C_meas'] = C_meas
con['C_meas_err'] = C_meas_err

meas_fig = Figure(figsize=(fig_width, fig_height))
meas_canvas = FigureCanvas(meas_fig)
ax = meas_fig.add_subplot(111)
ax.set_title('Synthetic Concentration Measurements', y=1.03, fontsize=12)
ax.errorbar(C_meas, con['sample_depths_m'], xerr=C_meas_err, fmt='k.')
ax.invert_yaxis()
ax.set_xlabel('[$^{10}$Be] (atoms/g)', fontsize=10)
ax.set_ylabel('Depth (m)', fontsize=10)
ax.set_xscale('log')
meas_fig.subplots_adjust(left=0.15) #, top=0.95
meas_canvas.print_figure('meas_conc.png', dpi=500)


