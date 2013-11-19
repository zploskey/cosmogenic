from __future__ import division

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import joblib

import nuclide
import sim

# get the data from the simulation loaded into memory
concs = np.genfromtxt('concs.txt')
errors = np.genfromtxt('errors.txt')
models = np.genfromtxt('models.txt')
ms = np.atleast_2d(np.genfromtxt('ms.txt'))
misfits = np.atleast_1d(np.genfromtxt('misfits.txt'))
con = joblib.load('con.dat')
p = joblib.load('production_rate.dat')

ms_denorm = ms * (con['max_dz'] - con['min_dz']) + con['min_dz']
# denormalize the models
ms_m = ms_denorm / con['rho'] / 100.0

dvt_len = 2 * (con['n_gl'] + 1)
fit_t = np.zeros((misfits.size, dvt_len))
fit_z = np.empty((misfits.size, dvt_len))
for i in range(misfits.size):
    fit_t[i, :], fit_z[i, :] = sim.glacial_depth_v_time(con['t_gl'], con['t_int'],
                                                con['t_postgl'], ms_m[i], 
                                                n_gl=con['n_gl'])

#############################
# PLOTTING                  #
#############################

# zoft plot including all the close exposure histories
many_zoft_fig = plt.figure()
many_zoft_ax = many_zoft_fig.add_subplot(111)
# make the brightness of each curve dependent on how low its error was
alphas = 1 - (misfits / con['dof'])
#alpha = 0.5
for i in range(misfits.size):
    plt.plot(fit_t[i] / 1e6, fit_z[i], 'k', alpha=alphas[i])
var_line, = many_zoft_ax.plot(con['t'] * 1e-6, con['z_targ'], 'r')
many_zoft_ax.invert_yaxis()
many_zoft_ax.invert_xaxis()
plt.xlabel('Myr B.P.')
plt.ylabel('Depth (m)')
plt.title('Erosion Histories')
plt.show()

figname = 'best_histories'
plt.savefig(figname + '.svg')
plt.savefig(figname + '.png')

###########################################
# CONCENTRATION PLOTS                     #
###########################################

# convert depth to meters to be used below
z_m = con['sample_depths_m']

# measured
conc_fig = plt.figure()
conc_ax = conc_fig.add_subplot(111)
conc_ax.set_xscale('log')
plt.errorbar(con['C_meas'], z_m,
             xerr=con['C_meas_err'],
             fmt='.',
             markersize=8,
             ecolor='k',
             label='Measured')
conc_ax.invert_yaxis()
plt.legend(loc='lower right')
plt.xlabel('$^{10}$Be Concentration (atoms / g)')
plt.ylabel('Depth (m)')
plt.title('$^{10}$Be Depth Profile')
plt.show()

figname = 'measured_conc'
plt.savefig(figname + '.svg')
plt.savefig(figname + '.png')

# concentration profile with best fit
min_misfit = np.min(misfits)
best_m = ms_denorm[misfits == min_misfit][0]
many_depths = np.linspace(0, con['sample_depths'][-1], 200)
best_conc_manypts = sim.multiglaciate(best_m, con['t_gl'], con['t_int'], 
                    con['t_postgl'], many_depths, con['nuclide'], p, 
                    n_gl=con['n_gl'], postgl_shielding=con['postgl_shielding'])
many_depths_m = many_depths / 100 / con['rho']
plt.semilogx(best_conc_manypts, many_depths_m, '-b', 
             label='Best Fit (ChiSquared = %0.2f)' % min_misfit)
plt.legend(loc='lower right')
plt.show()

figname = 'best_conc'
plt.savefig(figname + '.svg')
plt.savefig(figname + '.png')
