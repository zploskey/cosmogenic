from __future__ import division

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import joblib

import nuclide
import sim

n = nuclide.Be10Qtz()

# get the data from the simulation loaded into memory
concs = np.genfromtxt('concs.txt')
errors = np.genfromtxt('errors.txt')
models = np.genfromtxt('models.txt')
conc_meas = np.genfromtxt('conc_meas.txt')
conc_meas_err = n.measurement_error(conc_meas)
ms = np.atleast_2d(np.genfromtxt('ms.txt'))
misfits = np.atleast_1d(np.genfromtxt('misfits.txt'))
constraints = joblib.load('constraints.dat')
con = constraints
p = joblib.load('production_rate.dat')
dof = con['n_gl']
ms_denorm = ms * (con['max_dz'] - con['min_dz']) + con['min_dz']
# denormalize the models
ms_m = ms_denorm / con['rho'] / 100.0

dvt_len = 2 * (con['n_gl'] + 1)
fit_t = np.zeros((misfits.size, dvt_len))
fit_z = np.empty((misfits.size, dvt_len))
for i in range(misfits.size):
    fit_t[i, :], fit_z[i, :] = sim.depth_v_time(con['t_gl'], con['t_int'],
                                                con['t_postgl'], ms_m[i], 
                                                n_gl=con['n_gl'])

#############################
# PLOTTING                  #
#############################

# zoft plot including all the close exposure histories
many_zoft_fig = plt.figure()
many_zoft_ax = many_zoft_fig.add_subplot(111)
# make the brightness of each curve dependent on how low its error was
#alphas = 1 - (misfits / dof)
alpha = 0.02
for i in range(misfits.size):
    plt.plot(fit_t[i] / 1e6, fit_z[i], 'k', alpha=alpha)
plt.legend(loc='lower right')
many_zoft_ax.invert_yaxis()
many_zoft_ax.invert_xaxis()
plt.xlabel('Myr B.P.')
plt.ylabel('Depth (m)')
plt.title('Past depth of the surface sample')
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
plt.errorbar(conc_meas, z_m,
             xerr=conc_meas_err,
             fmt='.',
             markersize=8,
             ecolor='k',
             label='Measured')
conc_ax.invert_yaxis()
plt.legend(loc='lower right')
plt.xlabel('$^{10}$Be Concentration (atoms / g)')
plt.ylabel('Depth (m)')
plt.title('Jay Quarry $^{10}$Be Concentrations')
plt.show()

figname = 'measured_conc'
plt.savefig(figname + '.svg')
plt.savefig(figname + '.png')

# concentration profile with best fit
min_misfit = np.min(misfits)
best_m = ms_denorm[misfits == min_misfit][0]
#best_conc = sim.multiglaciate(best_m, con['t_gl'], con['t_int'], con['t_postgl'],
#                con['sample_depths'], con['nuclide'], p, n_gl=con['n_gl'], 
#                postgl_shielding=con['postgl_shielding'])
many_depths = np.linspace(0, con['sample_depths'][-1], 200)
best_conc_manypts = sim.multiglaciate(best_m, con['t_gl'], con['t_int'], 
                    con['t_postgl'], many_depths, con['nuclide'], p, 
                    n_gl=con['n_gl'], postgl_shielding=con['postgl_shielding'])
many_deths_m = many_depths / 100 / con['rho']
plt.semilogx(best_conc_manypts, many_deths_m, '-b', 
             label='Best Simulation ($\Chi^2$=%0.2f)' % min_misfit)
plt.show()

figname = 'best_conc'
plt.savefig(figname + '.svg')
plt.savefig(figname + '.png')
