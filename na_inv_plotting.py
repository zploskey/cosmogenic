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
conc_true = np.genfromtxt('conc_true.txt')
conc_meas = np.genfromtxt('conc_meas.txt')
conc_meas_err = n.measurement_error(conc_meas)
ms = np.genfromtxt('ms.txt')
misfits = np.genfromtxt('misfits.txt')
constraints = joblib.load('constraints.dat')
con = constraints
p = joblib.load('production_rate.dat')
dof = con['n_gl']

# denormalize the models
ms *= con['max_dz'] / con['rho'] / 100.0

dz_true_m = con['dz_true_m']
# get data for plotting a depth vs time curve
t_true, z_true = sim.depth_v_time(con['t_gl'], con['t_int'], 
                                  con['t_postgl'], dz_true_m, 
				                  n_gl=con['n_gl'])
dvt_len = 2 * (con['n_gl'] + 1)
fit_t = np.zeros((misfits.size, dvt_len))
fit_z = np.empty((misfits.size, dvt_len))
for i in range(misfits.size):
    fit_t[i, :], fit_z[i, :] = sim.depth_v_time(con['t_gl'], con['t_int'],
                                                con['t_postgl'], ms[i], 
                                                n_gl=con['n_gl'])

#############################
# PLOTTING                  #
#############################

# input zoft plot (just input depth vs time)
zoft_fig = plt.figure(1)
zoft_ax = zoft_fig.add_subplot(111)
plt.plot(t_true / 1e6, z_true)
zoft_ax.invert_yaxis()
zoft_ax.invert_xaxis()
plt.xlabel('Myr before present')
plt.ylabel('Depth (m)')
plt.title('Past depth of the surface sample')
plt.show()

# zoft plot including all the close exposure histories
many_zoft_fig = plt.figure(2)
many_zoft_ax = many_zoft_fig.add_subplot(111)
plt.plot(t_true / 1e6, z_true, label='Target')
# make the brightness of each curve dependent on how low its error was
alphas = 1 - (misfits / dof)
alpha = 0.15
for i in range(misfits.size):
    plt.plot(fit_t[i] / 1e6, fit_z[i], 'k', alpha=alpha)
plt.legend(loc='lower right')
many_zoft_ax.invert_yaxis()
many_zoft_ax.invert_xaxis()
plt.xlabel('Myr before present')
plt.ylabel('Depth (m)')
plt.title('Past depth of the surface sample')
plt.show()

figname = 'best_histories'
plt.savefig(figname + '.eps')
plt.savefig(figname + '.png')

###########################################
# CONCENTRATION PLOTS                     #
###########################################

# convert depth to meters to be used below
z_m = con['sample_depths_m']

# Predicted concentration profile
true_conc_fig = plt.figure(3)
true_conc_ax = true_conc_fig.add_subplot(111)
plt.semilogx(conc_true, z_m, '*-', label='Target')
true_conc_ax.invert_yaxis()
plt.legend(loc='lower right')
plt.xlabel('$^{10}$Be Concentration (atoms / g)')
plt.ylabel('Depth (m)')
plt.title('Synthetic $^{10}$Be Concentration Profile')
plt.show()

figname = 'target_history'
plt.savefig(figname + '.eps')
plt.savefig(figname + '.png')

## calculate measurement error

# Perturbed + true
pert_conc_fig = plt.figure(4)
pert_conc_ax = pert_conc_fig.add_subplot(111)
plt.semilogx(conc_true, z_m, '*-', label='Target')
plt.errorbar(conc_meas, z_m,
             xerr=conc_meas_err,
             fmt='.',
             ecolor='k',
             label='Target + Error')
pert_conc_ax.invert_yaxis()
plt.legend(loc='lower right')
plt.xlabel('$^{10}$Be Concentration (atoms / g)')
plt.ylabel('Depth (m)')
plt.title('Synthetic and Perturbed $^{10}$Be Concentration Profiles')
plt.show()

# concentration profile with best fit
best_conc_fig = plt.figure(5)
best_conc_ax = best_conc_fig.add_subplot(111)
plt.semilogx(conc_true, z_m, '*-', label='Target')
plt.errorbar(conc_meas, z_m,
             xerr=conc_meas_err,
             fmt='.',
             ecolor='k',
             label='Target + Error')
min_misfit = min(misfits)
best_m = ms[misfits == min_misfit].T * 100.0 * con['rho']
n_gl = con['n_gl']
best_tgl = np.ones(n_gl) * con['t_gl']
best_tint = np.ones(n_gl) * con['t_int']
best_conc = sim.multiglaciate(best_m, best_tgl, best_tint, con['t_postgl'],
                con['sample_depths'], con['nuclide'], p, n_gl=con['n_gl'])
plt.semilogx(best_conc, z_m, '-', label='Best fit (chi2=%f.2)' % min_misfit)
best_conc_ax.invert_yaxis()
plt.legend(loc='lower right')
plt.xlabel('$^{10}$Be Concentration (atoms / g)')
plt.ylabel('Depth (m)')
plt.title('$^{10}$Be Depth Profiles')
plt.show()

figname = 'best_conc'
plt.savefig(figname + '.eps')
plt.savefig(figname + '.png')
