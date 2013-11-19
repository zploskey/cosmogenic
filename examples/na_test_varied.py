from __future__ import division

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from cosmogenic import util

import sim
import nuclide
import na
import production
import save_na_results

# constraints on the problem, central data storage
con = {
    'n_samples':      20,
    'n_exact_depths': 3,
    'n_gl':           20,     # number of glaciations
    'bottom_depth_m': 10,
    'alt':            220,   # surface elevation, assumed constant, m
    'lat':            44.544,
    'rho':            2.67,  # g/cm2
    # define max allowable parameter values to normalize to
    'max_dz_m':       15,    # max m of rock eroded each glaciation
    'min_dz_m':       0.01,  # m
    't_gl':           15000,
    't_int':          85000,
    't_postgl':       15500,
    'postgl_shielding': 80, # postglacial shielding correction (for snow/till cover)
    'nuclide':        nuclide.Be10Qtz(),

    # define our parameters for the Neighborhood Algorithm
    'ns':             50, # number of samples each iteration
    'nr':             25,  # number of voronoi cells that we explore in each iteration
    'ensemble_size':  10000,
#    'n_best':         200,
    'interp_pts': 500,
    'tol_reduced_chi2': 2,
}

con['n_params'] = con['n_exact_depths'] + 1
con['dof'] = con['n_samples'] - con['n_params'];
con['bottom_depth'] = con['bottom_depth_m'] * 100 * con['rho']
con['max_dz'] = con['max_dz_m'] * 100 * con['rho']
con['min_dz'] = con['min_dz_m'] * 100 * con['rho']

# logarithmically place our sample depths
con['sample_depths_m'] = np.logspace(0, np.log(con['bottom_depth_m'] + 1), con['n_samples'], base=np.e) - 1
con['sample_depths'] = con['sample_depths_m'] * 100 * con['rho']

# Construct the target erosion history
con['dz_m'] = np.array([  4.5,   3.5,   2.5,  1.5,
        1,   1.2,   0.5,   0.3,
        0.7,   0.5,   0.4 ,   1,
        1.2,   0.2,   0.3,   0.4,
        0.25 ,   0.75,  1.2,   0.8])
con['dz'] = con['dz_m'] * 100 * con['rho']

# create points for the true erosion history model
con['t'], con['z_true'] = sim.glacial_depth_v_time(con['t_gl'], con['t_int'], con['t_postgl'], 
                             con['dz_m'], n_gl=con['n_gl'])

# plot the true model erosion history
#true_hist_fig = plt.figure()
#ax = true_hist_fig.add_subplot(111)
#plt.title('True Model Erosion History')
#var_line, = plt.plot(con['t'] * 1e-6, con['z_true'], 'k', lw=2)
#ax.invert_xaxis()
#ax.invert_yaxis()
#plt.xlabel('Time Before Present (Myr)')
#plt.ylabel('Depth (m)')
#plt.grid(linestyle='-', color='0.75')
#plt.xlim(left=2)
#plt.ylim(bottom=75)
#plt.savefig('true_hist.png')

# interpolate a production function
max_possible_depth = con['n_gl'] * con['max_dz'] + con['bottom_depth']

# if production file exists already
p = production.interpolate_P_tot(max_possible_depth, 
                                 npts=con['interp_pts'], alt=con['alt'],
                                 lat=con['lat'], n=con['nuclide'])

# calculate the true concentration profile (w/ many points, not sample depths)
z_true_manypts = np.linspace(0, con['bottom_depth'], 200)
C_true_manypts = sim.multiglaciate(con['dz'], con['t_gl'], con['t_int'], 
                             con['t_postgl'], z_true_manypts, con['nuclide'], p,
                             con['n_gl'], 
                             postgl_shielding=con['postgl_shielding'])

## make the true concentration graph
#true_conc_fig = plt.figure()
#ax = true_conc_fig.add_subplot(111)
#ax.title('True Concentration Profile')
#plt.semilogx(C_true_manypts, z_true_manypts / con['rho'] / 100, lw=2)
#ax.invert_yaxis()
#plt.xlabel(r'[$^{10}$Be] (atoms/g)')
#plt.ylabel('Depth (m)')
##ax.tick_params(labelsize=6)
##plt.setp(true_conc_ax.get_xticklabels(), rotation='horizontal')
##true_conc_fig.subplots_adjust(left=0.15, bottom=0.12, top=0.95)
#plt.savefig('true_conc.png')

# calculate true concentrations at samples depths
C_true = sim.multiglaciate(con['dz'], con['t_gl'], con['t_int'], 
                             con['t_postgl'], con['sample_depths'], 
                             con['nuclide'], p, con['n_gl'], 
                             postgl_shielding=con['postgl_shielding'])
con['C_true'] = C_true                             
con['sigma'] = con['nuclide'].measurement_error(C_true)
C_meas = np.random.normal(loc=C_true, scale=con['sigma'])
C_meas_err = con['nuclide'].measurement_error(C_meas)
con['C_meas'] = C_meas
con['C_meas_err'] = C_meas_err

#meas_fig = plt.figure()
#ax = meas_fig.add_subplot(111)
#ax.title('Synthetic Concentration Measurements', y=1.03, fontsize=12)
#ax.errorbar(C_meas, con['sample_depths_m'], xerr=C_meas_err, fmt='k.')
#ax.invert_yaxis()
#ax.set_xlabel('[$^{10}$Be] (atoms/g)', fontsize=10)
#ax.set_ylabel('Depth (m)', fontsize=10)
#ax.set_xscale('log')
#meas_fig.subplots_adjust(left=0.15) #, top=0.95
#plt.savefig('meas_conc.png')

# we need a way to measure difference between models, use reduced Chi Squared!
def chi2(a, b, sigma):
    """ Chi squared of two vectors """
    return (((a - b) / sigma)**2).sum() / con['dof']

# assess how much error was introduced in permutation
print 'Degrees of freedom =', con['dof']
perm_err = chi2(con['C_meas'], con['C_true'], con['C_meas_err'])
print 'Error from permutation =', perm_err
con['permutation_error'] = perm_err

# limits of the parameter space, normalized to be in [0, 1]
hi_lim = np.ones(con['n_params'])
lo_lim = np.zeros(con['n_params'])

concs = []
indep_errs = []
SAVE_CONCENTRATION_DATA = True
def fn(m):
    """ Our objective function, takes an ndarray m that contains depths removed
    during each glaciation as a fraction of con['dz_max'].
    """
    if SAVE_CONCENTRATION_DATA:
        global concs
        global indep_errs
    raw_dz = m * (con['max_dz'] - con['min_dz']) + con['min_dz']
    dz = np.ones(con['n_gl']) * raw_dz[-1]
    dz[0:con['n_exact_depths']] = raw_dz[0:-1]
    conc = sim.multiglaciate(dz, con['t_gl'], con['t_int'], con['t_postgl'],
                con['sample_depths'], con['nuclide'], p, n_gl=con['n_gl'],
                postgl_shielding=con['postgl_shielding'])

    error = chi2(conc, con['C_meas'], con['C_meas_err'])
    print "Chi**2:", error

    if SAVE_CONCENTRATION_DATA:
        concs.append(conc)
        indep_errs.append(error)
    return error

# adjust our error tolerance for a fitting model upwards if error from the
# permutation exceeds our previously defined tolerance. If we don't do this
# the inversion could run forever and never find a fitting model.
if con['permutation_error'] > con['tol_reduced_chi2']:
    con['actual_chi2_tol'] = con['permutation_error'] + 1
else:
    con['actual_chi2_tol'] = con['tol_reduced_chi2']

# Run the inversion
sampler = na.NASampler(con['ns'], con['nr'], fn, lo_lim, hi_lim,
                       tol=con['actual_chi2_tol'],
                       min_eval=con['ensemble_size'])
sampler.generate_ensemble(con['ensemble_size'])

errors = sampler.misfit
models = sampler.m * (con['max_dz'] - con['min_dz']) + con['min_dz']
models_m = models / con['rho'] / 100.0

#ms, misfits, best_idx = sampler.best_models(con['n_best'])
#ms_m = (ms * (con['max_dz'] - con['min_dz']) + con['min_dz']) / con['rho'] / 100.0

good_idx = errors < con['actual_chi2_tol']
misfits = errors[good_idx].copy()
ms = models[good_idx].copy()
ms_m = models_m[good_idx].copy()
# extract models that fit to within tolerance
#fit_ms, fit_errs = sampler.fitting_models(con['actual_chi2_tol'])
#fit_idx = errors < con['actual_chi2_tol']
#fitting_models_m = models_m[fit_idx].copy()
#fitting_misfits = errors[fit_idx].copy()

# from those, randomly select our desired number of models to plot
#if con['n_best'] < len(fitting_misfits):
#    idx = np.random.permutation(np.arange(len(fitting_misfits)
#                                            ))[0:con['n_best']]
#    ms_m = fitting_models_m[idx].copy()
#    misfits = fitting_misfits[idx].copy()
#else:
#ms_m = fitting_models_m.copy()
#misfits = fitting_misfits.copy()

vecs_to_save = ('concs', 'errors', 'models', 'ms', 'misfits')
for v in vecs_to_save:
    np.savetxt(v + '.txt', eval(v))

dvt_len = 2 * (con['n_gl'] + 1)
fit_t = np.zeros((misfits.size, dvt_len))
fit_z = np.empty((misfits.size, dvt_len))
for i in range(misfits.size):
    cur_m = np.ones(con['n_gl']) * ms_m[i, -1]
    cur_m[0:con['n_params']] = ms_m[i]
    fit_t[i, :], fit_z[i, :] = sim.glacial_depth_v_time(con['t_gl'], con['t_int'],
                                                con['t_postgl'], cur_m, 
                                                n_gl=con['n_gl'])

min_idx = np.argmin(errors)
best_m_m = models_m[min_idx].copy()
full_best_m = np.ones(con['n_gl']) * best_m_m[-1]
full_best_m[0:con['n_params']] = best_m_m.copy()
_, best_fit_z = sim.glacial_depth_v_time(con['t_gl'], con['t_int'],
                                          con['t_postgl'], full_best_m,
                                          n_gl=con['n_gl'])


#############################
# PLOTTING                  #
#############################

# zoft plot including all the close exposure histories
zoft_fig = plt.figure()
ax = zoft_fig.add_subplot(111)
# make the brightness of each curve dependent on how low its error was
#alphas = 1 - (misfits / con['tol_reduced_chi2'])
#alpha = 0.5

shade_num = 0.4 if misfits[i] < con['tol_reduced_chi2'] else 0.8
shade = "%0.2f" % round(shade_num, 2)
for i in range(misfits.size):
    plt.plot(fit_t[i] / 1e6, fit_z[i], color=shade)

plt.plot(con['t'] * 1e-6, con['z_true'], 'k', lw=3)
plt.plot(con['t'] * 1e-6, best_fit_z, 'r', lw=3)
plt.xlim((0, con['t'][-1] * 1e-6))
ax.invert_yaxis()
ax.invert_xaxis()
plt.xlabel('Myr B.P.')
plt.ylabel('Depth (m)')
plt.title('Erosion Histories')
figname = 'zoft'
for ext in ('.eps', '.svg', '.png'):
    plt.savefig(figname + ext)


###########################################
# CONCENTRATION PLOT                      #
###########################################

# convert depth to meters to be used below
z_m = con['sample_depths_m']

# measured concentration profile plus the best fit
conc_fig = plt.figure()
ax = conc_fig.add_subplot(111)
ax.set_xscale('log')

many_depths = np.linspace(0, con['sample_depths'][-1], 200)
best_conc_manypts = sim.multiglaciate(full_best_m * 100.0 * con['rho'],
                    con['t_gl'], con['t_int'], con['t_postgl'], many_depths,
                    con['nuclide'], p, 
                    n_gl=con['n_gl'], postgl_shielding=con['postgl_shielding'])
many_depths_m = many_depths / 100 / con['rho']
plt.plot(best_conc_manypts, many_depths_m, 'k', label='best fit', lw=2)

ax.errorbar(con['C_meas'], z_m,
             xerr=con['C_meas_err'],
             fmt='.',
             markersize=7,
             color='b',
             label='measured')

ax.invert_yaxis()
ax.set_title('Observed and predicted concentration profile')
ax.set_xlabel('[$^{10}$Be] (atoms / g)')
ax.set_ylabel('Depth (m)')
ax.legend(loc='lower right')

# Record the chi squared value 
chi2_annotation = '$\chi^2_\\nu$ = %0.2f' % np.round(errors[min_idx], 2)
ax.annotate(chi2_annotation, xy=(0.75, 0.5), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center')

figname = 'conc'
for ext in ('.eps', '.svg', '.png'):
    plt.savefig(figname + ext)

# sampling distribution plotting code
#sampling_fig = plt.figure()
#ax = sampling_fig.add_subplot(111)
#plt.plot(models_m[:, 0], models_m[:, 1], 'k.', markersize=2)
#plt.plot(con['dz_m'][0], np.mean(con['dz_m'][1:]), 'xb', label='True', markeredgewidth=3, markersize=8, lw=2)
#plt.plot(best_m_m[0], best_m_m[1], '+r', label='best model', markeredgewidth=3, markersize=8, lw=2)
#plt.xlabel('Erosion Depth, Last glaciation (m)')
#plt.ylabel('Prior representative erosion depth')

#figname = 'sampling_dist'
#for ext in ('.svg', '.png'):
#    plt.savefig(figname + ext)

f = open('constraints.txt', 'w')
f.write(str(con))
f.close()
save_na_results.save_na_results()
