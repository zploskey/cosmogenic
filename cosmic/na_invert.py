from __future__ import division

import numpy as np
import joblib

import sim
import nuclide
import na
import production

con = joblib.load('con.dat')
print con
p = joblib.load('production_rate.dat')

# we need a way to measure error between models
def chi2(a, b, sigma):
    """ Chi squared of two vectors """
    return (((a - b) / sigma)**2).sum()

# degrees of freedom in our problem

print 'Degrees of freedom =', con['dof']
try:
    perm_err = chi2(con['C_meas'], con['C_target'], con['C_meas_err'])
    print 'Error from permutation =', perm_err
except:
    pass

# limits of the parameter space, normalized to be in [0, 1]
hi_lim = np.ones(con['n_gl'])
lo_lim = np.zeros(con['n_gl'])

concs = []
SAVE_CONCENTRATION_DATA = True
def fn(m):
    """ Our objective function, takes an ndarray m that contains depths removed
    during each glaciation as a fraction of con['dz_max'].
    """
    if SAVE_CONCENTRATION_DATA:
        global concs
    dz = m * (con['max_dz'] - con['min_dz']) + con['min_dz']
    conc = sim.multiglaciate(dz, con['t_gl'], con['t_int'], con['t_postgl'],
                con['sample_depths'], con['nuclide'], p, n_gl=con['n_gl'],
                postgl_shielding=con['postgl_shielding'])
    if SAVE_CONCENTRATION_DATA:
        concs.append(conc)
    error = chi2(conc, con['C_meas'], con['sigma'])
    print "Chi**2:", error
    return error

# concs, errors, models

sampler = na.NASampler(con['ns'], con['nr'], fn, lo_lim, hi_lim, tol=con['dof'])
sampler.generate_ensemble(con['ensemble_size'])
ms, misfits = sampler.fitting_models()
errors = sampler.misfit
models = sampler.m * (con['max_dz'] - con['min_dz']) + con['min_dz']
vecs_to_save = ('concs', 'errors', 'models', 'ms', 'misfits')
for v in vecs_to_save:
    np.savetxt(v + '.txt', eval(v))
