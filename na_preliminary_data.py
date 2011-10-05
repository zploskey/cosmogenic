from __future__ import division

import numpy as np
import joblib

from scipy.interpolate import UnivariateSpline

import sim
import nuclide
import na
import production

constraints = {
    'alt':            260,   # surface elevation, assumed constant, m
    'lat':            44.544,
    'rho':            2.67,  # g/cm2
    # define max allowable parameter values to normalize to
    'max_dz_m':       10,    # max m of rock eroded each glaciation
    'min_dz_m':       0.01,  # m
    't_gl':           15000,
    't_int':          85000,
    't_postgl':       15500,
    'postgl_shielding': 58, # postglacial shielding correction (for snow/till cover)
    'nuclide':        nuclide.Be10Qtz(),
    'n_gl':           30,     # number of glaciations
    # define our parameters for the Neighborhood Algorithm
    'ns':             100, # number of samples each iteration
    'nr':             50,  # number of voronoi cells that we explore in each iteration
    'ensemble_size':  100,
    'n_prod_interp_pts': 500,
}
con = constraints

con['sample_depths_cm'] = np.array([
1,
14.25,
19,
34,
113,
208,
367])
con['sample_depths_m'] = con['sample_depths_cm'] / 100
con['n_samples'] = con['sample_depths_m'].size

conc_meas = np.array([
57287.0,
47274.1,
42731.2,
32661.6,
12838.1,
8803.6,
5615.4])
con['conc_meas'] = conc_meas
con['sample_depths'] = con['sample_depths_m'] * 100 * con['rho']
con['bottom_depth'] = con['sample_depths'][-1]
con['bottom_depth_m'] = con['sample_depths_m'][-1]
con['max_dz'] = con['max_dz_m'] * 100 * con['rho']
con['min_dz'] = con['min_dz_m'] * 100 * con['rho']

# Interpolate the production rate
max_possible_depth = con['n_gl'] * con['max_dz'] + con['bottom_depth']
npts = con['n_prod_interp_pts'] # points to interpolate through
zs = np.unique(np.logspace(0, np.log2(max_possible_depth + 1), npts, base=2)) - 1
prod_rates = production.P_tot(zs, con['alt'], con['lat'], con['nuclide'])
#p = interp1d(zs, prod_rates, kind=3) # DON'T USE THIS
# interp1d used as above gives crazy negative dives around z = 82000, do not
# want that! UnivariateSpline seems to work much much better
p = UnivariateSpline(zs, prod_rates, k=3, s=0)
joblib.dump(p, 'production_rate.dat')

# we need a way to measure error between models
def chi2(a, b, sigma):
    """ Chi squared of two vectors """
    return (((a - b) / sigma)**2).sum()

# degrees of freedom in our problem
dof = con['n_gl']
print 'Degrees of freedom =', dof
sigma = con['nuclide'].measurement_error(conc_meas)

# limits of the parameter space, normalized to be in [0, 1]
hi_lim = np.ones(con['n_gl'])
lo_lim = np.zeros(con['n_gl'])

joblib.dump(con, 'constraints.dat') # save that input data!
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
    error = chi2(conc, conc_meas, sigma)
    print "Chi**2:", error
    return error

# concs, errors, models

sampler = na.NASampler(con['ns'], con['nr'], fn, lo_lim, hi_lim, tol=dof) 
sampler.generate_ensemble()
tol = np.max([dof, np.min(sampler.misfit)]) + 0.05 # just our best few if didn't meet tolerance
fit_idx = sampler.misfit < tol
ms = np.atleast_2d(sampler.m[fit_idx])
misfits = sampler.misfit[fit_idx] # sampler.fitting_models(350)
errors = sampler.misfit
models = sampler.m * (con['max_dz'] - con['min_dz']) + con['min_dz']
vecs_to_save = ('concs', 'errors', 'models', 'ms', 'misfits', 'conc_meas')
for v in vecs_to_save:
    np.savetxt(v + '.txt', eval(v))
