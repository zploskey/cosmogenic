from __future__ import division

import numpy as np

from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

import sim
import nuclide
import na
import production

constraints = {
    'n_samples':      10,
    'bottom_depth_m': 10,
    'h_surface':      0.0,   # surface elevation, assumed constant, m
    'lat':            65.0,
    'rho':            2.67,  # g/cm2
    # define max allowable parameter values to normalize to
    'dz_true_m':      2.5,  # m / glaciation
    'max_dz_m':       10,    # m, setting a lower max to see if we still get
                            # high old erosion rate
    'min_dz_m':       0.01,  # m
    't_gl':           20000,
    't_int':          80000,
    't_postgl':       12000, #yr
    'nuclide':        nuclide.Be10Qtz(),
    'n_gl':           30,     # number of glaciations
    'alt':            0,
    # define our parameters for the Neighborhood Algorithm
    'ns':             100, # number of samples each iteration
    'nr':             20,  # number of voronoi cells that we explore in each iteration
    'ensemble_size':  100,
}
con = constraints

con['sample_depths_m'] = np.logspace(0, np.log(con['bottom_depth_m'] + 1),
                                     con['n_samples'], base=np.e) - 1
con['sample_depths'] = con['sample_depths_m'] * 100 * con['rho']
con['bottom_depth'] = con['sample_depths'][-1]
con['bottom_depth_m'] = con['sample_depths_m'][-1]
con['max_dz'] = con['max_dz_m'] * 100 * con['rho']
con['min_dz'] = con['min_dz_m'] * 100 * con['rho']
  
dz_true = np.ones(con['n_gl']) * (con['dz_true_m'] * con['rho'] * 100)
assert con['max_dz'] > dz_true[0]
mtrue = dz_true / con['max_dz']

# Interpolate the production rate
max_possible_depth = con['n_gl'] * con['max_dz'] + con['bottom_depth']
npts = 500 # points to interpolate through
zs = np.unique(np.logspace(0, np.log2(max_possible_depth + 1), npts, base=2)) - 1
prod_rates = production.P_tot(zs, con['alt'], con['lat'], con['nuclide'])
#p = interp1d(zs, prod_rates, kind=3)
# interp1d used as above gives crazy negative dives around z = 82000, do not
# want that! UnivariateSpline seems to work much much better
p = UnivariateSpline(zs, prod_rates, k=3, s=0)
util.pickle(p, 'production_rate.dat')

# get data for plotting a depth vs time curve, meters and years
t_true, z_true = sim.glacial_depth_v_time(con['t_gl'], con['t_int'], con['t_postgl'], 
                                  con['dz_true_m'], n_gl=con['n_gl'])

conc_true = sim.multiglaciate(dz_true, con['t_gl'], con['t_int'],
                              con['t_postgl'], con['sample_depths'], 
                              con['nuclide'], p, con['n_gl'])

sigma_true = con['nuclide'].measurement_error(conc_true)
conc_meas = np.random.normal(loc=conc_true, scale=sigma_true)

# we need a way to measure error between models
def chi2(a, b, sigma):
    """ Chi squared of two vectors """
    return (((a - b) / sigma)**2).sum()

# degrees of freedom in our problem
dof = dz_true.size
print 'Degrees of freedom =', dof
sigma = con['nuclide'].measurement_error(conc_meas)
perm_err = chi2(conc_meas, conc_true, sigma)
print 'Error from permutation =', perm_err

# limits of the parameter space, normalized to be in [0, 1]
hi_lim = np.ones(con['n_gl'])
lo_lim = np.zeros(con['n_gl'])

util.pickle(con, 'constraints.dat') # save that input data!
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
                con['sample_depths'], con['nuclide'], p, n_gl=con['n_gl'])
    if SAVE_CONCENTRATION_DATA:
        concs.append(conc)
    error = chi2(conc, conc_meas, sigma)
    print "Chi**2:", error
    return error

# concs, errors, models
sampler = na.NASampler(con['ns'], con['nr'], fn, lo_lim, hi_lim, tol=dof)
sampler.generate_ensemble(con['ensemble_size'])
ms, misfits = sampler.fitting_models()
errors = sampler.misfit
models = sampler.m * (con['max_dz'] - con['min_dz']) + con['min_dz']
vecs_to_save = ('concs', 'errors', 'models', 'ms', 'misfits', 'conc_meas', 'conc_true')
for v in vecs_to_save:
    np.savetxt(v + '.txt', eval(v))
