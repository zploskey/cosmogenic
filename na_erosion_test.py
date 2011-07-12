from __future__ import division

import numpy as np
import joblib

from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline as USpline

import sim
import nuclide
import na
import production

constraints = {
    'n_samples':      10,
    'h_surface':      0.0,   # surface elevation, assumed constant, m
    'lat':            65.0,
    'rho':            2.67,  # g/cm2
    # define max allowable parameter values to normalize to
    'max_dz_m':       10,    # m
    'min_dz_m':       0.01,  # m
    't_postgl':       12000, #yr
    'nuclide':        nuclide.Be10Qtz(),
    'n_gl':           30,     # number of glaciations
    'alt':            0,
}

con = constraints

# evenly spaces samples in m
con['bottom_depth_m'] = 10
con['sample_depths_m'] = np.linspace(0, con['bottom_depth_m'], con['n_samples'])
con['sample_depths'] = con['sample_depths_m'] * 100 * con['rho']
con['bottom_depth'] = con['sample_depths'][-1] 
con['max_dz'] = con['max_dz_m'] * 100 * con['rho']
con['min_dz'] = con['min_dz_m'] * 100 * con['rho']
  
# Define our "true" model
# 25 m per 20 kyr glaciation, 80kyr interglacials
con['t_gl'] = 20000
con['t_int'] = 80000
t_gl = con['t_gl']
t_int = con['t_int']
con['dz_true_m']  = 2.5 # m / glaciation
dz_true = np.ones(con['n_gl']) * (con['dz_true_m'] * con['rho'] * 100)
assert con['max_dz'] > dz_true[0]

mtrue = dz_true / con['max_dz']

# Interpolate the production rate
max_possible_depth = con['n_gl'] * con['max_dz'] + con['bottom_depth']
npts = 500 # points to interpolate through
zs = np.unique(np.logspace(0, np.log2(max_possible_depth), npts, base=2)) - 1
zs = np.concatenate((zs, [max_possible_depth + 50]))
prod_rates = production.P_tot(zs, con['alt'], con['lat'], con['nuclide'])
#p = interp1d(zs, prod_rates, kind=3)
# interp1d used as above gives crazy negative dives around z = 82000, do not
# want that! UnivariateSpline seems to work much much better
p = USpline(zs, prod_rates, k=3, s=0)
joblib.dump(p, 'production_rate.dat')

# get data for plotting a depth vs time curve, meters and years
t_true, z_true = sim.depth_v_time(t_gl, t_int, con['t_postgl'], con['dz_true_m'], n_gl=con['n_gl'])

conc_true = sim.multiglaciate(dz_true, t_gl, t_int, con['t_postgl'],
                              con['sample_depths'], con['nuclide'], p, 
                              con['n_gl'])

sigma_true = con['nuclide'].measurement_error(conc_true)
conc_meas = np.random.normal(loc=conc_true, scale=sigma_true)

np.savetxt('conc_meas.txt', conc_meas)
np.savetxt('conc_true.txt', conc_true)

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

# Ok, there's the input data. Let's recover some depth histories

# Define some limits to the parameter space
hi_lim = np.ones(con['n_gl'])
lo_lim = np.zeros(con['n_gl'])

# define our parameters for the Neighborhood Algorithm
con['ns'] = 100 # number of samples each iteration
con['nr'] = 20  # number of voronoi cells that we explore in each iteration

joblib.dump(con, 'constraints.dat')

concs = np.array([])
errors = np.array([])
models = np.array([])
num_unsaved = 0
UNSAVED_LIMIT = 200
vecs_to_save = ('concs', 'errors', 'models')

def save_vecs(vecs):
    for v in vecs:
        np.savetxt(v + '.txt', eval(v))
    return

def fn(m):
    """ Our objective function, takes an ndarray m that contains depths removed
    during each glaciation as a fraction of con['dz_max'].
    """
    global errors, concs, models, num_unsaved
    
    dz = m * con['max_dz']
    conc = sim.multiglaciate(dz, t_gl, t_int, con['t_postgl'],
                con['sample_depths'], con['nuclide'], p, n_gl=con['n_gl'])
    
    error = chi2(conc, conc_meas, sigma)
    
    print "Chi**2:", error
    
    if concs.size == 0:
        concs = conc.copy()
        errors = error.copy()
        models = m.copy()
    else:
        concs = np.vstack((concs, conc))
        errors = np.append(errors, error)
        models = np.vstack((models, m))
    
    if num_unsaved >= UNSAVED_LIMIT:
        save_vecs(vecs_to_save)
        print 'Saved current progress to disk.'
        num_unsaved = 0
    num_unsaved += 1
    return error

sampler = na.NASampler(con['ns'], con['nr'], fn, lo_lim, hi_lim, tol=dof)
sampler.generate_ensemble(1000)
save_vecs(vecs_to_save)
ms, misfits = sampler.fitting_models()
save_vecs(('ms', 'misfits'))

