from __future__ import division

import numpy as np
from scipy.interpolate import interp1d

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
#    'max_dzdt_m':     50,    # long term erosion m / Myr
#    'min_dzdt_m':     1,     # m / Myr
    'max_dz_m':       10,    # m
    'min_dz_m':       0.01,  # m
#    'max_t':          5e5,   # longest time period (glacial or interglacial), yr
#    'min_t':          1,     # yr
    't_postgl':       12000, #yr
    'nuclide':        nuclide.Be10Qtz(),
    'n_gl':           30,     # number of glaciations
    'alt':            0,
}

con = constraints

# evenly spaces samples in m
con['sample_depths_m'] = np.linspace(0, 10, 20) # meters depth 
con['sample_depths'] = con['sample_depths_m'] * 100 * con['rho']
con['bottom_depth'] = con['sample_depths'][-1] 
con['sample_depths'] = np.linspace(0, con['bottom_depth'], con['n_samples'])
con['max_dz'] = con['max_dz_m'] * 100 * con['rho']
con['min_dz'] = con['min_dz_m'] * 100 * con['rho']
  
# Define our "true" model
# 25 m per 20 kyr glaciation, 80kyr interglacials
dz_m  = 2.5 # m / glaciation
dz = dz_m * con['rho'] * con['rho']
t_gl  = 20000
t_int = 80000

mtrue_units = np.ones(con['n_gl']) * dz
# mtrue_units[-2:-1] = [t_gl, t_int]
mtrue = mtrue_units / con['max_dz']

# Interpolate the production rate
max_possible_depth = con['n_gl'] * con['max_dz'] + con['bottom_depth']
npts = 200 # points to interpolate through
zs = np.unique(np.logspace(0, np.log(max_possible_depth), 200, base=np.e)) - 1
zs = np.concatenate((zs, [max_possible_depth]))
prod_rates = production.P_tot(zs, con['alt'], con['lat'], con['nuclide'])
p = interp1d(zs, prod_rates, kind=3) # our production rate function

# get data for plotting a depth vs time curve
#t_true, z_true = sim.steady_depth_v_time(mtrue_units, con['t_postgl'],
#                                         con['nuclide'])
#
#conc_true = sim.steady_multiglaciate(mtrue_units, con)
#
#conc_meas = np.random.normal(loc=conc_true, 
#                             scale=con['nuclide'].measurement_error(conc_true))
#np.savetxt('conc_meas.txt', conc_meas)
#np.savetxt('conc_true.txt', conc_true)
#
#def chi2(a, b, n_meas):
#    """ Chi squared of two vectors """
#    return (((a - b) / n_meas)**2).sum()
#
## degrees of freedom in our problem
#dof = t_gl.size() + t_intergl.size() + dz.size()
## target_err = err(conc_meas, conc_true, conc_meas.size)
#print 'Target error is', dof
#
## Ok, there's the input data. Let's recover some depth histories
#
## Define some limits to the parameter space
#hi_lim = np.concatenate((con[dzdt_max], con['max_t'], con['max_t']))
#
## define our parameters for the Neighborhood Algorithm
#ns = 10 # number of samples each iteration
#nr = 2  # number of voronoi cells that we explore in each iteration
#
#concs = np.array([])
#errors = np.array([])
#models = np.array([])
#num_unsaved = 0
#UNSAVED_LIMIT = ns
#vecs_to_save = ('concs', 'errors', 'models')
#
#def save_vecs(vecs):
#    for v in vecs:
#        np.savetxt(v + '.txt', eval(v))
#    return
#
#def fn(m):
#    """
#    Our objective function, takes an ndarray m.
#    
#    m[0] -- erosion depth in cm
#    m[1] -- length of glacial periods (years)
#    m[2] -- length of interglacial periods (years)
#    """
#    global errors, concs, models, num_unsaved
#    
#    cur_dz = m[0] * rho
#    cur_t_gl = m[1]
#    cur_t_intergl = m[3]
#
#    conc = sim.multiglaciate(cur_dz, cur_t_gl, cur_t_intergl, t_postgl, z, n,
#                             h_surface, lat)
#    
#    error = chi2(conc, conc_true, )
#    
#    print "Chi**2:", error
#    
#    if concs.size == 0:
#        concs = conc.copy()
#        errors = error.copy()
#        models = m.copy()
#    else:
#        concs = vstack((concs, conc))
#        errors = append(errors, error)
#        models = vstack((models, m))
#    
#    if num_unsaved >= UNSAVED_LIMIT:
#        save_vecs(vecs_to_save)
#        print 'Saved current progress to disk.'
#        num_unsaved = 0
#    num_unsaved += 1
#    return error
#
#sampler = na.NASampler(ns, nr, fn, lo_lim, hi_lim, tol=dof)
#
#sampler.generate_ensemble(100)
