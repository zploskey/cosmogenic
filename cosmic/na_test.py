from numpy import array, append, concatenate, linspace
from numpy import ones, random, savetxt, vstack, zeros

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import sim
import nuclide

from na import NASampler

n = nuclide.Be10Qtz()
rho = 2.67 # density, g / cm^3
ngl = 20 # number of glaciations
z = linspace(0, 500, ngl) * rho # evenly spaces sample depths in g / cm^2
h_surface = 200 # roughly 200 meters?
lat = 44.5      # degrees N

# keep it simple to start... 100 cm per glaciation, 20 glaciations

dz = 100.0 * rho * ones(ngl)
t_gl = 20000 * ones(ngl)
t_intergl = 80000 * ones(ngl)
t_postgl = 12000

# get data for plotting a depth vs time curve
t_true, z_true = sim.depth_v_time(t_gl, t_intergl, t_postgl, dz)

# expose is fnc of depth z, surface_elevation, exposure history, depths_eroded
conc_true = sim.multiglaciate(dz, t_gl, t_intergl, t_postgl, z, n, h_surface, 
                              lat)

conc_meas = random.normal(loc=conc_true, scale=n.measurement_error(conc_true))
savetxt('conc_meas.txt', conc_meas)

def err(a, b, n_meas):
    """ Chi-squared error for two vectors """
    return ((a - b) ** 2 / n_meas).sum()

# degrees of freedom in our problem
dof = 3 * ngl
# target_err = err(conc_meas, conc_true, conc_meas.size)
print 'Target error is', str(dof)

# Ok, there's the input data. Let's recover some depth histories

# Define some limits to the parameter space
dz_max = 1000 * rho # max 15 m removed per glaciation... equiv to ~150 m / myr
tstep_max = 3e5 # 500kyr interglacial or glacial period is really long
lo_lim = zeros(3 * ngl) # all our lower bounds are zeros
hi_lim = concatenate(([dz_max] * ngl, [tstep_max] * 2 * ngl))

# define our parameters for the Neighborhood Algorithm
ns = 10 # number of samples each iteration
nr = 2  # number of voronoi cells that we explore in each iteration

concs = array([])
errors = array([])
models = array([])
num_unsaved = 0
UNSAVED_LIMIT = 10
vecs_to_save = ('concs', 'errors', 'models')

def save_vecs(vecs):
    for v in vecs:
        savetxt(v + '.txt', eval(v))
    return

def f(m):
    global errors, concs, models, num_unsaved
    
    cur_dz = m[0 : ngl]
    cur_t_gl = m[ngl : 2*ngl]
    cur_t_intergl = m[2*ngl : 3*ngl]
    
    conc = sim.multiglaciate(cur_dz, cur_t_gl, cur_t_intergl, t_postgl, z, n,
                             h_surface, lat)
    
    error = ((conc - conc_true) ** 2 / conc_true).sum()
    
    print "Chi2:", error
    
    if concs.size == 0:
        concs = conc.copy()
        errors = error.copy()
        models = m.copy()
    else:
        concs = vstack((concs, conc))
        errors = append(errors, error)
        models = vstack((models, m))
    
    if num_unsaved >= UNSAVED_LIMIT:
        save_vecs(vecs_to_save)
        print 'Saved current progress to disk.'
        num_unsaved = 0
    num_unsaved += 1
    return error

sampler = NASampler(ns, nr, f, lo_lim, hi_lim, dof)

sampler.generate_ensemble(100)
