
import numpy as np
import matplotlib.pyplot as plt

import cosmogenic.na as na
import cosmogenic.util as util

def positions(m, ts):
    a = -9.81
    v0 = m[0:2]
    p0 = m[2:4]
    pos = np.empty((ts.size, 2), dtype=np.float64)
    for i, t in enumerate(ts):
        pos[i,:] = 0.5 * a * t**2 + v0 * t + p0
    return pos

conf = {
    "description": "Test of NA on measurements of a projectile location",
    "ns": 20,
    "nr": 3,
#    "n_initial": 500,
    "lo_lim": np.array([ -50,   0, -100,    0], dtype=np.float64),  
    "hi_lim": np.array([  50, 200,  100,  200], dtype=np.float64),
    "d": 4,
    "ne": 1000,
    "shape": (4, 1),
    "m_true": np.array([ 0.0, 100.0, 0.0, 100.0], dtype=np.float64),
    # resampling
    'Nw': 48,
    'pts_per_walk': 100,
    # for benchmarking
    'seed': 10453,
    'plot': False,
}
conf['ts'] = np.linspace(2, 20, 1000)
conf['dof'] = conf['ts'].size - conf['d']

sigma_obs = 1.0 # meters
pos_true = positions(conf['m_true'], conf['ts'])

util.pickle(conf, 'conf.pkl')

def chi2v(pre, obs, sigma_obs, nu):
    return (((obs - pre) / sigma_obs)**2).sum() / nu

def fn(m):
    pos = positions(m, conf['ts'])
    misfit = chi2v(pos, pos_true, sigma_obs, conf['dof'])
    return misfit

#na.SINGLE_PROCESS_DEBUG = True
na.search(fn, conf)
