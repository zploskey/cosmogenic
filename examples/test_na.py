# -*- coding: utf-8 -*-

import numpy as np

import na

# configuration for the neighborhood algorithm
conf = {
    'desc': 'Test of the neighorbood algorithm code.',
    'datadir': 'combined_test01',
    'd': 3,
    'n': 20,
    'sigma': 0.2,
    #
    'ns': 10,
    'nr': 2,
    'ne': 10000,
    # optional:
    'm_true': np.array([3, 7, 4], dtype=np.float64),
    # resampling
    'Nw': 8, # Number of walks
    'pts_per_walk': 300,
}
conf['lo_lim'] = np.array([0] * conf['d'])
conf['hi_lim'] = np.array([10] * conf['d'])

def fwd(m):
    t = np.linspace(0, 10, conf['n'])
    return (t * m[0] + m[1] * (t * m[2] - m[0]))**2

# predicted value (measured)
conf['pre'] = fwd(conf['m_true'])
pre = conf['pre']
v = conf['n'] - conf['d']
sigma = conf['sigma']

def chi2v(obs):
    tmp = obs - pre
    tmp /= sigma
    return np.dot(tmp, tmp) / v

# function to minimize
def func(m):
    return chi2v(fwd(m), pre)

if __name__ == "__main__":
    na.run(func, conf)
