# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:08:19 2011

@author: zploskey
"""

import numpy as np
import joblib

import na

# for debugging purposes
#np.seterr(all='raise')

data_dir = 'debug1/' # include trailing slash

m = np.loadtxt(data_dir + 'models.txt')
x2v = np.loadtxt(data_dir + 'errors.txt')
con = joblib.load(data_dir + 'constraints.dat')
dof = con['dof']
Nw = 100
n = 1000

d = m.shape[1]

limits = (np.ones(d) * con['min_dz'], np.ones(d) * con['max_dz'])

mr = na.resample(m, x2v, dof, Nw, n, limits, n_jobs=2)
np.savetxt(data_dir + 'mr_100k.txt', mr)
