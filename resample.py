# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:08:19 2011

@author: zploskey
"""
import os
import sys

import numpy as np
import joblib

import na

argv = sys.argv
data_dir = argv[1]

if not os.path.exists(data_dir):
    raise Exception('Directory \'%s\' does not exist. Aborted.', data_dir)

if len(argv) > 2:
    outfile = argv[2]
else:
    outfile = 'mr.txt'

m = np.loadtxt(os.path.join(data_dir, 'models.txt'))
x2v = np.loadtxt(os.path.join(data_dir, 'errors.txt'))
con = joblib.load(os.path.join(data_dir, 'constraints.dat'))
dof = con['dof']
d = m.shape[1]
limits = (np.ones(d) * con['min_dz'], np.ones(d) * con['max_dz'])

Nw = 2 # number of walks
n = 2 # number of samples to take along each walk

mr = na.resample(m, x2v, dof, Nw, n, limits, n_jobs=2)
np.savetxt(os.path.join(data_dir, outfile), mr)
