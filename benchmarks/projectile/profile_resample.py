from __future__ import print_function

import pstats
import cProfile

import numpy as np

from cosmogenic import na, util

conf = util.unpickle('conf.pkl')
mr = np.empty((conf['Nw'] * conf['pts_per_walk'], conf['d']), dtype=float)
cProfile.runctx("mr = na.resample(config=conf)", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
print(mr)
