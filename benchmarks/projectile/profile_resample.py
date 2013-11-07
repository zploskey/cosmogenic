import numpy as np
import pstats, cProfile

from cosmogenic import na, util

conf = util.unpickle('conf.pkl')
cProfile.runctx("na.resample(config=conf)", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
