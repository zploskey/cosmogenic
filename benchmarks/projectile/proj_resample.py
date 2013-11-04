import numpy as np
import time

from cosmogenic import na, util

conf = util.unpickle('conf.pkl')

tic = time.time()
mr_cy = na.resample(config=conf)
toc = time.time()
cy_time = toc - tic
print "cy_time = ", cy_time, " seconds"
