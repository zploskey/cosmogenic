from cosmogenic import na, util

#na.SINGLE_PROCESS_DEBUG = True
conf = util.unpickle('conf.pkl')
mr = na.resample(config=conf)
