#!/usr/bin/python

import numpy as np

# np_util
# utility functions to be used with numpy
# just a function wrapper right now to vectorize function
def autovec(f):
    """
    Function decorator to do vectorization only as necessary.
    """
    def wrapper(input):
        if type(input) == np.ndarray:
            return np.vectorize(f)(input)
        return f(input)
    
    return wrapper