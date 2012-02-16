#!/usr/bin/python
"""
np_util
utility functions to be used with numpy
"""

import functools

from numpy import vectorize, ndarray

def autovec(f):
    """
    Function decorator to do vectorization only as necessary.
    """
    def wrapper(input):
        """
        Vectorizes the function if the input is an ndarray and evaluates
        """
        if type(input) == ndarray:
            return vectorize(f)(input)
        return f(input)
    
    return wrapper

class memoized(object):
    """Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}
    
    def __call__(self, *args):
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value
    
    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__
    
    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)
