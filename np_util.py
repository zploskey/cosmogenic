#!/usr/bin/python

from numpy import vectorize, ndarray

# np_util
# utility functions to be used with numpy
# just a function wrapper right now to vectorize function
def autovec(f):
    """
    Function decorator to do vectorization only as necessary.
    """
    def wrapper(input):
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
      try:
         return self.cache[args]
      except KeyError:
         value = self.func(*args)
         self.cache[args] = value
         return value
      except TypeError:
         # uncachable -- for instance, passing a list as an argument.
         # Better to not cache than to blow up entirely.
         return self.func(*args)
   def __repr__(self):
      """Return the function's docstring."""
      return self.func.__doc__
   def __get__(self, obj, objtype):
      """Support instance methods."""
      return functools.partial(self.__call__, obj)
