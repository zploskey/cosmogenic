#!/usr/bin/env python

from distutils.core import setup

try:
    from Cython.Build import cythonize
except ImportError as e:
    print "You need Cython version >= 0.18 build and install Cosmic. \n", e.value
    raise ImportError

setup(name = 'cosmogenic',
      version = '0.1',
      description = 'Library for modeling cosmogenic nuclides',
      author = 'Zach Ploskey',
      author_email = 'zploskey@uw.edu',
      packages = ['cosmogenic'],
      ext_modules = cythonize("cosmogenic/*.pyx"),
)
