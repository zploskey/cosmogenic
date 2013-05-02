#!/usr/bin/env python

# we'll support python 2.6 and newer
from __future__ import division, print_function, unicode_literals

from distutils.core import setup

try:
    from Cython.Build import cythonize
except ImportError as e:
    print("Cython is required.", e.value)
    raise ImportError

setup(name = 'cosmogenic',
      version = '0.1',
      description = 'Library for modeling cosmogenic nuclides',
      author = 'Zach Ploskey',
      author_email = 'zploskey@uw.edu',
      packages = ['cosmogenic'],
      ext_modules = cythonize(["cosmogenic/*.pyx", "cosmogenic/*.pyd"],
                              exclude_failures=False),
)
