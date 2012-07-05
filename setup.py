#!/usr/bin/env python

from distutils.core import setup
from Cython.Build import cythonize

setup(name = 'Cosmic',
      version = '0.1',
      description = 'Cosmic Ray Exposure Utilities',
      author = 'Zach Ploskey',
      author_email = 'zploskey@uw.edu',
      packages = ['cosmic'],
      ext_modules = cythonize("cosmic/*.pyx") 
)
