from __future__ import print_function

import glob
import multiprocessing
import os

from distutils.core import setup
from distutils.extension import Extension

def is_source_pkg():
    cwd = os.path.abspath(os.path.dirname(__file__))
    return not os.path.exists(os.path.join(cwd, "PKG-INFO"))

ext_names = ["na"]

def get_exts():
    if is_source_pkg():
        exts = []
        paths = glob.glob("cosmogenic/*.pyx")
        for mod_path in paths:
            mod = os.path.join("cosmogenic", 
                mod_path.split(os.sep)[-1].split(".")[0])
            exts.append(Extension(mod, [mod_path])) 
    else:
        from Cython.Build import cythonize
        parallel_builds = 1.5 * multiprocessing.cpu_count()
        exts = cythonize("cosmogenic/*.pyx", nthreads=parallel_builds)
    return exts 

setup(
    name = "cosmogenic",
    packages = ["cosmogenic"],
    version = "0.1.0",
    description = "Library for modeling cosmogenic nuclides",
    author = "Zach Ploskey",
    author_email = "zploskey@gmail.com",
    url = "http://github.com/cosmogenic/cosmogenic",
    keywords = ["cosmic rays", "geomorphology", "modeling"],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        ],
    requires=[
        "numpy (>=1.6)",
        "scipy (>=0.11)",
        "matplotlib (>=1.1)",
        "ipython (>=0.14)",
        "numexpr (>=2.0)",
        "joblib (>=0.6)",
        ],
    ext_modules = get_exts(),
) 
