
import os
import sys
import subprocess

import numpy

import cosmogenic

from setuptools import setup
from setuptools.extension import Extension

ISRELEASED = False

try:
    from sphinx.setup_command import BuildDoc
    HAVE_SPHINX = True
except:
    HAVE_SPHINX = False

if HAVE_SPHINX:
    class CosmogenicBuildDoc(BuildDoc):
        """Run in-place build before Sphinx doc build"""
        def run(self):
            ret = subprocess.call([sys.executable, sys.argv[0], 'build_ext', 
                                   '--inplace'])
            if ret != 0:
                raise RuntimeError("Building Cosmogenic failed!")
            BuildDoc.run(self)


def setup_package():
    
    if HAVE_SPHINX:
        cmdclass = {'build_sphynx': CosmogenicBuildDoc}
    else:
        cmdclass = {}
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
        # Compile Cython modules unless building from source release.
        from Cython.Build import cythonize 
        
        numpy_include = numpy.get_include()
        ext_modules = cythonize([
            Extension(
                "cosmogenic.muon",
                ["cosmogenic/muon.pyx"],
                libraries=["m"],
                include_dirs=[numpy_include]),
            Extension(
                "cosmogenic.parma",
                ["cosmogenic/parma.pyx"],
                include_dirs=[numpy_include]),
            ])
    
    setup(
        name="cosmogenic",
        packages=["cosmogenic"],
        version=cosmogenic.__version__,
        description="Library for modeling cosmogenic nuclides",
        author="Zach Ploskey",
        author_email="zploskey@gmail.com",
        url="http://github.com/zploskey/cosmogenic",
        keywords=["cosmic rays", "geomorphology", "modeling"],
        license="BSD",
        cmdclass=cmdclass,
        classifiers=[
            "License :: OSI Approved :: BSD License",
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            ],
        requires=[
            "numpy (>=1.6)",
            "scipy (>=0.11)",
            "matplotlib (>=1.1)",
            ],
        ext_modules=ext_modules,
    )

if __name__ == '__main__':
    setup_package()
