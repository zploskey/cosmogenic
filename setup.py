
import numpy

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


setup(
    name="cosmogenic",
    packages=["cosmogenic"],
    version="0.1.0",
    description="Library for modeling cosmogenic nuclides",
    author="Zach Ploskey",
    author_email="zploskey@gmail.com",
    url="http://github.com/cosmogenic/cosmogenic",
    keywords=["cosmic rays", "geomorphology", "modeling"],
    classifiers=[
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
        "joblib",
        ],
    ext_modules=cythonize(
        Extension(
            "cosmogenic.na",
            ["cosmogenic/na.pyx"],
            libraries=["m"],
            include_dirs=[numpy.get_include()])
        )
)
