
import numpy
import os

from setuptools import setup

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'cosmogenic')

setup(
    name="cosmogenic",
    packages=["cosmogenic"],
    version="0.1.0",
    description="Library for modeling cosmogenic nuclides",
    author="Zach Ploskey",
    author_email="zploskey@gmail.com",
    url="http://github.com/zploskey/cosmogenic",
    keywords=["cosmic rays", "geomorphology", "modeling"],
    license="BSD",
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
        "ipython (>=0.14)",
        ],
    )
