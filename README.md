Cosmogenic
==========

Cosmogenic is a Python library for modeling in-situ cosmogenic nuclide production and geomorphic processes.
Cosmogenic is still in beta.
We welcome contributions in the form of pull requests and bug reports.

Modules
-------

* muon:          production from muons
* production:    spallation and total production rate functions
* scaling:       functions for scaling cosmogenic nuclide production rates
* sim:           geomorphic and exposure models
* util:          utility functions
* na:            Cython implementation of the neighborhood algorithm
* datareduction: functions for reducing AMS data (work in progress)


Installation
------------

To work properly, Cosmogenic expects that you have already installed a recent version of the SciPy stack.
See here for your options:

http://www.scipy.org/install.html

The simplest way to get a working development system is to install Enthought Canopy or Anaconda from Continuum Analytics.
On Linux systems you may want to install the dependencies using your distribution's package manager.
The author prefers to install development headers and build the package using pip inside a virtualenv.
This can be accomplished by running "pip install -r pip-requirements" in the project root directory.

Until we begin to produce official release tarballs, you will need Cython and a working C compiler to build this project. Testing has been done using GCC.

To build, execute the following commands:

```
$ python setup.py build_ext --inplace
```

Then install with either:

```
$ python setup.py install
```

Or if you intend to make changes to the library's code, you can use:

```
$ python setup.py develop
```

This will allow any changes you make to the library code to be available without reinstalling each time.
