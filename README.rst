==========
Cosmogenic
==========

Cosmogenic is a Python library for modeling in-situ cosmogenic nuclide accumulation during various geomorphic processes.

This package contains:

*   Functions to calculate total nuclide production rates (from reactions due to spallation + muons) for Be-10 and Al-26 in quartz, and Cl-36 in K-feldspar.
*   Production rate functions from both spallation and muons.
    Muon production is modeled after Heisinger (2002a,b) with some modified constants.
*   Scaling functions to scale production rates to a site latitude and altitude using the Lal/Stone scaling scheme (Stone 2000). More scaling schemes and production pathways are planned to be implemented in the future.
*   Functions to integrate the production rate functions over time in different exposure, erosion, and burial scenarios.     It provides the tools to model and predict cosmogenic nuclide concentrations in surface samples or depth profiles given a specific erosion history.
*   Tools to invert cosmogenic nuclide data for model parameters of your choice using the Neighborhood Algorithm (Sambridge 1999a).
    The Neighborhood Algorithm is completely general, and can be used to invert for the parameters of any model callable from Python.
    The user can write a Python misfit function to determine the goodness of fit to the data for a set of model parameters, and passes this to the library to perform the inversion.

This library is under active development.
Predictive models of cosmogenic concentrations for a prescribed erosion history should be correct.
Tools in this package for calculating exposure ages are still experimental.
If you want to calculate exposure ages, you would be best served by using the exposure age calculator at http://hess.ess.washington.edu.

Cosmogenic is open source software (under the BSD license), and is currently in beta.
It should run using either Python 2 (>=2.6) or Python 3 (>=3.2).
We welcome contributions in the form of pull requests and bug reports.
Please report problems, suggestions and requests on the issue tracker:

https://github.com/zploskey/cosmogenic/issues

Modules
=======

* muon:          production from muons
* production:    spallation and total production rate functions
* scaling:       functions for scaling cosmogenic nuclide production rates
* sim:           geomorphic history models/simulations
* na:            Implementation of the neighborhood algorithm search phase
* nuclide:       specific nuclides / target material models
* util:          utility functions

Experimental modules still in heavy development or missing functionality:

* dating:        functions for calculating exposure ages
* datareduction: functions for reducing AMS data
* parma:         PARMA (Sato et al. 2008) analytical radiation model of the atmosphere 

Dependencies
============

Cosmogenic requires the following Python packages to be installed.

* NumPy
* SciPy
* Matplotlib
* Cython

Building the documentation requires:

* Sphinx

Building the Cython code requires a C compiler such as gcc, clang or mingw.

Installation
============

To work properly, Cosmogenic expects that you have already installed a recent version of the SciPy stack.
See here for your options:

http://www.scipy.org/install.html

The simplest way to get a working development system is to install Enthought Canopy or Anaconda from Continuum Analytics.
On Linux systems you may want to install the dependencies using your distribution's package manager.


Then install with either::

    $ python setup.py install

Or if you intend to make changes to the library's code, you can use::

    $ python setup.py develop

This will allow any changes you make to the library code to be available without reinstalling each time.
