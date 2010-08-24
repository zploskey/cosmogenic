#!/usr/bin/python

import time

from numpy import array, arange, linspace, ones
import matplotlib as mpl
import matplotlib.pyplot as plt

import sample
import muon
import nuclide as nuc
import sim

if __name__ == '__main__':

    # s = sample.Sample(rho=2.67, h=1000, lat=65, shielding=1.0, z=1)
    # print "Total muon flux: %f " % muon.tot_mu_flux(s)
    rho = 2.67

    start = time.time()

    z0 = linspace(0, 500)
    h = 2500
    lat = 39 # array([54,32,34,32,12,45,65,76,43,78])
    shielding = 1.0
    be10 = nuc.Be10Qtz()
    t = linspace(0,1e6,33)
    nt = len(t)

    z_removed = 267 * ones(nt / 2)
    print "z0 was", z0
    
    N = sim.fwd_profile(z0, z_removed, t, be10, h, lat)

    print "z0 is now", z0
    
    end = time.time()
    elapsed = end - start
    
    print "Took", elapsed, "seconds to run."

    print "N =", N

    plt.plot(z0, N)
    plt.show()
