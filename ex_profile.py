#!/usr/bin/python

import time

from numpy import arange, array
import matplotlib as mpl
import matplotlib.pyplot as plt

from production import P_sp
import sample
import muon
import nuclide as n


if __name__ == '__main__':

    start = time.time()
    
    # s = sample.Sample(rho=2.67, h=1000, lat=65, shielding=1.0, z=1)
    # print "Total muon flux: %f " % muon.tot_mu_flux(s)
    rho = 2.67
    h = array([352,4039,4232,1234,5343,321,958,3221,4948,509])
    lat = 39 # array([54,32,34,32,12,45,65,76,43,78])
    shielding = 1.0
    z = arange(0, 500, 50)
    be10 = n.Be10Qtz()
    prod_profile = P_sp(z, rho, h, lat, be10)
      
    end = time.time()
    elapsed = end - start
    
    print "Took", elapsed, "seconds to run."
    plt.scatter(z, prod_profile)
    plt.set_marker('.')
    plt.show()



