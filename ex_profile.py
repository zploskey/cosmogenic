#!/usr/bin/python

import sample
import muon
import nuclide as n

if __name__ == '__main__':
    s = sample.Sample(rho=2.67, h=1000, lat=65, shielding=1.0, z=1)
    print "Total muon flux: %f " % muon.tot_mu_flux(s) 