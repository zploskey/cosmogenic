#!/usr/bin/python

import scaling
import nuclide

class Sample:
    """
    Represents information for a collected cosmogenic nuclide sample
    """
    def __init__(self, rho, h, lat, z=0, shielding=1, n=nuclide.Be10Qtz()):
        self.rho = rho # density (g/cm2)
        self.h = h # elevation (m)
        self.lat = lat # latitude (deg) 
        self.shielding = shielding # shielding correction factor
        self.z = z # depth below surface (g/cm2)
        self.n = n # nuclide object
        self.f_sp_scaling = scaling.stone2000(lat=lat, alt=h)
