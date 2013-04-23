""" Plot production ratio of Al26 and Be10 """

import numpy as np
import pylab

import nuclide
import production

def plot_production_ratio(n1, n2, max_depth_gcm2, npts=200):
    """ Plot the production ratio of two nuclides with depth """
    z = np.linspace(start=0, stop=max_depth_gcm2, num=npts)
    alt = 0
    lat = 75
    p1 = production.P_tot(z, alt, lat, n1)
    p2 = production.P_tot(z, alt, lat, n2)
    
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.plot(p1 / p2, z)
    ax.invert_yaxis()
    ax.set_xlabel('n1/n2')
    ax.set_ylabel(r'Depth (g/cm^2)')
    pylab.show()
    return (fig, ax)

if __name__ == "__main__":
    import nuclide
    plot_production_ratio(nuclide.Al26Qtz(), nuclide.Be10Qtz(), 10680.0)