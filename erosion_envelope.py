import time

import numpy as np
import matplotlib.pyplot as plt

import sim
import nuclide

def get_zoft(tHol):
    times = np.random.rand(2*n) * (tsim - tHol)
    times.sort() # do an in-place sort of the data
    times += tHol
    times[0] = tHol
    times = np.append(times, tsim)
    times = np.append(0, times)
    tmpz = np.random.rand(n) * z0
    tmpz[-1] = z0
    tmpz.sort()
    depths = np.zeros(2*n)
    for i in range(n):
        depths[2*i] = tmpz[i]
        depths[2*i+1] = tmpz[i]
    depths = np.append([0,0], depths)
    return (depths, times)

if __name__ == '__main__':

    start_time = time.time()
    tsim = 2.0 # Myr
    er = 25.0  # m / Myr
    z0 = er * tsim # initial depth
    n = 20 # number of events
    lat = 65
    h = 1
    rho = 2.67

    tHol = 0.014 # Myr postglacial exposure

    be10 = nuclide.Be10Qtz()

    plt.figure(1)

    ncurves = 1
    depths = np.zeros((ncurves, 2*(n+1)))
    times  = np.zeros((ncurves, 2*(n+1)))
    for i in range(ncurves):
        depths[i], times[i] = get_zoft(tHol)
        plt.plot(times[i], depths[i], lw=1)
    plt.xlim((-0.1,tsim+0.1))
    plt.ylim((-1,z0+1))
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    plt.figure(2)

    z = np.linspace(0,20*100.0*rho,50)
    ndzs = 0
    conc = np.zeros((ncurves,len(z)))
    for i in range(ncurves):
        d = depths[i][1:-1]
        ndzs = len(d) - 1
        dz = np.zeros(ndzs)
        for j in range(ndzs):
            dz[j] = d[j+1] - d[j]
        dz = dz.copy()[0::2]
        t = 1e6 * times[i][1:]
        conc[i] = sim.fwd_profile(z, dz, t, be10, h, lat)
        plt.semilogx(conc[i], z / 100.0 / rho, lw=1)

    plt.gca().invert_yaxis()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print "Took {0} seconds.".format(total_time)
    
    plt.show()

    #test = sim.fwd_profile(z, dzs, t, be10, h, lat)

    #for i in range(ncurves):




