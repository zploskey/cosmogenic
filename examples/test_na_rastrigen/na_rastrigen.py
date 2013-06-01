
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import cosmic.na as na


def rastrigin(x):
    n = x.size
    return 10 * n + np.sum((x**2 - 10 * np.cos(2 * np.pi * x)))

def rastrigin2d(x, y):
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) 
                                    + np.cos(2 * np.pi * y))

def plot_rastrigin(n):  
    fig = plt.figure()
    fmax = 5.12
    fmin = -5.12
    x = np.linspace(fmin, fmax, n)
    y = np.linspace(fmin, fmax, n)
    xx, yy = np.meshgrid(x, y)
    pts = np.dstack((xx, yy)).reshape((xx.size, 2))
    rast = rastrigin2d(xx, yy)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(pts[:,0], pts[:,1], rast, rstride=1, cstride=1,
                        cmap=cm.jet, linewidth=0, antialiased=False)
    fig.colorbar(surf)
    plt.show()

conf = {
    "description": "Test of NA on the Rastrigen function",
    "ns": 100, 
    "nr": 100,
    "n_initial": 5000,
    "lo_lim": -5.12, 
    "hi_lim": 5.12, 
    "d": 2,
    "ne": 10000,
    "shape": (2, 1),
    "m_true": np.array([0.0, 0.0]),
    # resampling
    "dof": 1,
    'Nw': 8, # 100, # 8, # Number of walks
    'pts_per_walk': 100, # 12500, # should give 100000 resampled pts
}

#mp, misfit = na.search(rastrigin, conf)
#min_idx = misfit.argmin()
#print 'minimum:', misfit[min_idx]
#print 'at:', mp[min_idx] 

#na.run(rastrigin, conf)

mp = np.genfromtxt('m.dat')
na.resample_and_plot(conf, mp)
