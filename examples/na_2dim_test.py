import numpy as np
import matplotlib.pyplot as plt

from cosmogenic.na import NASampler

def func(x):
    return x[0] ** 2 + x[1] ** 2

lo = -5
hi = 5

ns = 20
nr = 3
ndim = 2
lo = np.ones(ndim) * lo
hi = np.ones(ndim) * hi
sampler = NASampler(func, ns, nr, lo, hi, d=2) 

print("Neighborhood Algorithm Minima Finder")
print("Finding the local minima using the neighborhood algorithm...")

sampler.generate_ensemble()

plt.figure(1)
plt.title('Sampling locations')
plt.plot(sampler.m[:, 0], sampler.m[:, 1], '.')
plt.ylabel('$m_1$')
plt.ylim((lo[0], hi[0]))
plt.xlabel('$m_0$')
plt.xlim((lo[0], hi[0]))
plt.show()
