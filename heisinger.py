import numpy as np
import matplotlib.pyplot as plt

import muon

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
h = np.exp(np.arange(0, np.log(1e5), np.log(1e5) / 250.0)) # hg cm-2
z = h * 100.0 # convert to g cm-2 from hg cm-2
phi_v = muon.phi_vert_slhl(z)
plt.loglog(h, phi_v)
plt.xlim((1, 1e5))
plt.ylim((1e-14, 1e-1))
plt.xlabel('lithospheric depth h [hg/cm2]')
plt.ylabel('phi_v (h) [cm-2 s-1 sr-1]')

plt.figure(5)
h = np.arange(1, 1e4, 1)
z = h * 100.0
ebar = muon.ebar(z)
plt.loglog(h, ebar)
plt.xlim((1, 1e4))
plt.ylim((1, 1e3))
plt.ylabel('Mean E(h)')
plt.xlabel('lithospheric depth h [hg/cm2]')

plt.figure(6)
z = np.arange(0,100000,1) # g / cm2
beta = muon.beta(z)
h = z / 100.0 # convert to hg / cm2 ... Heisinger paper has typo
plt.semilogx(h, beta) 
plt.xlim((1, 1000))
plt.xlabel('lithospheric depth [hg/cm2]')
plt.ylim((0.82, 0.96))
plt.ylabel('Beta')

plt.show()

