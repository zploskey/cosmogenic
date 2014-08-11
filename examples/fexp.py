import matplotlib.pyplot as plt
import numpy as np

from cosmogenic import nuclide, sim

# rock density
rho = 2.67

# number of glaciations
n = 20

# depth eroded per glaciation (500 cm in g/cm**2)
dz = 500 * rho * np.ones(n)

# predict concentrations for these depths (to 8 m) 
z0 = np.linspace(0, 800 * rho)

be10 = nuclide.Be10Qtz()

# most recent exposure time
tH = 17000

# holocene exposure only
Nhol = sim.expose(z=z0, t_init=tH, t_final=0.0, n=be10)

# vary glaciation length (t_gl), keeping glacial cycle length constant
N10 = sim.multiglaciate(dz=dz, t_gl=10000, t_intergl=90000, t_postgl=tH,
                        z=z0, n=be10, n_gl=n)
N50 = sim.multiglaciate(dz=dz, t_gl=50000, t_intergl=50000, t_postgl=tH,
                        z=z0, n=be10, n_gl=n)
N90 = sim.multiglaciate(dz=dz, t_gl=90000, t_intergl=10000, t_postgl=tH,
                        z=z0, n=be10, n_gl=n)

# sample depths in meters
z = z0 / 100.0 / rho

ax1 = plt.subplot(121)
plt.semilogx(Nhol, z, 'k', lw=2)
plt.semilogx(N10, z, N50, z, N90, z, lw=2)
ax1.invert_yaxis()
plt.xlabel('[Be-10] (atoms / g quartz)')
plt.ylabel('Depth (m)')
plt.legend(('Simple exposure', 'f_exp = 0.9', 'f_exp = 0.5', 'f_exp = 0.1'),
    loc='upper left')

# alternative erosion rate
dz_b = 100 * rho * np.ones(n)
N10b = sim.multiglaciate(dz=dz_b, t_gl=10000, t_intergl=90000, t_postgl=tH,
                        z=z0, n=be10, n_gl=n)
N50b = sim.multiglaciate(dz=dz_b, t_gl=50000, t_intergl=50000, t_postgl=tH,
                        z=z0, n=be10, n_gl=n)
N90b = sim.multiglaciate(dz=dz_b, t_gl=90000, t_intergl=10000, t_postgl=tH,
                        z=z0, n=be10, n_gl=n)

ax2 = plt.subplot(122)
plt.semilogx(Nhol, z, 'k', lw=2)
plt.semilogx(N10b, z, N50b, z, N90b, z, lw=2)
ax2.invert_yaxis()
plt.show()
