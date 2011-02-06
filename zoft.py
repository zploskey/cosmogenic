#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

n = 20 # number of glaciations

avg_intergl = 80 # kyr
sig_intergl = avg_intergl * 0.3
avg_gl = 20 # kyr
sig_gl = avg_gl * 0.3
t_hol = 15 # kyr
sig_t_hol = t_hol * 0.2
dz = 1.5 # m
sig_dz = dz * 0.4

t_intergl = np.abs(np.random.normal(avg_intergl, sig_intergl, n))
t_gl = np.abs(np.random.normal(avg_gl, sig_gl, n))

t_periods = \
    np.append([0], \
        np.append( \
            np.random.normal(t_hol, sig_t_hol), \
            np.reshape(np.column_stack((t_gl, t_intergl)), (2*n, 1))))

erosion_depths = np.abs(np.random.normal(dz, sig_dz, n))
z_thickness = \
    np.append( \
        np.append( \
            [0],
            np.reshape( \
                np.column_stack((np.zeros(n), erosion_depths)), \
                (2*n, 1))), \
        [0])

z = np.add.accumulate(z_thickness)
t = np.add.accumulate(t_periods)

eros_rate = z[-1] * 1000 / t[-1]

print "t =", t
print "z =", z
# print "erosion_depths =", erosion_depths
# print "t_periods =", t_periods[1:]
np.savetxt("t_periods.txt", t_periods[1:])
np.savetxt("erosion_depths.txt", erosion_depths)
np.savetxt("erosion_rate.txt", np.array([eros_rate, 0]))
print "erosion rate [m/Myr] =", eros_rate

plt.plot(t, z, linewidth=2.0)
ax = plt.gca()
ax.invert_xaxis()
ax.invert_yaxis()
plt.xlabel('Time [kyr]')
plt.ylabel('Depth [m]')
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.15)
plt.show()
