import numpy as np

# Rough fit to Greg Balco's error data on his blog.

x_hi = 1e8
x_lo = 1e4
ends = np.array([[1e4, 6e-2], [1e8, 4e-3]])
x = np.array([x_lo, x_hi])
y = ends[:,1]
line = np.polyfit(np.log10(x), np.log10(y), 1)
print 10**line[1]
