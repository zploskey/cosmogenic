import numpy as np

# Rough fit to Greg Balco's error data on his blog.

#ends = np.array([[x_lo, ], [x_hi, 0.01]])
x = np.array([1e5, 1.1e8])
y = np.array([0.1, 0.01])
line = np.polyfit(np.log10(x), np.log10(y), 1)
print 10**line[1]
