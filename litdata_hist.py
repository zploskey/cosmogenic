# litdata_hist

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Get literature data, currently in percent inherited
data = np.genfromtxt("inherit_litdata.txt")

plt.hist(data, 14, range=(0,1400), facecolor='0.8')
plt.xlim(-50, 1450)
plt.ylim(0, 65)
plt.xlabel("% Inheritance")
plt.ylabel("# of Samples")
plt.show()
