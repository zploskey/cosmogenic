from __future__ import division, print_function, unicode_literals

import numpy as np

import scipy.constants as const

N_A = const.N_A
A_Be = 9.012182  # g / mol Be


def N10(R10to9, Mq=0.0, Mc=0.0, n10b=0.0, unc_R10to9=0.0, unc_Mc=0.0, unc_n10b=0.0):
    
    # If no quartz mass is provided,
    # calculate a number of atoms instead of a concentration.
    Mq = np.atleast_1d(Mq)
    Mq[Mq == 0.0] = 1.0
    Mq = np.array(Mq, dtype=np.float)
    
    N10 = ((R10to9 * Mc * N_A / A_Be) - n10b) / Mq

    # Standard error propagation, ignoring error in quartz weighing.
    A = unc_R10to9 / Mq * (Mc * N_A / A_Be)
    B = unc_Mc / Mq * (R10to9 * N_A / A_Be)
    # special case if unc_n10b is a pandas series
    if hasattr(unc_n10b, 'values'):
        unc_n10b = unc_n10b.values
    C = unc_n10b * -1 / Mq
    unc_N10 = np.sqrt(A ** 2 + B ** 2 + C ** 2)

    return (N10, unc_N10)
