from __future__ import division, print_function, unicode_literals

import numpy as np

import scipy.constants as const

N_A = const.N_A
A_Be = 9.012182  # g / mol Be


def N10(R10to9, Mq, Mc, n10b=0.0, unc_R10to9=0.0, unc_Mc=0.0, unc_n10b=0.0):

    N10 = ((R10to9 * Mc * N_A / A_Be) - n10b) / Mq

    # standard error propagation
    # we assume error in quartz weighing is so small as to not matter
    A = unc_R10to9 / Mq * (Mc * N_A / A_Be)
    B = unc_Mc / Mq * (R10to9 * N_A / A_Be)
    C = unc_n10b * -1 / Mq
    unc_N10 = np.sqrt(A ** 2 + B ** 2 + C ** 2)

    return (N10, unc_N10)
