# cython: profile=True
# cython: cdivision=True

from __future__ import division

from libc.math cimport log

import numpy as np
cimport numpy as np

import cython
cimport cython

DTYPE = float
ctypedef cython.floating DTYPE_t

@cython.profile(False)
cdef inline int argmin(np.ndarray[DTYPE_t, ndim=1] a):
    cdef int i, lowi = 0
    for i in range(a.shape[0]):
        if a[i] < a[lowi]:
            lowi = i

    return lowi


cdef tuple _lower_bounds(DTYPE_t xA, int i,
        np.ndarray[DTYPE_t, ndim=2] v, int k,
        np.ndarray[DTYPE_t, ndim=1] d2, DTYPE_t li):

    cdef list lower_bounds = []
    cdef list lower_idxs = []
    cdef DTYPE_t xj, dx, x
    cdef int lower_bound_idx, j

    while True:
        lower_bound_idx = -1 # mark so we know if we never find one

        # xj is next possible boundary position.
        # Shrink in from the lower boundary as points qualify
        xj = li

        for j in range(v.shape[0]): # iterate over models
            dx = v[k, i] - v[j, i]

            if j == k or dx == 0.0: # avoid division by zero
                continue

            # calculate the position along the axis
            x = 0.5 * (v[k, i] - v[j, i] - (d2[k] - d2[j]) / dx)

            if x >= xj and x <= xA:
                # good boundary position, move to it
                xj = x
                lower_bound_idx = j

        if lower_bound_idx == -1:
            break

        lower_bounds.append(xj)
        lower_idxs.append(lower_bound_idx)
        k = lower_bound_idx
        xA = xj # move the current position to the last boundary

    lower_bounds.reverse()
    lower_idxs.reverse()

    return lower_bounds, lower_idxs


cdef tuple _upper_bounds(DTYPE_t xA, int i,
        np.ndarray[DTYPE_t, ndim=2] v, int k,
        np.ndarray[DTYPE_t, ndim=1] d2, DTYPE_t ui):

    cdef list upper_bounds = []
    cdef list upper_idxs = [k] # Include the starting central index k.
                               # Only do this for the upper bounds.
    cdef DTYPE_t xl, dx, x
    cdef int upper_bound_idx, j

    while True:
        upper_bound_idx = -1 # mark so we know if we never find one

        # xl is next possible boundary position.
        # Shrink in from the upper boundary as points qualify
        xl = ui

        for j in range(v.shape[0]): # iterate over models
            dx = v[k, i] - v[j, i]

            if j == k or dx == 0.0: # avoid division by zero
                continue

            # calculate the position along the axis
            x = 0.5 * (v[k, i] - v[j, i] - (d2[k] - d2[j]) / dx)

            if x <= xl and x >= xA:
                # good boundary position, move to it
                xl = x
                upper_bound_idx = j

        if upper_bound_idx == -1:
            break

        upper_bounds.append(xl)
        upper_idxs.append(upper_bound_idx)
        k = upper_bound_idx
        xA = xl # move the current position to the last boundary

    upper_bounds.append(ui)
    return upper_bounds, upper_idxs


cdef tuple _conditional_bounds(DTYPE_t xA, int i,
        np.ndarray[DTYPE_t, ndim=2] v, int k, np.ndarray[DTYPE_t, ndim=1] d2,
        DTYPE_t low_lim, DTYPE_t up_lim):
    """Calculate the conditional probability distribution along the axis.

    See Sambridge 1999a, p. 7-8.

    We begin at the starting point xA on the current axis ax.
    We calculate the lower and upper bound of the current cell along the axis.
    """
    cdef list lower_bounds
    cdef list upper_bounds
    cdef list lower_idxs
    cdef list upper_idxs

    # calculate the lower bounds and cell indices
    lower_bounds, lower_idxs = _lower_bounds(xA, i, v, k, d2, low_lim)

    # calculate the upper bounds and cell indices
    upper_bounds, upper_idxs = _upper_bounds(xA, i, v, k, d2, up_lim)

    # combine the upper and lower lists for bounds and indices, respectively
    cdef list bounds = lower_bounds + upper_bounds
    cdef list idxs = lower_idxs + upper_idxs

    return bounds, idxs

@cython.profile(False)
cdef inline np.ndarray[DTYPE_t, ndim=1] sse2d(
        np.ndarray[DTYPE_t, ndim=2] A,
        np.ndarray[DTYPE_t, ndim=1] b):
    #assert(A.shape[1] == b.shape[0])
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.empty(A.shape[0], dtype=DTYPE)
    cdef int j, i
    cdef DTYPE_t tmp
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            tmp = A[i, j] - b[j]
            res[i] += tmp * tmp
    return res


cpdef np.ndarray[DTYPE_t, ndim=2] _walk(
        int n, np.ndarray[DTYPE_t, ndim=2] m, int start_idx,
        np.ndarray[DTYPE_t, ndim=1] logP, tuple lup,
        int walk_num, seed):
    """ Produces models in a random walk over the initial ensemble.

    Models should have a sampling density that approximates the model space
    posterior probability distribution (PPD).
    """
    # Each walk should have an independent random state so we do not overlap
    # in the numbers generated in each walk.
    rng = np.random.RandomState(seed * (walk_num + 2))

    # Number of models in the ensemble
    cdef int Ne = m.shape[0]
    # Number of dimensions in the model space
    cdef int d = m.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] low, up, xp, d2
    cdef DTYPE_t logPmax, dev, intersection, logPxp, r
    dev = 0.0
    low, up = lup
    cdef np.ndarray[DTYPE_t, ndim=2] resampled_models = (
            np.empty((n, d), dtype=m.dtype))

    # don't dare overwrite our ensemble data, copy our walk position
    xp = m[start_idx].copy() # our current position in the walk
    d2 = sse2d(m, xp)
    cdef int i, j, ii, ax, prev_ax = 0, cell_idx = argmin(d2)
    cdef bint accepted
    for i in range(n):
        for ax in range(d):
            # keep track of squared perpendicular distances to axis
            if i != 0:
                for j in range(Ne):
                    d2[j] += (m[j,prev_ax] - dev)**2 - (m[j,ax] - xp[ax])**2

            ints, idxs = _conditional_bounds(xp[ax], ax, m, start_idx, d2,
                    low[ax], up[ax])

            # Calculate the conditional ppd along this axis
            logPmax = np.max(logP[idxs])

            accepted = False
            while not accepted:
                # generate proposed random deviate along this axis
                dev = rng.uniform(low[ax], up[ax])
                # determine voronoi cell index the deviate falls into
                n_ints = len(ints)
                for ii in range(n_ints):
                    if dev < ints[ii]:
                        cell_idx = idxs[ii]
                        break

                # get that cell's log relative probability and max along axis
                logPxp = logP[cell_idx]
                # generate another random deviate between 0 and 1
                r = rng.uniform()
                accepted = log(r) <= (logPxp - logPmax)

            # we accepted a model, this is our new position in the walk
            xp[ax] = dev
            prev_ax = ax

        resampled_models[i,:] = xp

    return resampled_models
