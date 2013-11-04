from __future__ import division

import numpy as np
cimport numpy as np

from libc.math cimport log

def _calc_upper_intersect(xp, ax, m, cell_idx, d2, up_lim):
    """Calculate the upper intersection of the cell and the axis. 
    
    xp: the previous model
    ax: the current axis (int)
    m:  all the models
    cell_idx: index of the current cell
    d2: squared distances
    """
    vki = m[cell_idx][ax]
    vji = m[:,ax]
    dx = vki - vji
    # squared perpendicular distance to the axis from vk
    dk2 = d2[cell_idx]
    # calculate upper intersection point
    x = 0.5 * (vki + vji + (dk2 - d2) / dx)
    x[cell_idx] = np.inf
    x[x <= xp[ax]] = np.inf

    hi_idx = np.argmin(x)
    xu = x[hi_idx]
    if up_lim <= xu:
        xu = up_lim
        hi_idx = -1

    return (xu, hi_idx)


def _calc_conditional(xp, ax, m, d2, low_lim, up_lim):
    """Calculate the conditional probability distribution along the axis. """
    xA = xp.copy()
    xA[ax] = low_lim
    # calculate distances to the lowest point on this axis
    m_thisax = m[:,ax]
    xA_thisax = xA[ax]
    d2lowedge = d2 + (m_thisax - xA_thisax) ** 2
    cell_idx = np.argmin(d2lowedge)

    intxs = []
    idxs = [cell_idx]
    len_idxs = 1
    while cell_idx != -1:
        prev_cell_idx = cell_idx
        intx, cell_idx = _calc_upper_intersect(xA, ax, m, cell_idx, d2, up_lim)
        len_idxs += 1
        if len_idxs > m.shape[0]:
            print('Error: Runaway conditional! Conditional idx > num. models')
            import pdb
            pdb.set_trace()

        intxs.append(intx)
        idxs.append(cell_idx)
        xA[ax] = intx #  m[cell_idx, ax]
        # Occasionally due to rounding error we do not reach the upper limit
        # but we should then be extremely close. If that is the case then we've
        # remained in the same cell and have to manually break the loop.
        if cell_idx == prev_cell_idx:
            # make the upper boundary the true upper limit
            print('Warning: Encountered repeat cell.')
            intxs[-1] = up_lim
            break

    idxs.pop()
    return intxs, idxs


def _walk(n, m, start_idx, logP, lup, walk_num, seed):
    """ Produces models in a random walk over the initial ensemble.

    Models should have a sampling density that approximates the model space
    posterior probability distribution (PPD). Random numbers generated in
    the process of producing models are use an independently seeded
    RandomState object making it safe to use in parallel computations, at
    least for relatively few processes and short models runs.
    """
    # Each walk should have an independent random state so we do not overlap
    # in the numbers generated in each walk.

    if seed is not None:
        walk_seed = seed * (walk_num + 1)
    else:
        walk_seed = None

    random_state = np.random.RandomState(walk_seed)

    # Number of models in the ensemble
    Ne = m.shape[0]
    # Number of dimensions in the model space
    d = m.shape[1]

    low, up = lup

    resampled_models = np.empty((n, d))
    # don't dare overwrite our ensemble data, copy our walk position
    xp = m[start_idx].copy()
    d2_prev_ax = np.zeros(Ne)
    d2 = np.sum((m - xp) ** 2, axis=1)
    #print "d2=", d2
    cell_idx = np.argmin(d2)
    # for each sample we take along our walk
    for i in range(n):
     #   print i
        # first, randomly select order of the axes to walk
        axes = random_state.permutation(d)
        for ax in axes:
            # keep track of squared perpendicular distances to axis
            m_thisax = m[:,ax]
#            print "m_thisax=", m_thisax
            xp_thisax = xp[ax]
 #           print "xp_thisax=", xp_thisax
            d2_this_ax = (m_thisax - xp_thisax) ** 2
#            print "d2_prev_ax=", d2_prev_ax
#            print "d2_this_ax=", d2_this_ax
            d2 += d2_prev_ax - d2_this_ax
            ints, idxs = _calc_conditional(xp, ax, m, d2, low[ax], up[ax])
            # Calculate the conditional ppd along this axis
            logPmax = np.max(logP[idxs])

            accepted = False
            while not accepted:
                # generate proposed random deviate along this axis
                dev = random_state.uniform(low[ax], up[ax])
                # determine voronoi cell index the deviate falls into
                for ii, intersection in enumerate(ints):
                    if dev < intersection:
                        cell_idx = idxs[ii]
                        break
                # get that cell's log relative probability and max along axis
                logPxp = logP[cell_idx]
                # generate another deviate between 0 and 1
                r = random_state.uniform()
                accepted = log(r) <= (logPxp - logPmax)
            # we accepted a model, this is our new position in the walk
            xp[ax] = dev
            d2_prev_ax = d2_this_ax
        resampled_models[i][:] = xp.copy()

    return resampled_models


