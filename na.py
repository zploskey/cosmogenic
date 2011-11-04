"""Implementation of the Neighborhood Algorithm

Author: Zach Ploskey (zploskey@uw.edu), University of Washington

Implementation of the Neighborhood Algorithm after Sambridge (1999)
in Geophys. J. Int.    
"""

from __future__ import division

import time
import os

import numpy as np
import scipy as sp

from joblib import Parallel, delayed

from walk import walk_wrapper

class NASampler(object):
    """ Sample a parameter space using the Neighborhood Algorithm."""
    
    def __init__(self, ns, nr, fn, lo_lim, hi_lim, tol=None, min_eval=10000):        
        """Create an NASampler to sample a parameter space using the
        Neighborhood algorithm.
        
        Keyword Arguments:
        ns     -- # of samples to take during each iteration
        nr     -- # of best samples whose Voronoi cells should be sampled
        fn     -- Objective function f(x) where x is an ndarray
        lo_lim -- Lower limit of the search space (size of x)
        hi_lim -- Upper limit of the search space
        tol    -- Minimum value of f that is an acceptable solution.
                  Defaults to len(lo_lim) if not supplied.
        """
        lo_lim = np.atleast_1d(lo_lim)
        hi_lim = np.atleast_1d(hi_lim)
        assert hi_lim.shape == lo_lim.shape, 'Limits must have the same shape.'
        self.ns = ns # number of models for each step
        self.nr = nr # number of Voronoi cells to explore in each step
        self.fn = fn # function to minimize
        self.np = 0  # number of previous samples
        self.lo_lim = lo_lim
        self.hi_lim = hi_lim
        self.m_len = len(lo_lim) # model length
        self.m = self.generate_random_models()
        self.misfit = np.zeros(ns)
        self.chosen_misfits = np.ones(nr) * np.inf
        self.lowest_idxs = -1 * np.ones(nr, dtype=np.int)
        self.tol = self.m_len if tol == None else tol
        self.n_fitting = 0
        self.min_eval = min_eval
        
    # functions to make this pickleable
    #def __getstate__(self):
    #    state = copy.deepcopy(self.__dict__)
    #   return state

    #def __setstate__(self, state):
    #    self.__dict__ = state

    def best_nr_models(self):
        """Row matrix of nr best models."""
        return self.m[self.lowest_idxs]
        
    def best_models(self, n):
        """Return the n best models and their misfits."""
        idx = np.argsort(self.misfit)[:n]
        return self.m[idx], self.misfit[idx], idx
    
    def fitting_models(self, tol=None):
        """Row matrix of models that fit better than tol.
        
        If tol is not supplied it defaults to the attribute tol, which by
        default is the length of a model vector.
        """
        if tol == None:
            tol = self.tol
        fit_idx = self.misfit < tol
        return (self.m[fit_idx], self.misfit[fit_idx])
    
    def generate_ensemble(self, n):
        """Generate an ensemble of at least n models that fit to tolerance.
        
        Keyword Arguments:
        n -- # of models to find
        """
        assert n > 0, 'n must be at least 1'        
        
        start_time = time.time()
        print 'Start time:', time.asctime(time.localtime(start_time))
        max_chosen = max(self.chosen_misfits)
        it = 1 # iteration number
        idx = 0
        while (self.np < self.min_eval) and self.n_fitting < n:
            if it != 1 and self.np != 0:
                # increase size of m
                self.m = np.vstack((self.m, self.select_new_models()))
                self.misfit = np.hstack((self.misfit, np.zeros(self.ns)))
            # calculate the misfit function for our ns models
            # record the best (lowest) ones so far
            for j in range(self.ns):
                print "Iteration %i, model index %i. " % (it, idx), 
                self.misfit[idx] = self.fn(self.m[idx])
                if self.misfit[idx] < max_chosen: 
                    old_low_idx = np.where(self.chosen_misfits ==
                                              max_chosen)[0][0]
                    self.chosen_misfits[old_low_idx] = self.misfit[idx]
                    self.lowest_idxs[old_low_idx] = idx
                    max_chosen = np.max(self.chosen_misfits)
                if self.misfit[idx] < self.tol:
                    self.n_fitting += 1
                    print "FIT!",
                print "\n",
                idx += 1
            self.np += self.ns
            it += 1
        end_time = time.time()
        print 'End time:', time.asctime(time.localtime(end_time))
        runtime = end_time - start_time
        print 'Inversion took %i minutes and %0.2f seconds.' % (runtime / 60,
              runtime % 60)
        return True

    def generate_random_models(self):
        """Generates a row matrix of random models in the parameter space."""
        rands = np.random.random_sample((self.ns, self.m_len))
        lows = np.empty_like(rands)
        model_range = self.hi_lim - self.lo_lim
        lows[:] = self.lo_lim
        models = model_range * rands + lows
        return models

    def select_new_models(self):
        """ Returns a 2D ndarray where each row is newly selected model.
        
        Selects new models using the Gibbs Sampler method from the 
        Voronoi cells of the best models found so far.
        """
        m_len = self.m_len
        chosen_models = self.best_nr_models()
        new_models = np.zeros((self.ns, m_len))
        m = self.m
        sample_idx = 0
        # Loop through all the voronoi cells
        for chosen_idx, vk in enumerate(chosen_models):
            n_take = int(np.floor(self.ns / self.nr))
            if chosen_idx == 0:
                # Give any remaining samples to our best model
                n_take += int(np.floor(self.ns % self.nr))
            k = self.lowest_idxs[chosen_idx]
            notk = np.where(np.arange(self.np) != k)
            d2_prev_ax = np.zeros(self.np)
            # Vector of perpendicular distances to cell boundaries along the
            # current axis (initially from vk)
            d2 = ((m - vk) ** 2).sum(axis=1)
            for s in range(n_take):
                # Current random walk location, start at voronoi cell node vk
                xA = vk.copy()
                # Iterate through axes in random order, doing a random walk
                component = np.random.permutation(m_len)
                for idx, i in enumerate(component):
                    # keep track of squared perpendicular distances to axis
                    d2_this_ax = (m[:, i] - xA[i]) ** 2
                    d2 += d2_prev_ax - d2_this_ax
                    dk2 = d2[chosen_idx]
                    # Find distances along axis i to all cell edges
                    vji = m[notk, i]
                    x = 0.5 * (vk[i] + vji + (dk2 - d2[notk]) / (vk[i] - vji))
                    # Find the 2 closest points to our chosen node on each side
                    li = np.max(np.hstack((self.lo_lim, x[x <= xA[i]])))
                    ui = np.min(np.hstack((self.hi_lim, x[x >= xA[i]])))
                    # Randomly sample the interval and move there
                    xA[i] = (ui - li) * np.random.random_sample() + li
                    d2_prev_ax = d2_this_ax
                
                new_models[sample_idx] = xA.copy()
                sample_idx += 1
        return new_models

def resample(m, x2v, dof, Nw, n, lup, n_jobs=1):
    """Resample an ensemble of models.    
    
    Resample an ensemble of models according to the method described
    by Malcolm Sambridge (1999) Geophysical inversion with a neighbourhood
    algorithm  --II. Appraising the ensemble, Geophys. J. Int., 727--746.    
    
    Parameters:
        m: Row matrix of model vectors
        x2v: reduced chi squared for each model
        dof: number of degrees of freedom in the problem
        Nw: number of random walks to do starting from Nw best models
        n: number of samples to collect along each walk
        lup: tuple of lower and upper bounds of the space
        n_jobs: number of parallel jobs to run. Typically this should be set to
                the number of processors (or virtual processors)
    """
    
    # Number of models in the ensemble
    Ne = m.shape[0]
    # Number of dimensions in the model space
    d = m.shape[1]
    
    walkstart_idxs = np.argsort(x2v)[0:Nw]
    logP = -(0.5 * dof) * x2v
    start_time = time.time()
    print 'Start time:', time.asctime(time.localtime(start_time))
    print 'Generating list of random walks to perform...'
    walklist = ((n, m, wi, logP, lup, i) 
                 for i, wi in enumerate(walkstart_idxs))
    print 'Importance resampling with %i simultaneous walks...' % n_jobs
    res = Parallel(n_jobs=n_jobs, verbose=100)(
                                    delayed(walk_wrapper)(w) for w in walklist)
    assert sum(res) == Nw, 'One of the parallel jobs failed!'

    # single process for loop for debugging purposes
    #for i in range(Nw):
    #    _walk(n, m, walkstart_idxs[i], logP, lup, i)

    print 'Finished importance sampling at', time.asctime()
    print 'Recombining samples from each walk...'
    mr = np.zeros((Nw * n, d))
    for i in range(Nw):
        cur_file = '_walk_tmp' + str(i) + '.npy'
        mr[i*n:(i+1)*n, :] = np.load(cur_file)
        os.remove(cur_file)
    print 'FINISHED'
    end_time = time.time()
    runtime = end_time - start_time
    print 'End time:', time.asctime(time.localtime(end_time))
    print 'Resampling took %i hours, %i minutes and %0.2f seconds.' % (
                runtime / 3600,
                runtime / 60,
                runtime % 60)
    return mr

def _walk(n, m, start_idx, logP, lup, walk_num):
    """ Calculates an average value along its walk. """
    low, up = lup

    # Number of models in the ensemble
    Ne = m.shape[0]
    # Number of dimensions in the model space
    d = m.shape[1]

    resampled_models = np.empty((n, d))
    
    # don't dare overwrite our ensemble data, copy our walk position
    xp = m[start_idx].copy() 
    #mean = np.zeros(d)
    d2_prev_ax = np.zeros(Ne)
    d2 = ((m - xp) ** 2).sum(axis=1)
    cell_idx = np.argmin(d2)
    # for each sample we take along our walk
    for i in range(n):
        # first, randomly select order of the axes to walk
        axes = np.random.permutation(d)
        for ax in axes:
            # keep track of squared perpendicular distances to axis
            d2_this_ax = (m[:, ax] - xp[ax]) ** 2
            d2 += d2_prev_ax - d2_this_ax
            ints, idxs = _calc_conditional(xp, ax, m, d2, lup)
            # Calculate the conditional ppd along this axis
            Pmax = np.max(logP[idxs])
            
            accepted = False
            while not accepted:
                # generate proposed random deviate along this axis            
                dev = np.random.uniform(low[ax], up[ax])
                # determine voronoi cell index the deviate falls into
                for ii, intersection in enumerate(ints):
                    if dev < intersection:
                        cell_idx = idxs[ii]
                        break
                # get that cell's log relative probability and max along axis
                logPxp = logP[cell_idx]
                logPmax = np.max(logP[idxs])
                # generate another deviate between 0 and 1
                r = np.random.uniform()
                accepted = np.log(r) <= (logPxp - logPmax)
            # we accepted a model, this is our new position in the walk
            xp[ax] = dev
            d2_prev_ax = d2_this_ax
        resampled_models[i, :] = xp.copy()
    np.save('_walk_tmp%i' % walk_num, resampled_models)
    print 'Saved walk #%i' % walk_num
    return True

def _calc_conditional(xp, ax, m, d2, lup):
    """Calculate the conditional probability distribution along the axis. """
    xA = xp.copy()
    low_lim, up_lim = lup
    xA[ax] = low_lim[ax]
    # calculate distances to the lowest point on this axis
    d2lowedge = d2 + (m[:, ax] - xA[ax]) ** 2
    cell_idx = np.argmin(d2lowedge)
    
    intxs = []
    idxs = [cell_idx]
    while cell_idx != -1:
        prev_cell_idx = cell_idx
        intx, cell_idx = _calc_upper_intersect(xA, ax, m, cell_idx, d2, 
                                               up_lim[ax])
        intxs.append(intx)
        idxs.append(cell_idx)
        xA[ax] = intx # m[cell_idx, ax]
        # Occasionally due to rounding error we do not reach the upper limit
        # but we should then be extremely close. If that is the case then we've
        # remained in the same cell and have to manually break the loop.
        if cell_idx == prev_cell_idx:
            # make the upper boundary the true upper limit
            print 'Encountered repeat cell'
            import pdb; pdb.set_trace()
            intxs[-1] = up_lim[ax]
            break
        if len(idxs) > 400:
            print 'Runaway conditional!'
            import pdb; pdb.set_trace()
            print 'Should be tracing here'
            break
    idxs.pop()
    return intxs, idxs

def _calc_upper_intersect(xp, ax, m, cell_idx, d2, up_lim):
    """Calculate the upper intersection of the cell and the axis. """
    vki = m[cell_idx, ax]
    vji = m[:, ax]
    # squared perpendicular distance to the axis from vk
    dk2 = d2[cell_idx]
    # calculate upper intersection point
    x = 0.5 * (vki + vji + (dk2 - d2) / (vki - vji))
    x[cell_idx] = np.inf
    x[x <= xp[ax]] = np.inf
    tmp = np.hstack((up_lim, x))
    hi_idx = np.argmin(tmp) - 1
    xu = tmp[hi_idx + 1]
    return (xu, hi_idx)

# Potentially useful for search phase, try to work this in above
def _calc_intersects(xp, ax, m, cell_idx, d2, lup):
    """Finds upper and lower intersection of axis through xp along axis ax. """
    low, up = lup
    vki = m[cell_idx, ax]
    vji = m[:, ax]
    # squared perpendicular distance to the axis from vk
    dk2 = d2[cell_idx]
    x = 0.5 * (vki + vji + (dk2 - d2) / (vki - vji))
    # calculate lower intersection
    tmp = np.hstack((low[ax], x[x <= xp[ax]]))
    xl = tmp[np.argmax(tmp)]
    # calculate upper intersection
    tmp = np.hstack((up[ax], x[x >= xp[ax]]))
    hi_idx = np.argmin(tmp)
    xu = tmp[hi_idx]
    return xl, xu
    
