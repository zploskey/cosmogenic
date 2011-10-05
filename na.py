"""Implementation of the Neighborhood Algorithm

Author: Zach Ploskey (zploskey@uw.edu), University of Washington

Implementation of the Neighborhood Algorithm after Sambridge (1999)
in Geophys. J. Int.    
"""

from __future__ import division

import time
import copy

import numpy as np

class NASampler(object):
    """ Sample a parameter space using the Neighborhood Algorithm."""
    
    def __init__(self, ns, nr, fn, lo_lim, hi_lim, tol=None, max_eval=10000):        
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
        self.fn = copy.deepcopy(fn)   # function to minimize
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
        self.max_eval = max_eval
        
    # functions to make this pickleable
    def __getstate__(self):
        state = copy.deepcopy(self.__dict__)
        return state

    def __setstate__(self, state):
        self.__dict__ = state

    def best_models(self):
        """Row matrix of nr best models."""
        return self.m[self.lowest_idxs]
    
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
        while (n_eval < self.max_eval) and self.n_fitting < n:
            if it != 1 and self.np != 0:
                # increase size of m
                self.m = np.vstack((self.m, self.select_new_models()))
                self.misfit = np.hstack((self.misfit, np.zeros(self.ns)))
            # calculate the misfit function for our ns models
            # record the best (lowest) ones so far
            for j in range(self.ns):
                idx = self.np + j
                print "Got to iteration", str(it) + ", index", idx
                self.misfit[idx] = self.fn(self.m[idx])
                if self.misfit[idx] < max_chosen: 
                    old_low_idx = np.where(self.chosen_misfits ==
                                              max_chosen)[0][0]
                    self.chosen_misfits[old_low_idx] = self.misfit[idx]
                    self.lowest_idxs[old_low_idx] = idx
                    max_chosen = max(self.chosen_misfits)
                if self.misfit[idx] < self.tol:
                    self.n_fitting += 1
            self.np += self.ns
            i += 1
        end_time = time.time()
        print 'End time:', time.asctime(time.localtime(end_time))
        elapsed_time = end_time - start_time
        print 'Took', elapsed_time, 's (', elapsed_time % 60, 'min,', (elapsed_time 
              % 60), 's)'
        return True

    def generate_random_models(self):
        """Generates a row matrix of random models in the parameter space."""
        rands = np.random.random_sample((self.ns, self.m_len))
        model_range = np.empty_like(rands)
        lows = np.empty_like(rands)
        model_range[:] = self.hi_lim - self.lo_lim
        lows[:] = self.lo_lim
        models = model_range * rands + lows
        return models

    def select_new_models(self):
        """ Returns a 2D ndarray where each row is newly selected model.
        
        Selects new models using the Gibbs Sampler method from the 
        Voronoi cells of the best models found so far.
        """
        m_len = self.m_len
        chosen_models = self.best_models()
        new_models = np.zeros((self.ns, m_len))
        sample_idx = 0
        # Loop through all the voronoi cells
        for chosen_idx, vk in enumerate(chosen_models):
            n_take = int(np.floor(self.ns / self.nr))
            if chosen_idx == 0:
                # Give any remaining samples to our best model
                n_take += int(np.floor(self.ns % self.nr))
            k = self.lowest_idxs[chosen_idx]
            for s in range(n_take):
                # Current random walk location, start at voronoi cell node vk
                xA = vk.copy()
                # Iterate through axes in random order, doing a random walk
                component = np.random.permutation(m_len)
                # Vector of perpendicular distances to cell boundaries along the
                # current axis (initially from vk)
                notk = np.where(np.arange(self.np) != k)
                c0 = component[0]
                d2 = (self.m[notk, c0] - vk[c0]) ** 2
                dk2 = 0
                for idx, i in enumerate(component):
                    # Find distances along axis i to all cell edges
                    vj = self.m[notk, i]
                    x = 0.5 * (vk[i] + vj + (dk2 - d2) / (vk[i] - vj))
                    # Find the 2 closest points to our chosen node on each side
                    li = np.max(np.hstack((self.lo_lim, x[x <= xA[i]])))
                    ui = np.min(np.hstack((self.hi_lim, x[x > xA[i]])))
                    # Randomly sample the interval and move there
                    xB = xA.copy()
                    xB[i] = (ui - li) * np.random.random_sample() + li
                    if idx != m_len - 1:
                        # We're not on the last walk step.
                        # Update dj2 for next axis
                        next_i = component[idx + 1]
                        d2 += (vj - xB[i]) ** 2 - (self.m[notk, next_i] 
                               - xB[next_i]) ** 2
                        dk2 += (vk[i] - xB[i]) ** 2 - (self.m[notk, next_i] 
                               - xB[next_i]) ** 2
                    xA = xB.copy()
                new_models[sample_idx] = xA
                sample_idx += 1
        return new_models
