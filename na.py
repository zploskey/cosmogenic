"""
Implementation of the Neighborhood Algorithm
See Sambridge (1999) in Geophys. J. Int.

"""
import time

import numpy
from numpy import arange, inf, ones, zeros, empty_like,  hstack, vstack

class NASampler(object):
    def __init__(self, ns, nr, f, lo_lim, hi_lim, tol=None):        
        lo_lim = numpy.atleast_1d(lo_lim)
        hi_lim = numpy.atleast_1d(hi_lim)
        assert hi_lim.shape == lo_lim.shape, 'Limits must have the same shape.'
        self.ns = ns # number of models for each step
        self.nr = nr # number of Voronoi cells to explore in each step
        self.f = f   # function to minimize
        self.np = 0  # number of previous samples
        self.lo_lim = lo_lim
        self.hi_lim = hi_lim
        self.m_len = len(lo_lim) # model length
        self.m = self.generate_random_models()
        self.misfit = zeros(ns)
        self.chosen_misfits = ones(nr) * inf
        self.lowest_idxs = -1 * ones(nr, dtype=numpy.int)
        self.tol = self.m_len if tol == None else tol
        #if tol == None:
        #    self.tol = self.m_len
        #else:
        #    self.tol = tol
        self.n_fitting = 0

    def best_models(self):
        return self.m[self.lowest_idxs]

    def generate_ensemble(self, n):
        """
        Find models that fit the data with a chi**2 <= dof
        """
        print 'Start time:', time.asctime(time.localtime(time.time()))
        max_chosen = max(self.chosen_misfits)
        self.n_fitting = 0
        i = 0
        while self.n_fitting < n:
            if i != 0:
                # increase size of m
                self.m = vstack((self.m, self.select_new_models()))
                self.misfit = hstack((self.misfit, zeros(self.ns)))
            # calculate the misfit function for our ns models
            # record the best (lowest) ones so far
            for j in range(self.ns):
                idx = self.np + j
                print "Got to iteration", str(i+1) + ", index", idx
                self.misfit[idx] = self.f(self.m[idx])
                if self.misfit[idx] < max_chosen and idx not in self.lowest_idxs:
                    old_low_idx = numpy.where(self.chosen_misfits == max_chosen)[0]
                    self.chosen_misfits[old_low_idx] = self.misfit[idx]
                    self.lowest_idxs[old_low_idx] = idx
                    max_chosen = max(self.chosen_misfits)
                if self.misfit[idx] < self.tol:
                    self.n_fitting += 1
            self.np += self.ns
            i += 1
        # for the moment we leave it to the user to extract all the best
        # fits
        print 'End time:', time.asctime(time.localtime(time.time()))
        return True

    def generate_random_models(self):
        rands = numpy.random.random_sample((self.ns, self.m_len))
        model_range = empty_like(rands)
        lows = empty_like(rands)
        model_range[:] = self.hi_lim - self.lo_lim
        lows[:] = self.lo_lim
        models = model_range * rands + lows
        return models

#    def run_for(self, iters):
#        # generate an initial set of ns models randomly in
#        # the parameter space
#        max_chosen = max(self.chosen_misfits)
#        for i in range(iters):
#            if i != 0:
#                # increase size of m
#                self.m = vstack((self.m, self.select_new_models())).copy()
#                self.misfit = hstack((self.misfit, zeros(self.ns)))
#            # calculate the misfit function for our ns models
#            # record the best (lowest) ones so far
#            for j in range(self.ns):
#                idx = self.np + j
#                print "Got to iteration", str(i+1) + ", index", idx
#                print "misfit shape is", numpy.shape(self.misfit)
#                self.misfit[idx] = self.f(self.m[idx])
#                if (self.misfit[idx] < max_chosen and (idx not in self.lowest_idxs)):
#                    old_low_idx = numpy.where(self.chosen_misfits == max_chosen)[0]
#                    self.chosen_misfits[old_low_idx] = self.misfit[idx]
#                    self.lowest_idxs[old_low_idx] = idx
#            self.np += self.ns
#        return True
    
    def select_new_models(self):
        """
        Selects a batch of new models using the Gibbs Sampler method described
        in Sambridge 1999.
        """
        m_len = self.m_len
        chosen_models = self.best_models()
        new_models = zeros((self.ns, m_len))
        sample_idx = 0
        # loop through all the voronoi cells
        for chosen_idx, vk in enumerate(chosen_models):
            n_take = self.ns / self.nr
            if chosen_idx == 0:
                # give any remaining samples to our best model
                n_take +=self.ns % self.nr
            k = self.lowest_idxs[chosen_idx]
            for s in range(n_take):
                # current random walk location, start at voronoi cell node vk
                xA = vk.copy()
                # iterate through axes in random order, doing a random walk
                component = numpy.random.permutation(m_len)
                # vector of perpendicular distances to cell boundaries along the
                # current axis (initially from vk)
                notk = numpy.where(arange(self.np) != k)
                c0 = component[0]
                d2 = (self.m[notk, c0] - vk[c0]) ** 2
                dk2 = 0
                for idx, i in enumerate(component):
                    # find distances along axis i to all cell edges
                    vj = self.m[notk, i]
                    x = 0.5 * (vk[i] + vj + (dk2 - d2) / (vk[i] - vj))
                    # find the 2 closest points to our chosen node on each side
                    li = numpy.max(hstack((self.lo_lim, x[x <= xA[i]])))
                    ui = numpy.min(hstack((self.hi_lim, x[x > xA[i]])))
                    # randomly sample that interval and move there
                    xB = xA.copy()
                    xB[i] = (ui - li) * numpy.random.random_sample() + li
                    if idx != m_len - 1: # if not on last walk step
                        # update dj2 for next axis
                        next_i = component[idx + 1]
                        d2 += (vj - xB[i]) ** 2 - (self.m[notk, next_i] - xB[next_i]) ** 2
                        dk2 += (vk[i] - xB[i]) ** 2 - (self.m[notk, next_i] - xB[next_i]) ** 2
                    xA = xB.copy()
                new_models[sample_idx] = xA
                sample_idx += 1
        return new_models

