"""
Implementation of the Neighborhood Search Algorithm and Bayesian resampling

Implementation of the Neighborhood Algorithm after Sambridge (1999a,b)
in Geophys. J. Int.
"""

from __future__ import division, print_function

import datetime
import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)


class NASampler(object):

    """ Samples a parameter space using the Neighborhood Algorithm."""

    def __init__(self, fn, ns=10, nr=2, lo_lim=0.0, hi_lim=1.0, d=1, ne=10000,
                 config=None, description='', n_initial=None, seed=None):
        """Create an NASampler to sample a parameter space using the
        Neighborhood algorithm.

        Keyword Arguments
        -----------------
        fn     -- Objective function f(x) where x is an ndarray
        ns     -- # of samples to take during each iteration
        nr     -- # of best samples whose Voronoi cells should be sampled
        lo_lim -- Lower limits of the search space (size of x)
        hi_lim -- Upper limits of the search space
        d      -- number of model dimensions
        config -- dict of configuration parameters, see below
        description -- Description of the model/inversion

        If config dictionary is supplied, the remaining parameters to the
        function should be defined within the dict.
        """
        if config is not None:
            if 'seed' in config:
                self.seed = config['seed']
            else:
                self.seed = None
            self.rng = np.random.RandomState(self.seed)
            # save this for later
            self.config = config
            self.d = config['d']
            self.datadir = (config['datadir'] if 'datadir' in config
                            else os.getcwd())
            self.description = config['description']
            self.hi_lim = np.atleast_1d(config['hi_lim'])
            self.lo_lim = np.atleast_1d(config['lo_lim'])
            self.ne = config['ne']
            self.nr = config['nr']
            self.ns = config['ns']
            if 'n_initial' in config:
                self.n_initial = config['n_initial']
            else:
                self.n_initial = self.ns

        else:
            self.rng = np.random.RandomState(seed)
            self.seed = seed
            self.d = d  # model length
            self.description = description
            self.lo_lim = np.atleast_1d(lo_lim)
            self.hi_lim = np.atleast_1d(hi_lim)
            self.ne = ne
            self.nr = nr  # number of Voronoi cells to explore in each step
            self.ns = ns  # number of models for each step
            if n_initial is None:
                self.n_initial = self.ns
            else:
                self.n_initial = n_initial

        assert self.hi_lim.shape == self.lo_lim.shape, \
            'Limits must have the same shape.'

        self.date = datetime.datetime.now()
        self.m = np.zeros((self.ne, self.d))
        self.misfit = np.zeros(self.ne)
        self.mdim = np.zeros(self.m.shape)
        self.fn = fn  # function to minimize
        self.chosen_misfits = np.ones(self.nr) * np.inf
        self.lowest_idxs = -1 * np.ones(self.nr, dtype=int)
        self.np = 0   # number of previous models
        self.lo_lim_nondim = 0.0
        self.hi_lim_nondim = 1.0

    def dimensionalize(self, x):
        return self.lo_lim + (self.hi_lim - self.lo_lim) * x

    def save(self, path=None):
        """
        Save collected data and search parameters to disk.

        Keyword Arguments
        -----------------
        path -- The path where files will be saved (optional). If not
                supplied, files are saved in the current directory.

        """
        if path is None:
            path = self.datadir

        abspath = os.path.abspath(path)
        np.savetxt(os.path.join(abspath, "m.dat"), self.mdim)
        np.savetxt(os.path.join(abspath, "m_nondim.dat"), self.m)
        np.savetxt(os.path.join(abspath, 'misfit.dat'), self.misfit)
        with open(os.path.join(abspath, 'search_params.txt'), 'w') as fd:
            fd.write(self.date.strftime("%Y-%b-%d %H:%M:%S\n"))
            fd.write("Description: %s\n" % self.description)
            fd.write("nr:          %i\n" % self.nr)
            fd.write("ns:          %i\n" % self.ns)
            fd.write("# of models: %i\n" % self.m.shape[0])
            fd.write("Dimensions:      %i\n" % self.d)
            fd.write("Lower limit:\n")
            fd.write("%s\n" % str(self.lo_lim))
            fd.write("Upper limit:\n")
            fd.write("%s\n" % str(self.hi_lim))

    def best_nr_models(self):
        """Row matrix of nr best models."""
        return self.m[self.lowest_idxs]

    def best_models(self, n):
        """Return the n best models, their misfits and indices.

        Return: (models, misfit_vector, indices) """
        idx = np.argsort(self.misfit)[:n]
        return self.m[idx], self.misfit[idx], idx

    def fitting_models(self, tol):
        """Row matrix of models that fit better than tol."""
        # if tol == None:
        #    tol = self.tol
        fit_idx = self.misfit < tol
        return (self.m[fit_idx], self.misfit[fit_idx])

    def generate_ensemble(self):
        """Generate an ensemble of self.ne models.

        Keyword Arguments:
        n -- # of models to find
        """
        n = self.ne
        assert n > 0, 'There must be at least 1 model in the ensemble.'

        start_time = time.time()
        logger.info('Generating ensemble...')
        logger.info('Start time: %s'
                    % time.asctime(time.localtime(start_time)))

        max_chosen = max(self.chosen_misfits)
        it = 1   # iteration number
        idx = 0  # last used index
        ns = self.n_initial
        self.m[0:ns, :] = self.generate_random_models(ns)

        while (self.np < n):

            if it != 1 and self.np != 0:
                end_idx = self.np + self.ns

                if end_idx > n:
                    ns = n - self.np
                    end_idx = n

                self.m[self.np:end_idx, :] = self.select_new_models(ns)
            # calculate the misfit function for our ns models
            # record the best (lowest) ones so far
            for j in range(ns):

                # store a dimensional version of each model
                mdim = self.dimensionalize(self.m[idx])
                self.mdim[idx] = mdim
                # evaluate the model with the objective function
                self.misfit[idx] = self.fn(mdim)

                if self.misfit[idx] < max_chosen:
                    old_hi_idx = np.where(
                        self.chosen_misfits == max_chosen)[0][0]
                    self.chosen_misfits[old_hi_idx] = self.misfit[idx]
                    self.lowest_idxs[old_hi_idx] = idx
                    max_chosen = np.max(self.chosen_misfits)
                idx += 1
            self.np += ns
            minimum = self.chosen_misfits[0]
            print("Min: %0.4g, iter %i, %i models" % (minimum, it, idx + 1))
            it += 1
            ns = self.ns

        end_time = time.time()
        print('End time:', time.asctime(time.localtime(end_time)))
        runtime = end_time - start_time
        print('Inversion took %i minutes and %0.2f seconds.' % (
            round(runtime / 60), runtime % 60))
        return True

    def generate_random_models(self, n=None):
        """Generates a row matrix of random models in the parameter space."""
        if n is None:
            n = self.ns

        return self.rng.random_sample((n, self.d))

    def select_new_models(self, ns=None):
        """ Returns a 2D ndarray where each row is newly selected model.

        Selects new models using the Gibbs Sampler method from the
        Voronoi cells of the best models found so far.
        """
        if ns is None:
            ns = self.ns

        chosen_models = self.best_nr_models()
        new_models = np.zeros((ns, self.d))
        m = self.m[:self.np, :]  # select all previous models
        sample_idx = 0
        # Loop through all the voronoi cells
        for chosen_idx, vk in enumerate(chosen_models):
            n_take = int(np.floor(ns / self.nr))
            if chosen_idx == 0:
                # Give any remaining samples to our best model
                n_take += int(np.floor(ns % self.nr))
            k = self.lowest_idxs[chosen_idx]
            d2_prev_ax = np.zeros(self.np)
            # Vector of perpendicular distances to cell boundaries along the
            # current axis (initially from vk)
            d2 = ((m - vk) ** 2).sum(axis=1)
            for s in range(n_take):
                # Current random walk location, start at voronoi cell node vk
                xA = vk.copy()
                # Iterate through axes in random order, doing a random walk
                component = self.rng.permutation(self.d)
                for idx, i in enumerate(component):
                    # keep track of squared perpendicular distances to axis
                    d2_this_ax = (m[:, i] - xA[i]) ** 2
                    d2 += d2_prev_ax - d2_this_ax
                    dk2 = d2[chosen_idx]
                    # Find distances along axis i to all cell edges
                    vji = m[:, i]
                    x = 0.5 * (vk[i] + vji + (dk2 - d2) / (vk[i] - vji))
                    # Find the 2 closest points to our chosen node on each side
                    li = np.nanmax(
                        np.hstack((self.lo_lim_nondim, x[x <= xA[i]])))
                    ui = np.nanmin(
                        np.hstack((self.hi_lim_nondim, x[x >= xA[i]])))
                    # Randomly sample the interval and move there
                    xA[i] = (ui - li) * self.rng.random_sample() + li
                    d2_prev_ax = d2_this_ax

                new_models[sample_idx] = xA.copy()
                sample_idx += 1
        return new_models


def search(fn, config):
    """ Search the parameter space defined by config for minima using the
    objective function fn, then save the results to disk."""
    sampler = NASampler(fn, config=config)
    sampler.generate_ensemble()
    sampler.save()
    return (sampler.mdim, sampler.misfit)


def dimensionalize(val, lo_lim, hi_lim):
    return (hi_lim - lo_lim) * val + lo_lim


def nondimensionalize(val, lo_lim, hi_lim):
    return (val - lo_lim) / (hi_lim - lo_lim)
