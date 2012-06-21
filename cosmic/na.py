"""Implementation of the Neighborhood Algorithm

Author: Zach Ploskey (zploskey@uw.edu), University of Washington

Implementation of the Neighborhood Algorithm after Sambridge (1999)
in Geophys. J. Int.    
"""

from __future__ import division

import cPickle
import datetime
import logging
#import multiprocessing as mp
import os
import time

import numpy as np
import scipy as sp
import pylab
import numexpr

from joblib import Parallel, delayed

import walk

logger = logging.getLogger('neighborhood')

class NASampler(object):
    """ Sample a parameter space using the Neighborhood Algorithm."""
    
    def __init__(self, fn, ns=10, nr=2, lo_lim=0, hi_lim=1, d=1, ne=10000,
                 config=None, description=''):       
        """Create an NASampler to sample a parameter space using the
        Neighborhood algorithm.
        
        Keyword Arguments
        -----------------
        fn     -- Objective function f(x) where x is an ndarray
        ns     -- # of samples to take during each iteration
        nr     -- # of best samples whose Voronoi cells should be sampled
        lo_lim -- Lower limit of the search space (size of x)
        hi_lim -- Upper limit of the search space
        d      -- number of model dimensions
        config -- dict of configuration parameters, see below
        description -- Description of the model/inversion
        
        If config dictionary is supplied, the remaining parameters to the 
        function should be defined within the dict.
        """
        if config != None:
            # save this for later
            self.config = config
            self.d = config['d']
            self.description = config['description']
            self.hi_lim = np.atleast_1d(config['hi_lim'])
            self.lo_lim = np.atleast_1d(config['lo_lim'])
            self.ne = config['ne']
            self.nr = config['nr']
            self.ns = config['ns']
        else:
            self.d = d # model length
            self.description = description
            self.lo_lim = np.atleast_1d(lo_lim)
            self.hi_lim = np.atleast_1d(hi_lim)
            self.ne = ne
            self.nr = nr # number of Voronoi cells to explore in each step
            self.ns = ns # number of models for each step
        
        assert self.hi_lim.shape == self.lo_lim.shape, 'Limits must have the same shape.'
        n_models = self.ne + (self.ne % self.ns)
        self.date = datetime.datetime.now()
        self.m = np.zeros((n_models, self.d))        
        self.misfit = np.zeros(n_models)
        self.mdim = np.zeros(self.m.shape)
        self.fn = fn # function to minimize
        self.chosen_misfits = np.ones(self.nr) * np.inf
        self.lowest_idxs = -1 * np.ones(self.nr, dtype=int)
        self.np = 0 # number of previous models
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
        if path == None:
            if 'datadir' in self.config:
                path = config['datadir']
            else:
                path = os.getcwd()
        
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
        """Return the n best models and their misfits."""
        idx = np.argsort(self.misfit)[:n]
        return self.m[idx], self.misfit[idx], idx
    
    def fitting_models(self, tol):
        """Row matrix of models that fit better than tol."""
        #if tol == None:
        #    tol = self.tol
        fit_idx = self.misfit < tol
        return (self.m[fit_idx], self.misfit[fit_idx])
    
    def generate_ensemble(self):
        """Generate an ensemble of at least n models that fit to tolerance.
        
        Keyword Arguments:
        n -- # of models to find
        """
        n = self.ne
        assert n > 0, 'There must be at least 1 model in the ensemble.'        
        
        start_time = time.time()
        logger.info('Generating ensemble')
        logger.info('Start time: %s' % time.asctime(time.localtime(start_time)))

        max_chosen = max(self.chosen_misfits)
        it = 1 # iteration number
        idx = 0
        self.m[0:self.ns, :] = self.generate_random_models()
        # self.mdim[0:self.ns, :] = self.dimensionalize(self.m[0:self.ns, :])
        while (self.np < n): # and self.n_fitting < n:
            ns = self.ns
            if it != 1 and self.np != 0:
                new_models = self.select_new_models()
                end_idx = self.np + self.ns
                if end_idx > n:
                    ns = n - self.np
                    new_models = new_models[0:ns]
                    end_idx = n
                self.m[self.np:end_idx, :] = new_models
            # calculate the misfit function for our ns models
            # record the best (lowest) ones so far
            for j in range(ns):
                print "Iteration = %i, model index = %i. " % (it, idx),
                
                # store a dimensional version of each model model
                mdim = self.dimensionalize(self.m[idx])
                # evaluate the model with the objective function
                self.misfit[idx] = self.fn(mdim)
                self.mdim[idx] = mdim
                if self.misfit[idx] < max_chosen: 
                    old_hi_idx = np.where(self.chosen_misfits ==
                                              max_chosen)[0][0]
                    self.chosen_misfits[old_hi_idx] = self.misfit[idx]
                    self.lowest_idxs[old_hi_idx] = idx
                    max_chosen = np.max(self.chosen_misfits)
                print ""
                idx += 1
            self.np += self.ns
            it += 1
        end_time = time.time()
        print 'End time:', time.asctime(time.localtime(end_time))
        runtime = end_time - start_time
        print 'Inversion took %i minutes and %0.2f seconds.' % (round(runtime / 60),
              runtime % 60)
        return True

    def generate_random_models(self):
        """Generates a row matrix of random models in the parameter space."""
        return np.random.random_sample((self.ns, self.d))

    def select_new_models(self):
        """ Returns a 2D ndarray where each row is newly selected model.
        
        Selects new models using the Gibbs Sampler method from the 
        Voronoi cells of the best models found so far.
        """

        chosen_models = self.best_nr_models()
        new_models = np.zeros((self.ns, self.d))
        m = self.m[:self.np, :] # select all previous models
        sample_idx = 0
        # Loop through all the voronoi cells
        for chosen_idx, vk in enumerate(chosen_models):
            n_take = int(np.floor(self.ns / self.nr))
            if chosen_idx == 0:
                # Give any remaining samples to our best model
                n_take += int(np.floor(self.ns % self.nr))
            k = self.lowest_idxs[chosen_idx]
            d2_prev_ax = np.zeros(self.np)
            # Vector of perpendicular distances to cell boundaries along the
            # current axis (initially from vk)
            d2 = ((m - vk) ** 2).sum(axis=1)
            for s in range(n_take):
                # Current random walk location, start at voronoi cell node vk
                xA = vk.copy()
                # Iterate through axes in random order, doing a random walk
                component = np.random.permutation(self.d)
                for idx, i in enumerate(component):
                    # keep track of squared perpendicular distances to axis
                    d2_this_ax = (m[:, i] - xA[i]) ** 2
                    d2 += d2_prev_ax - d2_this_ax
                    dk2 = d2[chosen_idx]
                    # Find distances along axis i to all cell edges
                    vji = m[:, i]
                    x = 0.5 * (vk[i] + vji + (dk2 - d2) / (vk[i] - vji))
                    # Find the 2 closest points to our chosen node on each side
                    li = np.nanmax(np.hstack((self.lo_lim_nondim, x[x <= xA[i]])))
                    ui = np.nanmin(np.hstack((self.hi_lim_nondim, x[x >= xA[i]])))
                    # Randomly sample the interval and move there
                    xA[i] = (ui - li) * np.random.random_sample() + li
                    d2_prev_ax = d2_this_ax
                
                new_models[sample_idx] = xA.copy()
                sample_idx += 1
        return new_models

def search(fn, config):
    """ Search the parameter space defined by config for minima using the
    objective function fn, then save the results to disk."""
    sampler = NASampler(fn, config=config)
    sampler.generate_ensemble()
    
    if "datadir" in config:
        sampler.save(config['datadir'])
    else:
        sampler.save()
    return (sampler.mdim, sampler.misfit)

def resample(m=None, x2v=None, dof=1, Nw=1, pts_per_walk=1000, lo_lim=0,
             hi_lim=1, config=None, path=None):
    """Resample an ensemble of models.    
    
    Resample an ensemble of models according to the method described
    by Malcolm Sambridge (1999) Geophysical inversion with a neighbourhood
    algorithm  --II. Appraising the ensemble, Geophys. J. Int., 727--746.

    Can be called with only the config dictionary. Will load data from "m.dat"
    in the directory config['datadir'] or the current directory.
    
    Parameters:
        m: Row matrix of model vectors (dimensional is fine)
        x2v: reduced chi squared for each model
        dof: number of degrees of freedom in the problem
        Nw: number of random walks to do starting from Nw best models
        n: number of samples to collect along each walk
        lo_lim: lower bounds of the parameter space (ndarray), dimensionalized
        hi_lim: upper bounds of the parameter space (ndarray), dimensionalized
        config: optional, can be only argument and defines all the above params
    """
    if config != None:
        if "datadir" in config:
            path = os.path.abspath(config["datadir"])
        else:
            path = os.getcwd()
        m = np.loadtxt(os.path.join(path, "m.dat"))
        x2v = np.loadtxt(os.path.join(path, "misfit.dat"))
        dof = config['dof']
        Nw = config['Nw']
        pts_per_walk = config['pts_per_walk']
        lo_lim = np.atleast_1d(config['lo_lim'])
        hi_lim = np.atleast_1d(config['hi_lim'])
    
    # Number of models in the ensemble
    Ne = m.shape[0]
    # Number of dimensions in the model space
    d = m.shape[1]
    
    assert lo_lim.shape == hi_lim.shape, "Limit vectors have different shapes"
    assert lo_lim.size == d or lo_lim.size == 1
    assert hi_lim.size == d or hi_lim.size == 1
    if lo_lim.size == 1:
        tmp = np.ones(d)
        lup = (lo_lim * tmp, hi_lim * tmp)
        del tmp
    else:
        lup = (lo_lim, hi_lim)
    
    walkstart_idxs = np.argsort(x2v)[0:Nw]
    logP = -(0.5 * dof) * x2v
    start_time = time.time()
    logger.info('Start time: %s' % time.asctime(time.localtime(start_time)))
    logger.info('Generating list of random walks to perform...')
    walk_params = ((pts_per_walk, m, wi, logP, lup, i) 
                 for i, wi in enumerate(walkstart_idxs))
                 
    if logger.isEnabledFor(logging.DEBUG):
        # in debug mode we resample in a single process
        logger.debug('Importance resampling with sequential random walks.')
        # res = map(walk.walk_wrapper, walk_params)
        n_jobs = 1
        numexpr.set_num_threads(numexpr.ncores)
    else: # run in parallel
        n_jobs = min(numexpr.ncores, Nw)
        logger.info('Importance resampling with %i simultaneous walks...' % n_jobs)
        # we're already parallel, so run numexpr in single-threaded mode
        numexpr.set_num_threads(1)
    
    # pool = mp.Pool(n_jobs)
    # res = pool.map(walk.walk_wrapper, walk_params, chunksize=1)
    res = Parallel(n_jobs=n_jobs, verbose=100)(delayed(walk.walk_wrapper)(par) 
                                                for par in walk_params)
    
    assert sum(res) == Nw, 'One of the random walks failed!'
    logger.info('Finished importance sampling at %s' % time.asctime())
    logger.info('Recombining samples from each walk...')
    mr = np.zeros((Nw * pts_per_walk, d), dtype=np.float64)
    for i in range(Nw):
        cur_file = '_walk_tmp' + str(i) + '.npy'
        n = pts_per_walk
        mr[i*n:(i+1)*n, :] = np.load(cur_file)
        os.remove(cur_file)
    logger.info("Storing resampling data...")
    
    if path != None:
        np.savetxt(os.path.join(path, "mr.dat"), mr)
    
    logger.info('FINISHED IMPORTANCE SAMPLING')
    end_time = time.time()
    runtime = end_time - start_time
    logger.info('End time: %s' % time.asctime(time.localtime(end_time)))
    tH = int(runtime / 3600.0)
    tm = int(runtime % 3600 / 60.0)
    ts = int(runtime % 60)
    logger.info('Resampling finished in %i hr, %i min and %0.2f sec.' 
               % (tH, tm, ts))
    
    # return numexpr to multithreaded mode
    numexpr.set_num_threads(numexpr.ncores)
    
    return mr

def plot_stats(stats, lo_lim, hi_lim, shape=None, labels=None, m_true=None):
       # stats = {
             # 'marginals': marginals,
             # 'bin_edges': bin_edges,
             # 'C': C,
             # 'R': R,
             # 'C_prior': C_prior,
            # }

    margs = stats['marginals']
    edges = stats['bin_edges']
    d = margs.shape[1] # number of model dimensions
    
    if shape == None:
        shape = (1, d)
    
    # monkey with the figure to make room for a shared label
    #figprops = dict(figsize=(8., 8. / 1.618), dpi=128)
    #adjustprops = dict(left=0.1, bottom=0.1, right=0.97, top=0.93, wspace=0.2 hspace=0.2)
    #fig = pylab.figure(**figprops)
    #fig.subplots_adjust(**adjustprops)
    fig = pylab.figure()
    #pylab.ylabel('Probability density')
    #pylab.xlabel('Parameter value')
    ax = None
    for i in range(d):
        pos = shape + (i + 1,)
        if m_true != None:
            ax = plot_marginal(margs[:, i], edges[:, i], pos, m_true[i])
        else:
            ax = plot_marginal(margs[:, i], edges[:, i], pos)
    
    pylab.suptitle('1-D Marginals', fontsize=14)
    pylab.show()

def plot_marginal(vals, edges, pos, fig, true_val=None):
    # add the subplot, sharing x and y axes labels
    ax = pylab.subplot(pos[0], pos[1], pos[2])
    pylab.plot(edges[:-1], vals, 'k')
    
    # print y labels only on the left side axes
    if pos[1] == 1 or ((pos[2] - 1) % pos[1]) == 0:
        pylab.ylabel('Posterior Probability Density')
    
    # print x labels only on the bottom row
    #if pos[2] > pos[0] * (pos[1] - 1):
    #    pylab.xlabel('Model value')
    
    if true_val != None:
        # Construct a vertical line at the true value 
        y = [0, np.max(vals)]
        x = [true_val, true_val]
        pylab.plot(x, y, 'b-')
    
    pylab.xlim((edges[0], edges[-1]))
    pylab.ylim((0, np.max(vals)))
    pylab.title(r'$m_' + str(pos[2]) + '$')

def run(func, conf):
    mp, misfit = search(func, conf)
    mr = resample(config=conf)
    logger.info("Run finished.")
    stats = calc_stats(mr, mp, lo_lim=conf['lo_lim'], hi_lim=conf['hi_lim'], 
                       save=True)
    shape = conf['shape'] if 'shape' in conf else None
    m_true = conf['m_true'] if 'm_true' in conf else None
    plot_stats(stats, conf['hi_lim'], conf['lo_lim'], shape=shape, m_true=m_true)
    
def calc_stats(mr, mprior, nbins=100, lo_lim=0.0, hi_lim=1.0, save=False, 
          datadir=None):
    """ Calculates some statistics on resampled ensemble of models.

    Takes:
        mr:    N x d row matrix of model
        mprior: 
    """
    d = mr.shape[1]
    low = np.atleast_1d(lo_lim)
    hi = np.atleast_1d(hi_lim)
    
    if low.size == 1:
        low = low * np.ones(d)
    
    if hi.size == 1:
        hi = hi * np.ones(d)
    
    mean = mr.mean(axis=0)
    
    def nondim(a):
        return nondimensionalize(a, lo_lim, hi_lim)
    
    # Covariance
    C = covariance(nondim(mr), nondim(mean))
    numexpr.set_num_threads(numexpr.ncores)
    mean_p = mprior.mean(axis=0)
    C_prior = covariance(nondim(mprior), nondim(mean_p))
    # residuals
    R = np.eye(d)
    R -= np.dot(np.linalg.inv(C_prior), C)
    # 1d marginals (probability density functions)
    # create a matrix of 1-d marginal distributions (histograms)
    # each column is a histogram along that dimension (column) of the data
    marginals = np.zeros((nbins, d))
    bin_edges = np.zeros((nbins + 1, d))
    for i in range(d):
        marginals[:, i], bin_edges[:, i] = np.histogram(mr[:, i], bins=nbins,
                                           range=(low[i], hi[i]), density=True)
    stats = {
             'marginals': marginals,
             'mean':      mean,
             'bin_edges': bin_edges,
             'C':         C,
             'R':         R,
             'C_prior':   C_prior,
            }
    
    # redimensionalize the data, this is I think not necessary
    #for key in ('marginals', 'mean', 'bin_edges'):
    #    stats[key] = dimensionalize(stats[key], lo_lim, hi_lim)
    
    logger.info("Statistics successfully calculated.")
    
    if save:
        if datadir:
            path = os.path.abspath(datadir)
        else:
            path = os.getcwd()
        
        stats_file = os.path.join(path, 'stats.pkl')
        with open(stats_file, 'w') as fd:
            cPickle.dump(stats, fd)
        logger.info("Stats saved to %s" % stats_file)
        
    return stats

def dimensionalize(val, lo_lim, hi_lim):
    return (hi_lim - lo_lim) * val + lo_lim

def nondimensionalize(val, lo_lim, hi_lim):
    return (val - lo_lim) / (hi_lim - lo_lim)
    
def covariance(ms, mean):
    Ne = ms.shape[0]
    # old, naive calculation, runs out of memory with many models
    # return (1 / Ne) * np.inner(ms, ms) - np.outer(mean, mean)
    C = np.dot(ms.T, ms)
    C *= (1 / Ne)
    C -= np.outer(mean, mean)
    return C

def _walk(n, m, start_idx, logP, lup, walk_num):
    """ Produces models in a random walk over the initial ensemble.
    
    Models should have a sampling density that approximates the model space
    posterior probability distribution (PPD). Random numbers generated in
    the process of producing models are use an independently seeded
    RandomState object making it safe to use in parallel computations, at
    least for relatively few processes and short models runs.
    """
    # Each walk should have an independent random state so we do not overlap
    # in the numbers generated in each walk.
    random_state = np.random.RandomState()
    
    # Number of models in the ensemble
    Ne = m.shape[0]
    # Number of dimensions in the model space
    d = m.shape[1]
    
    low, up = lup
    
    resampled_models = np.empty((n, d))
    
    # don't dare overwrite our ensemble data, copy our walk position
    xp = m[start_idx].copy() 
    #mean = np.zeros(d)
    d2_prev_ax = np.zeros(Ne)
    d2 = numexpr.evaluate('sum((m - xp) ** 2, axis=1)')
    cell_idx = np.argmin(d2)
    # for each sample we take along our walk
    for i in range(n):
        # first, randomly select order of the axes to walk
        axes = random_state.permutation(d)
        for ax in axes:
            # keep track of squared perpendicular distances to axis
            m_thisax = m[:, ax]
            xp_thisax = xp[ax]
            d2_this_ax = numexpr.evaluate('(m_thisax - xp_thisax) ** 2')
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
                accepted = np.log(r) <= (logPxp - logPmax)
            # we accepted a model, this is our new position in the walk
            xp[ax] = dev
            d2_prev_ax = d2_this_ax
        resampled_models[i, :] = xp.copy()
    
    np.save('_walk_tmp%i' % walk_num, resampled_models)
    logger.info('Saved walk #%i' % walk_num)
    return True

def _calc_conditional(xp, ax, m, d2, low_lim, up_lim):
    """Calculate the conditional probability distribution along the axis. """
    xA = xp.copy()
    xA[ax] = low_lim
    # calculate distances to the lowest point on this axis
    m_thisax = m[:, ax]
    xA_thisax = xA[ax]
    d2lowedge = numexpr.evaluate('d2 + (m_thisax - xA_thisax) ** 2')
    cell_idx = np.argmin(d2lowedge)
    
    intxs = []
    idxs = [cell_idx]
    len_idxs = 1
    while cell_idx != -1:
        prev_cell_idx = cell_idx
        intx, cell_idx = _calc_upper_intersect(xA, ax, m, cell_idx, d2, up_lim)
        len_idxs += 1
        if len_idxs > m.shape[0]:
            logger.debug('Error: Runaway conditional! Conditional indexes > # of input models')
            import pdb; pdb.set_trace()
        
        intxs.append(intx)
        idxs.append(cell_idx)
        xA[ax] = intx # m[cell_idx, ax]
        # Occasionally due to rounding error we do not reach the upper limit
        # but we should then be extremely close. If that is the case then we've
        # remained in the same cell and have to manually break the loop.
        if cell_idx == prev_cell_idx:
            # make the upper boundary the true upper limit
            logger.debug('Error: Encountered repeat cell.')
            import pdb; pdb.set_trace()
            intxs[-1] = up_lim
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
    x = numexpr.evaluate('0.5 * (vki + vji + (dk2 - d2) / (vki - vji))')
    x[cell_idx] = np.inf
    x[x <= xp[ax]] = np.inf
    
    hi_idx = np.argmin(x)
    xu = x[hi_idx]
    if up_lim < xu:
        xu = up_lim
        hi_idx = -1
    
    return (xu, hi_idx)

# Potentially useful for search phase, try to work this in above, currently unused
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
    
