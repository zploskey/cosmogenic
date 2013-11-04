"""
Implementation of the Neighborhood Search Algorithm and Bayesian resampling

Author: Zach Ploskey (zploskey@uw.edu), University of Washington

Implementation of the Neighborhood Algorithm after Sambridge (1999a,b)
in Geophys. J. Int.
"""
from __future__ import division, print_function

import datetime
import logging
import os
import time
import subprocess
import warnings

import math
import numexpr
import numpy as np
import pylab

from cosmogenic import util

from IPython.parallel import Client
from IPython.parallel.error import NoEnginesRegistered

logger = logging.getLogger(__name__)
SINGLE_PROCESS_DEBUG = False

MAX_IPCLUSTER_SPINUP_TIME = 10.0 # seconds
SLEEP_TIME = 0.1 # seconds

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

            if 'seed' in config:
                self.seed = config['seed']
            else:
                self.seed = None
        else:
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
            self.seed = seed

        assert self.hi_lim.shape == self.lo_lim.shape, \
            'Limits must have the same shape.'

        self.date = datetime.datetime.now()
        self.m = np.zeros((self.ne, self.d))
        self.misfit = np.zeros(self.ne)
        self.mdim = np.zeros(self.m.shape)
        self.fn = fn  # function to minimize
        self.chosen_misfits = np.ones(self.nr) * np.inf
        self.lowest_idxs = -1 * np.ones(self.nr, dtype=int)
        self.np = 0  # number of previous models
        self.lo_lim_nondim = 0.0
        self.hi_lim_nondim = 1.0

        if self.seed is not None:
            np.random.seed(self.seed)

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
        #if tol == None:
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
        it = 1  # iteration number
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

        return np.random.random_sample((n, self.d))

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
                    li = np.nanmax(
                        np.hstack((self.lo_lim_nondim, x[x <= xA[i]])))
                    ui = np.nanmin(
                        np.hstack((self.hi_lim_nondim, x[x >= xA[i]])))
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
    sampler.save()
    return (sampler.mdim, sampler.misfit)


def resample(m=None, x2v=None, dof=1, Nw=1, pts_per_walk=1000, lo_lim=0,
             hi_lim=1, config=None, path=None, ipy_profile=None):
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
    if config is not None:
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
        if ipy_profile in config:
            ipy_profile = config['ipy_profile']
        if "seed" in config:
            seed = config["seed"]
        else:
            seed = None

    if ipy_profile is None:
        subprocess.Popen('ipcluster start --daemonize --quiet', shell=True)

    # wait for the cluster to spin up
    v = _setup_engines(ipy_profile)

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

    # Log model probabilities, ignoring k constant (Sambridge 1999b eq. 29)
    logP = -(0.5 * dof) * x2v

    start_time = time.time()
    logger.info('Start time: %s' % time.asctime(time.localtime(start_time)))
    logger.info('Generating list of random walks to perform...')
    walk_params = [(pts_per_walk, m, wi, logP, lup, i, seed)
                 for i, wi in enumerate(walkstart_idxs)]

    if SINGLE_PROCESS_DEBUG:
        # in debug mode we resample in a single process
        logger.info('Importance resampling with sequential random walks.')
        res = list(map(walk_wrapper, walk_params))
    else:  # run in parallel
        logger.info('Importance resampling with parallel random walks.')
        asr = v.map(walk_wrapper, walk_params)
        asr.wait_interactive()
        res = asr.result
    
    logger.info('Finished importance sampling at %s' % time.asctime())

    if len(res) != Nw:
        warnings.warn('One of the random walks appears to have failed.')
    
    logger.info('Recombining samples from each walk...')
    mr = np.zeros((Nw * pts_per_walk, d), dtype=np.float64)

    n = pts_per_walk
    for i in range(Nw):
        mr[i * n:(i + 1) * n, :] = res[i]

    logger.info("Storing resampling data...")

    if path is not None:
        np.savetxt(os.path.join(path, "mr.dat"), mr)

    logging.info('FINISHED IMPORTANCE SAMPLING')
    end_time = time.time()
    runtime = end_time - start_time
    logger.info('End time: %s' % time.asctime(time.localtime(end_time)))
    tH = int(runtime / 3600.0)
    tm = int(runtime % 3600 / 60.0)
    ts = int(runtime % 60)
    logger.info('Resampling finished in %i hr, %i min and %0.2f sec.'
               % (tH, tm, ts))

    if ipy_profile is None:
        subprocess.Popen('ipcluster stop --quiet', shell=True)

    return mr

def walk_wrapper(w):
    """ Wrapper function for paralle call to the _walk function. """
    return _walk(w[0], w[1], w[2], w[3], w[4], w[5], w[6])

def _setup_engines(profile):
    tic = time.time()
    while True:
        try:
            try:
                client
            except:
                client = Client(profile=profile)
            dv = client[:]
        except (IOError, NoEnginesRegistered):
            toc = time.time()
            elapsed_time = toc - tic
            assert elapsed_time < MAX_IPCLUSTER_SPINUP_TIME, \
                "Cluster did not spin up properly in %d.1 seconds.".format(
                    elapsed_time)
            time.sleep(SLEEP_TIME)            
        else:
            break
    
    dv.execute('from cosmogenic.na import _walk', block=True)
    v = client.load_balanced_view()

    return v


def plot_stats(stats, lo_lim, hi_lim, shape=None, m_true=None,
               labels=None):
       # stats = {
             # 'marginals': marginals,
             # 'bin_edges': bin_edges,
             # 'C': C,
             # 'R': R,
             # 'C_prior': C_prior,
            # }
    margs = stats['marginals']
    edges = stats['bin_edges']
    d = margs.shape[1]  # number of model dimensions
    m_true = np.atleast_1d(m_true)

    if shape is None:
        shape = (1, d)

    if labels is None:
        labels = [None] * d

    pylab.figure()
    for i in range(d):
        pos = shape + (i + 1,)
        plot_marginal(margs[:, i], edges[:, i], pos, m_true[i],
                      label=labels[i])

    pylab.suptitle('1-D Marginals', fontsize=14)
    pylab.show()


def plot_marginal(vals, edges, pos, true_val=None, label=None):
    # add the subplot, sharing x and y axes labels
    pylab.subplot(pos[0], pos[1], pos[2])

    if true_val is not None:
        # Construct a vertical line at the true value
        y = [0, np.max(vals) * (1.2)]
        x = [true_val, true_val]
        pylab.plot(x, y, 'r-')

    # find the center of each bin for plotting
    bin_centers = edges[:-1] + 0.5 * np.diff(edges)
    pylab.plot(bin_centers, vals, 'k')

    # print y labels only on the left side axes
    if pos[1] == 1 or ((pos[2] - 1) % pos[1]) == 0:
        pylab.ylabel('PPD')

    pylab.xlim((edges[0], edges[-1]))
    max_val = np.max(vals)
    pylab.ylim((0, max_val * (1.1)))
    if label is not None:
        pylab.xlabel(label)

    pylab.title(r'$m_' + str(pos[2]) + '$')


def run(func, conf):
    mp, _ = search(func, conf)
    logger.info("Run finished.")
    stats = resample_and_plot(conf)
    return stats


def resample_and_plot(conf):
    """
    conf:   configuration dictionary
    mp:     prior models
    misfit: misfits of the prior model
    """
    mr = resample(config=conf)
    stats = calc_stats(mr, lo_lim=conf['lo_lim'], hi_lim=conf['hi_lim'],
                       save=True)
    shape = conf['shape'] if 'shape' in conf else None
    m_true = conf['m_true'] if 'm_true' in conf else None
    if ("plot" not in conf) or conf["plot"]:
        plot_stats(stats, conf['lo_lim'], conf['hi_lim'], shape=shape,
                   m_true=m_true)
    return stats


def plot_results(conf=None, stats=None, m_true=None, labels=None):
    if conf is None:
        conf = util.unpickle('conf.pkl')
    if stats is None:
        stats = util.unpickle('stats.pkl')
    shape = conf['shape'] if 'shape' in conf else None
    if m_true is None:
        m_true = conf['m_true'] if 'm_true' in conf else None

    plot_stats(stats, conf['lo_lim'], conf['hi_lim'], shape=shape,
               m_true=m_true, labels=labels)
    return stats


def calc_stats(mr=None, nbins=100, lo_lim=0.0, hi_lim=1.0, C_prior=None,
               save=False, datadir=None):
    """ Calculates some statistics on resampled ensemble of models.

    Takes:
        mr:      N x d row matrix of models
    """
    if mr == None:
        mr = np.genfromtxt('mr.dat')

    d = mr.shape[1]
    low = np.atleast_1d(lo_lim)
    hi = np.atleast_1d(hi_lim)

    if low.size == 1:
        low = low * np.ones(d)

    if hi.size == 1:
        hi = hi * np.ones(d)

    mean = mr.mean(axis=0)

    # Covariance
    C = np.cov(mr.T, bias=1)

    # Correlation coefficients
    corr = np.corrcoef(mr.T, bias=1)

    # calculate the resolution matrix R
    if C_prior is None:
        # if no prior covariance is provided we assume a uniform prior
        # See Sambridge 1999b Appendix 2, equation A.12
        # so C_prior is a diag. matrix w/ entries (1/12) * (U_i - L_i)**2
        # where U_i and L_i are the upper and lower boundaries of the search
        # space for model parameter i.
        C_prior_diag = (1 / 12.0) * (hi_lim - lo_lim) ** 2
        C_prior = np.diag(C_prior_diag)
        C_prior_inv = np.diag(1 / C_prior_diag)
    else:
        C_prior_inv = np.linalg.inv(C_prior)

    R = np.eye(d)
    R -= np.dot(C_prior_inv, C)
    # calculate nondimensional resolution matrix
    a = np.sqrt(np.diag(C_prior))  # temporary vector
    R_nondim = R * np.outer((1 / a), a)  # Sambridge 1999b eq. 33

    # 1d marginals (probability density functions)
    # create a matrix of 1-d marginal distributions (histograms)
    # each column is a histogram along that dimension (column) of the data
    marginals = np.zeros((nbins, d))
    bin_edges = np.zeros((nbins + 1, d))
    mode = np.zeros(d)
    std = np.zeros(d)
    ci95 = []
    ci68 = []
    for i in range(d):
        marginals[:, i], bin_edges[:, i] = np.histogram(mr[:, i], bins=nbins,
                                           range=(low[i], hi[i]), density=True)
        mode[i] = marginals[:, i].argmax() * (hi[i] - low[i]) / float(nbins)
        # calculate 95% confidence interval
        ci95.append(confidence_interval(marginals[:, i], low[i], hi[i],
            alpha=0.95))
        ci68.append(confidence_interval(marginals[:, i], low[i], hi[i],
            alpha=0.68))
        std[i] = mr[:, i].std()

    # correlation matrix -- Sambridge 1999b eq. 30
    # corr = C / np.sqrt(np.dot(np.diag(C,  )

    stats = {
             'marginals': marginals,
             'mean': mean,
             'mode': mode,
             'bin_edges': bin_edges,
             'C': C,
             'corr': corr,
             'R': R,
             'R_nondim': R_nondim,
             'ci68': ci68,
             'ci95': ci95,
             'std': std,
            }

    if save:
        if datadir:
            path = os.path.abspath(datadir)
        else:
            path = os.getcwd()
        stats_file = 'stats.pkl'
        util.pickle(stats, stats_file, path)
        logger.info("Stats saved to %s" % stats_file)

    return stats


def confidence_interval(marg, lo_lim, hi_lim, alpha=0.68):
    """Return confidence interval for a probability distribution

    Calculates a confidence interval for a marginal posterior probability
    distribution. By default, it calculates a 68% confidence interval.

    Parameters
    ----------
    marg : array_like
        1-d array of evenly spaced points of a probability distribution.
        The sum of the elements of this array should be 1.0.
    lo_lim : float
        The upper limit of the distribution.
    hi_lim : float
        The lower limit of the distribution.
    alpha : float, optional
        Fractional probability you would like to find the CI for. Must be
        between 0 and 1.
    """
    assert 0.0 < alpha < 1.0, "alpha must be between 0 and 1"
    assert hi_lim > lo_lim, "upper limit must be greater than the lower limit"
    nbins = marg.size
    dx = (hi_lim - lo_lim) / float(nbins)

    p_target = (1.0 - alpha) / 2.0

    p = 0
    low_idx = -1
    while p < p_target:
        low_idx += 1
        p_prev = p
        dp = marg[low_idx]
        p += dp * dx
    excess_p = p_target - p_prev
    dpdx = (p - p_prev) / dx
    x_overshoot = excess_p / dpdx
    ci_low = lo_lim + low_idx * dx - x_overshoot

    p = 0
    hi_idx = nbins
    while p < p_target:
        hi_idx -= 1
        p_prev = p
        dp = marg[hi_idx]
        p += dp * dx
    excess_p = p_target - p_prev
    dpdx = (p - p_prev) / dx
    x_overshoot = excess_p / dpdx
    ci_hi = hi_lim - (nbins - hi_idx) * dx + x_overshoot
    return (ci_low, ci_hi)


def dimensionalize(val, lo_lim, hi_lim):
    return (hi_lim - lo_lim) * val + lo_lim


def nondimensionalize(val, lo_lim, hi_lim):
    return (val - lo_lim) / (hi_lim - lo_lim)


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
                accepted = math.log(r) <= (logPxp - logPmax)
            # we accepted a model, this is our new position in the walk
            xp[ax] = dev
            d2_prev_ax = d2_this_ax
        resampled_models[i, :] = xp.copy()

    return resampled_models


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

def _calc_upper_intersect(xp, ax, m, cell_idx, d2, up_lim):
    """Calculate the upper intersection of the cell and the axis. 
    
    xp: the previous model
    ax: the current axis (int)
    m:  all the models
    cell_idx: index of the current cell
    d2: squared distances
    """
    vki = m[cell_idx, ax]
    vji = m[:, ax]
    dx = vki - vji
    # squared perpendicular distance to the axis from vk
    dk2 = d2[cell_idx]
    # calculate upper intersection point
    x = numexpr.evaluate('0.5 * (vki + vji + (dk2 - d2) / dx)')
    x[cell_idx] = np.inf
    x[x <= xp[ax]] = np.inf

    hi_idx = np.argmin(x)
    xu = x[hi_idx]
    if up_lim <= xu:
        xu = up_lim
        hi_idx = -1

    return (xu, hi_idx)


def _calc_intersects(xp, ax, m, cell_idx, d2, lup):
    """Finds upper and lower intersection of axis through xp along axis ax.

    Unused, potentially useful for search phase above.
    """
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
