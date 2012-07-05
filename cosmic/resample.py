"""
This script resamples data from a prior neighborhood algorithm search.
"""
import os
import sys

import numpy as np
import pylab

from numpy import array # need this for eval to work

import na

def main():
    argv = sys.argv
    data_dir = argv[1]

    if not os.path.exists(data_dir):
        raise Exception('Directory \'%s\' does not exist. Aborted.', data_dir)

    if len(argv) > 2:
        outfile = argv[2]
    else:
        outfile = 'mr.txt'

    m = np.loadtxt(os.path.join(data_dir, 'models.txt'))
    x2v = np.loadtxt(os.path.join(data_dir, 'errors.txt'))
    con = eval(f.open(os.path.join(data_dir, 'constraints_fixed.txt')).read())
    dof = con['dof']
    d = m.shape[1]
    limits = (np.ones(d) * con['min_dz'], np.ones(d) * con['max_dz'])
    Nw = 100 # number of walks
    n = 1000 # number of samples to take along each walk

    mr = na.resample(m, x2v, dof, Nw, n, limits)
    np.savetxt(os.path.join(data_dir, outfile), mr)
    stats = na.stats(mr, m)
    fig = plot_stats(stats)

def plot_stats(stats, true_vals=None):
            # stats = {
             # 'marginals': marginals,
             # 'bin_edges': bin_edges,
             # 'C': C,
             # 'R': R,
             # 'C_prior': C_prior,
            # }
    margs = stats['marginals']
    edges = stats['bin_edges']
    d = m.shape[1] # number of model dimensions
    # monkey with the figure to make room for a shared label
    #figprops = dict(figsize=(8., 8. / 1.618), dpi=128)
    #adjustprops = dict(left=0.1, bottom=0.1, right=0.97, top=0.93, wspace=0.2 hspace=0.2)
    #fig = pylab.figure(**figprops)
    #fig.subplots_adjust(**adjustprops)
    fig = pylab.figure()
    ax = None
    for i in range(d):
        if true_vals != None:
            ax = plot_marginal(margs[:, i], edges[:, i], (1, i), fig, ax,
                               true_vals[i])
        else:
            ax = plot_marginal(margs[:, i], edges[:, i], (1, i), fig, ax)
    
    pylab.xlabel('Parameter value')
    pylab.ylabel('Probability density')
    pylab.title('1-D Marginals')
    pylab.show()

def plot_marginal(vals, edges, pos, fig, prev_ax, true_val=None):
    d = m.shape[1]
    tot = d * m.shape[0]
    
    # add the subplot, sharing x and y axes labels
    ax = fig.add_subplot(tot, pos[0], pos[1]) #, sharex=prev_ax, sharey=prev_ax)
    pylab.plot(edges[:-1], vals, 'k')
    
    if true_vals != None:
        # Construct a vertical line at the true value 
        y = [0, np.max(vals)]
        x = [true_val, true_val]
        pylab.plot(x, y, 'k-')
    
    pylab.title(r'$m_' + str(pos[0] * pos[1]) + '$')
    
    
if __name__ == "__main__":
    main()
