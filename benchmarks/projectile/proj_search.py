
import numpy as np

import cosmogenic.na as na
import cosmogenic.util as util


def positions(m, ts):
    a = -9.81
    v0 = m[0:2]
    p0 = m[2:4]
    pos = np.empty((ts.size, 2), dtype=float)
    for i, t in enumerate(ts):
        pos[i, :] = 0.5 * a * t ** 2 + v0 * t + p0
    return pos


conf = {
    "description": "Test of NA on measurements of a projectile location",
    "ns": 10,
    "nr": 2,
    "n_initial": 500,
    "lo_lim": np.array([-50, 0, -100, 0], dtype=float),
    "hi_lim": np.array([50, 200, 100, 200], dtype=float),
    "d": 4,
    "ne": 1000,
    "m_true": np.array([20.0, 100.0, 0.0, 100.0], dtype=float),
    # for benchmarking
    'seed': 10453,
    'plot': False,
}
conf['ts'] = np.linspace(2, 20, 1000)
conf['dof'] = conf['ts'].size - conf['d']

sigma_obs = 1.0  # meters
pos_true = positions(conf['m_true'], conf['ts'])


def chi2v(pre, obs, sigma_obs, nu):
    return (((obs - pre) / sigma_obs) ** 2).sum() / nu


def fn(m):
    pos = positions(m, conf['ts'])
    misfit = chi2v(pos, pos_true, sigma_obs, conf['dof'])
    return misfit

if __name__ == "__main__":
    util.pickle(conf, 'conf.pkl')
    na.search(fn, conf)
