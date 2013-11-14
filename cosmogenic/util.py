"""
Utility functions
"""
from __future__ import division, print_function, unicode_literals

import os

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

def unpickle(filename):
    """
    Unpickle and return the contents of a file at filename.
    """
    path = os.path.abspath(filename)
    with open(path, 'rb') as fd:
        payload = pkl.load(fd)
    return payload


def pickle(obj, filename, path=None):
    """
    Pickle object obj and store it in the file filename.
    Optionally takes a path to save the file at.

    Warning: path as anything but none is untested
    """
    if path is None:
        path = os.getcwd()

    file_path = os.path.join(path, filename)
    with open(file_path, 'wb') as fd:
        pkl.dump(obj, fd)
