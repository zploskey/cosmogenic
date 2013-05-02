# walk.py

from . import na

def walk_wrapper(w):
    """ Wrapper function for paralle call to the na._walk function. """
    return na._walk(w[0], w[1], w[2], w[3], w[4], w[5])
