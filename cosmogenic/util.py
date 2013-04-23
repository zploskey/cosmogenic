import cPickle
import os


def unpickle(filename):
    path = os.path.abspath(filename)
    with open(path, 'r') as fd:
        payload = cPickle.load(fd)
    return payload


def pickle(obj, filename, path=None):
    # Warning, path as anything but none is untested
    if path is None:
        path = os.getcwd()

    file_path = os.path.join(path, filename)
    with open(file_path, 'w') as fd:
        cPickle.dump(obj, fd)
