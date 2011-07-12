#!/usr/bin/python

import os
import sys

def save_na_results(folder_path=None):
    """Move the results of a Neighborhood Algorithm run to a folder with metadata.
    
    This will copy all files with extensions of:
    npy, txt, eps, png and dat
    to the specified folder_path.
    
    For now there is nothing stored in the folder that will be taken in by this
    but we should be careful not to introduce anything that will be copied, or
    at least modify this script accordingly if we do.
    """    
    if sys.platform is 'win32':
        raise NotImplementedError('Will not currently work on Windows.')     
    if folder_path is None:
        folder_path = sys.argv[1]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        raise Exception('Directory already exists; %s' % folder_path)
    try:
            os.system("cp *.npy %s" % folder_path)
            os.system("cp *.txt %s" % folder_path)
            os.system("cp *.eps %s" % folder_path)
            os.system("cp *.png %s" % folder_path)
            os.system("cp *.dat %s" % folder_path)
    except OSError('Something went wrong while copying.'), e:
        print 'Got:', e.args
    else:
        print 'Results were copied into %s' % folder_path
    
    notes = generate_notes()
    
    
def generate_notes():
    notes = ''

if __name__ == '__main__':
    save_na_results()
