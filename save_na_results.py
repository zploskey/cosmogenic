#!/usr/bin/python

import os
import sys
import glob
import shutil

def copy_files(src_glob, dst_folder):
    for fname in glob.iglob(src_glob):
        shutil.copy(fname, os.path.join(dst_folder, fname))

def save_na_results(folder_path=None):
    """Move the results of a Neighborhood Algorithm run to a folder with metadata.
    
    This will copy all files with extensions of:
    npy, txt, eps, png and dat
    to the specified folder_path.
    
    For now there is nothing stored in the folder that will be taken in by this
    but we should be careful not to introduce anything that will be copied, or
    at least modify this script accordingly if we do.
    """    
    if folder_path is None:
        folder_path = sys.argv[1]
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        raise Exception('Directory already exists; %s' % folder_path)

    problem = False
    file_exts = ['npy', 'txt', 'png', 'svg', 'eps', 'dat']
    for ext in file_exts:
        try:
            copy_files('*.' + ext, folder_path)
        except Exception, e:
            print 'Got:', e.args
            problem = True
    
    if problem:
        print 'An error was encountered while copying...'
    else:
        print 'Results were moved into %s' % folder_path

if __name__ == '__main__':
    save_na_results()
