"""
    This file is intended to be used for listing output directories such that
    it is easy to choose a model from that project for experiments in this
    project.
    It works both for the IB-INN submodule output directory end the 
    `train_output` in this project. 
"""
import os
import re
from natsort import natsorted
import configparser

_library_base = 'submodules/IB-INN'

# Utility methods
def library_base_dir(): return _library_base

def default_cfg():  return os.path.join(library_base_dir(), 'default.ini')
def project_cfg():  return 'project.ini'

def list_dir(args):
    dst = args.output_dir
    dirs = [os.path.join(dst, f) for f in os.listdir(dst) if os.path.isdir(os.path.join(dst, f))]
    dir_list = natsorted(dirs)
    dir_list = [(i, f) for i, f in enumerate(dir_list)]

    if args.query is not None: dir_list = [t for t in dir_list if re.search(args.query, t[1])]

    return dir_list

# command line functions
def add_arguments(parser):  pass
def needs_config():         return False
def needs_model():          return False
def needs_data():           return False

def fn(args, *_):
    if not os.path.exists(args.output_dir) and args.output_dir[:12] == './submodules':
        print('%s is missing. Run\n> git submodule update --init\nto clone subprojects' % _output_dir)

    print("> Listing dirs")
    files = list_dir(args)
    for t in files: print("%03i: %s" % t)
    

    

