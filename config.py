"""
    This is a utillity file that allows loading config files from the IB-INN output directory.
"""

import configparser
import os
import listing 
import shutil

def  print_cfg(cfg):
    for sec in cfg.sections(): 
        print("# %s" % sec)
        for k, v in cfg.items(sec): print(" - %-20s:\t%s" % (k, v))

def _try_parse_int(value):
    try:
        return int(value), True
    except ValueError:
        return value, False

def _get_cfg_file(args, config=None): 
    config = config if config else args.config
    val, is_int = _try_parse_int(config)

    if is_int:
        l   = listing.list_dir(args)
        val = l[val][1]

    if os.path.isdir(val): conf_file = os.path.join(val, 'conf.ini')
    elif os.path.isfile(val) and val[-4:] == '.ini': conf_file = val
    else: raise ValueError("File '%s' does not seem to be a config file" % val)

    return conf_file

def get_configs(args):
    assert (args.config is not None) or (args.query is not None), "Command line arguments should have either `config` or `query` argument specified"

    def prepare_config(conf_file): 
        print(">> Using config:", conf_file)
        cfg = configparser.ConfigParser()
        cfg.read("submodules/IB-INN/default.ini") # Default configurations from ibinn
        cfg.read(listing.project_cfg()) # To allow additional project wide configurations
        # Override default settings with model specific settings
        cfg.read(conf_file)
        cfg['checkpoints']['config_file'] = conf_file
        return cfg

    print(args.config)
    if args.config: # Single config
        conf_files = [_get_cfg_file(args, config=c) for c in args.config] # Parse `config` argument as int or file path
        cfgs = [prepare_config(c) for c in conf_files]
    else:           # Multiple configs from query
        l = listing.list_dir(args)
        l = [os.path.join(f[1], 'conf.ini') for f in l]
        cfgs = [prepare_config(f) for f in l]

    return cfgs


def configure_output_dir(cfg):
    if 'output_dir' not in cfg['checkpoints']: 
        output_base_dir = cfg['checkpoints']['global_output_folder']
        base_name       = cfg['checkpoints']['base_name']

        os.makedirs( output_base_dir, exist_ok=True)
        num = len( [ f for f in os.listdir(output_base_dir) if f.startswith(base_name)] )
        base_name   = "%s_[%d]" % (base_name, num)

        output_dir  = os.path.join(output_base_dir, base_name)

        cfg['checkpoints']['output_dir'] = output_dir

    output_dir = cfg['checkpoints']['output_dir'] 
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'conf.ini'), 'w') as f:
        cfg.write(f)

