import sys
sys.path.append('submodules/IB-INN')

import argparse

# Functions
import counterfactual
import listing 
import train

from dataset.dataset import Dataset

import os
import ibinn

def check_cfg(cfg): assert cfg is not None, "To use the model or the data, you must specify config:\n > python main.py [command] -c [int | output_dir_name]"

def add_standard_args(parser):
    parser.add_argument('--query',      '-q', type=str,                           help='Query string for filtering output directories.')
    parser.add_argument('--output_dir', '-o', type=str, default='./train_output', help='Output dir to query for configs, models, etc.')

def add_config_args(parser):
    parser.add_argument('--config', '-c', type=str, nargs="*", help="Config file to use when loading/training/evaluating model")

def main(): 
    parser = argparse.ArgumentParser()
    # Add individual parsers for each function
    subparsers = parser.add_subparsers()
    modules = [
            ('list',            listing),
            ('train',           train),
            ('counterfactual',  counterfactual),
        ]

    for (name, module) in modules:
        p = subparsers.add_parser(name)
        p.set_defaults(module=module)

        add_standard_args(p)
        if module.needs_config(): add_config_args(p)

        module.add_arguments(p) # Add any additional arguments needed by the module

    args = parser.parse_args()

    # Read model configuration if needed
    cfgs = [None]
    if args.module.needs_config():
        import config
        cfgs = config.get_configs(args)
        for cfg in cfgs: config.configure_output_dir(cfg)

    results = []
    
    for cfg in cfgs:
        model = None # Note that we assume that the model architecture does not change here. Only weights

        # Load model if needed
        if args.module.needs_model():
            check_cfg(cfg)
            ckpt = os.path.join(cfg['checkpoints']['output_dir'], 'model.pt') 
            if os.path.exists(ckpt): cfg['checkpoints']['resume_checkpoint'] = ckpt
            try: 
                model = ibinn.load_model(args, cfg)
            except FileNotFoundError:
                print("Model file still not available, skipping") 

        # Load dataset if needed
        data = None
        if args.module.needs_data():
            check_cfg(cfg)
            data = Dataset(cfg)

        args.module.fn(args, cfg, model, data)
        print("\n\n", "- - " * 20, "\n\n")

if __name__ == "__main__":
    main()

