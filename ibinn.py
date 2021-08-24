# convenience file to load ibinn model with or without pretrained weights. 

import os

# This is the local model file, to override GenerativeClassifier from IB-INN module
# in order to add Celeba and FakeMNIST datasets.
from model import GenerativeClassifier

def load_model(args, cfg):
    N_epochs            = eval(cfg['training']['n_epochs'])
    beta                = eval(cfg['training']['beta_IB'])
    train_nll           = bool(not eval(cfg['ablations']['no_NLL_term']))
    label_smoothing     = eval(cfg['data']['label_smoothing'])
    train_vib           = eval(cfg['ablations']['vib'])

    resume              = cfg['checkpoints']['resume_checkpoint']

    inn = GenerativeClassifier(cfg)

    inn.cuda()

    if resume:
        print(">> Loading model weights", resume)
        inn.load(resume)
    else:
        fname = os.path.join(args.output_dir, cfg.get('checkpoints', 'base_name'), 'model.pt')
        if os.path.exists(fname): 
            print(">> Loading model weights", fname)
            inn.load(fname)
    return inn

