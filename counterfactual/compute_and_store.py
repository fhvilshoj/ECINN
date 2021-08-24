import torch 
from tqdm import tqdm
import pickle
import os

# Local imports
from utils import cf_dir
from .counterfactuals import mean_embeddings, posterior

@torch.no_grad()
def find_alpha(z1, z2, mu1, mu2):
    mu1 = mu1.view(1, -1)
    mu2 = mu2.view(1, -1)
    w = (mu2 - mu1) / torch.norm(mu2 - mu1)
    b = -0.5* (mu1 + mu2) @ w.t()

    ip1 = z1.view(1, -1) @ w.t()
    ip2 = z2.view(1, -1) @ w.t()

    return (-b - ip1) / (ip2 - ip1)

class IncrementIndex:
    def __init__(self):
        self._cnts = {}

    def next(self, key):
        if key not in self._cnts:
            self._cnts[key] = 0
        n = self._cnts[key]
        self._cnts[key] += 1
        return n

@torch.no_grad()
def compute_counterfactuals(args, cfg, model, data):
    """
        For each test sample, compute the counterfactual examples
        for both alpha_0 and alpha_1.
    """
    model.eval()
    model.inn.eval()

    # Prepare output dir
    cf_store    = cf_dir()
    output_dir  = cfg.get('checkpoints', 'base_name') 
    out_dir     = os.path.join(cf_store, output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # compute mean embeddings if they are not already computed.
    mus     = model.mu.data.squeeze() 
    means   = mean_embeddings(cfg, data, model)[1:] # Skip total mean
    cls, d  = means.shape

    idxes = IncrementIndex() 
    mapping = [] # Index mapping for keeping track of test set order, when comparing to other methods.

    steps = 0 
    for X, Y in tqdm(data.test_loader):
        if args.max_samples > 0 and steps > args.max_samples: break
        X       = X.cuda()
        Z       = model.inn(X)
        jac     = model.inn.log_jacobian(run_forward=False).unsqueeze(1)
        P_yx    = posterior(model, Z, jac)
        y_hat   = P_yx.argmax(1)

        for i, (x, y, y_hat, p, z) in enumerate(zip(X, Y, y_hat, P_yx, Z)):
            if args.max_samples > 0 and steps > args.max_samples: break
            steps  += 1

            # Store index mapping
            idx         = idxes.next(y_hat.item())
            mapping.append( (y.item(), y_hat.item(), idx) ) # [label, y_hat, filenameidx: /path/to/[y_hat]/%04d.pt % idx]

            state_dict  = {'idx': idx, 'y': y, 'p': p, 'y_hat': y_hat}

            CFs = []
            for q in range(data.n_classes):
                D_pq = means[q] - means[y_hat]
                a0 = find_alpha(z, z+D_pq, mus[y_hat], mus[q])
                a1 = 0.8 + 0.5*a0

                z0 = z + a0 * D_pq
                z1 = z + a1 * D_pq

                # Use f^{-1} to compute counterfactuals
                Z_   = torch.stack([z0.squeeze(), z1.squeeze()])
                CFs.append(model.inn(Z_.view(-1, d), rev=True))

            x  = data.de_augment(x).clamp(0., 1.) * 255
            x  = x.type(torch.uint8)

            # CFs [10, 2, d, d, c]
            CFs = torch.stack(CFs)
            CFs = data.de_augment(CFs).clamp(0., 1.) * 255
            CFs = CFs.type(torch.uint8)

            state_dict['X'] = x
            state_dict['CFs'] = CFs

            os.makedirs(os.path.join( out_dir, str(y_hat.item())), exist_ok=True)
            torch.save(state_dict, os.path.join( out_dir, str(y_hat.item()), "%04d.pt" % idx) )
            
    with open(os.path.join(out_dir, "index_mapping.pkl"), 'wb') as f:
        pickle.dump(mapping, f)

