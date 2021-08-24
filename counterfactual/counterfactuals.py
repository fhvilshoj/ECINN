import torch
from tqdm import tqdm
import os

from .cf_utils      import Avg
from .distances     import pairwise_distances

@torch.no_grad()
def posterior(model, z, jac, return_marginal=False):
        # Constants
        pi          = 2 * torch.acos(torch.zeros(1))
        log_tau     = torch.log(2*pi).item() # tau = 2pi
        log10       = torch.log( torch.ones(1, device=z.device)*10 )
        log10       = torch.log( torch.ones(1, device=z.device)*10 )

        zz          = pairwise_distances(z, model.mu.data.squeeze()) # || z-mu ||^2
        log_P_xy    = -model.ndim_tot / 2 * log_tau - 0.5*zz + jac # / model.ndim_tot 
        log_P_x     = torch.logsumexp( log_P_xy, dim=1, keepdims=True)                              - log10

        # Normalization doesn't seem right here.
        # Since normalization does not change the order of the probabilities, it should not change classification results, neither CF selection further down-stream.
        P_yx        = torch.exp( log_P_xy  - log_P_x ) 

        if not return_marginal: return P_yx
        else:
            return P_yx, log_P_x 

@torch.no_grad()
def mean_embeddings(config, data, model): 
    out = config.get('checkpoints', 'output_dir')
    out_file = os.path.join(out, 'mean_training_embeddings_pr_class.pt')
    if os.path.exists(out_file):
        print("Loading mean embeddings")
        d = torch.load(out_file)
        means = d['means']
    else: 
        print("Computing mean embeddings")
        overall_mean = Avg()
        means = [Avg() for _ in range(data.n_classes)]
        i = 0
        for x, l in tqdm(data.train_loader): 
            z       = model.inn(x.cuda())
            jac     = model.inn.log_jacobian(run_forward=False).unsqueeze(1)
            p       = posterior(model, z, jac)
            l_hat   = p.argmax(1)

            overall_mean(z)
            for l_, m in enumerate( means ):  # for each label class
                zs = z[l_hat==l_]
                m(zs)
        means = torch.cat([m.mean.unsqueeze(0) for m in [overall_mean] + means], 0)
        torch.save({'means': means}, out_file)
    return means

