import torch

def pairwise_distances(X, Y): 
    """
        Returns squared pairwise euclidean distances
    """
    p, d = X.shape
    q, _ = Y.shape

    DiDj = X.mm(Y.t()) # [p, q]
    DiDi = torch.sum(X**2, 1, True).expand(p, q) # [p, q]
    DjDj = torch.sum(Y**2, 1, True).T.expand(p, q)
    return DiDi + DjDj - 2*DiDj

