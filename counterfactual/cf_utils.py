import torch

class Avg:
    def __init__(self):
        super().__init__()
        self.cnt = 0
        self.mean = None

    def update(self, t): 
        if t.shape[0] == 0: return
        if self.mean is None: self.mean = torch.zeros_like(t.sum(0))
        frac        = t.size(0) / (self.cnt + t.size(0))
        self.mean   = frac * t.mean(0) + self.mean * (1-frac)
        self.cnt   += t.size(0)

    def __call__(self, t): self.update(t)


def get_samples(data, cnt=9, device='cuda'):
    for x, y in data.test_loader: break
    x = x[:cnt].to(device)
    y = y[:cnt].to(device)
    return x, y


def one_hot(labels, classes=10):
    I = torch.eye(classes).to(labels.device)
    oh = I[labels]
    return oh
