import torch
"""
Code to add a fixed classifier to a net
"""

def dsimplex(num_classes=10, device='cuda'):
    def simplex_coordinates(n, device):
        t = torch.zeros((n + 1, n), device=device)
        torch.eye(n, out=t[:-1,:], device=device)
        val = (1.0 - torch.sqrt(1.0 + torch.tensor([n], device=device))) / n
        t[-1,:].add_(val)
        t.add_(-torch.mean(t, dim=0))
        t.div_(torch.norm(t, p=2, dim=1, keepdim=True)+ 1e-8)
        return t.cpu()

    feat_dim = num_classes - 1
    ds = simplex_coordinates(feat_dim, device)
    return ds