import torch.nn as nn

__all__ = [
    'weights_init',
    'argsort',
    'count_parameters'
]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.001)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.001)
        m.bias.data.fill_(0.001)

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
