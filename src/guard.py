"""
Guarded routines for torch.
"""
import torch

def guard_exp(x, max_value=75, min_value=-13):
    """
    Guard exponential from becoming inf or -inf
    """
    x = torch.clamp(x, max=max_value, min=min_value)
    return torch.exp(x)

def guard_sqrt(x, minimum=1e-5):
    """
    Avoids negative values in the sqrt.
    """
    x = torch.clamp(x, min=minimum)
    return torch.sqrt(x)


def guard_acos(x):
    x = torch.clamp(x, min=-1.0, max=1.0)
    return torch.acos(x)
