import os
import numpy as np
import torch
from config import Config

def mixup_data(x, y, alpha=1.0):
    """Mixup for binary classification
    Args:
        x (torch.Tensor): batch of inputs
        y (torch.Tensor): batch of binary labels 
        alpha (float, optional): Defaults to 1.0.
    Returns:
        mixed_x, y_a, y_b, lam
    """


    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, weight):
    return lam * criterion(pred, y_a, weight=weight) + (1 - lam) * criterion(pred, y_b,weight=weight)