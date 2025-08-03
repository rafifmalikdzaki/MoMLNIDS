from torch import nn
from torch.nn import functional as F
import torch


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1).mean()
        return b


class MaximumSquareLoss(nn.Module):
    def __init__(self):
        super(MaximumSquareLoss, self).__init__()

    def forward(self, x):
        p = F.softmax(x, dim=1)
        b = torch.mul(p, p)
        b = -1.0 * b.sum(dim=1).mean() / 2
        return b
