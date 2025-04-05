import torch.nn as nn
from utils import normalize_nonzero

class Normalize(nn.Module):
    def forward(self, x):
        return normalize_nonzero(x)