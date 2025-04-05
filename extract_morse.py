import torch.nn as nn
from utils import extract_morse


class ExtractMorse(nn.Module):
    def forward(self, x):
        return extract_morse(x)