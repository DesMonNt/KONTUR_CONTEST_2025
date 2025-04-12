import torch
import torch.nn as nn
import torchaudio.transforms as T


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=10, time_mask_param=30, noise_std=0.02, num_masks=2):
        super().__init__()

        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.noise_std = noise_std
        self.num_masks = num_masks

    def forward(self, spec):
        for _ in range(self.num_masks):
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)

        noise = torch.randn_like(spec) * self.noise_std
        spec = spec + noise

        return spec
