import torch
import torch.nn as nn
import torchaudio

class MorsePreprocessing(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, power=2.0, threshold=3.0):
        super().__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=power
        )
        self.threshold = threshold

    def normalize(self, spec):
        mean = spec.mean(dim=(-2, -1), keepdim=True)
        std = spec.std(dim=(-2, -1), keepdim=True)

        return (spec - mean) / (std + 1e-6)

    def forward(self, waveform):
        waveform = waveform / waveform.abs().max()

        spec = self.spectrogram(waveform)
        spec = self.normalize(spec)
        spec = torch.where(spec > self.threshold, spec, torch.zeros_like(spec))

        return spec