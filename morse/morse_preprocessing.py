import torch
import torch.nn as nn
import torchaudio


class MorsePreprocessing(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, power=2.0):
        super().__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=power
        )

    def normalize_amplitude(self, waveform):
        max_val = waveform.abs().max()
        return waveform / max_val if max_val > 0 else waveform

    def normalize_spectrogram(self, spec):
        energy = spec.mean(dim=1).squeeze(0)
        mask = energy > -80

        if mask.sum() == 0:
            return spec

        active = spec[:, :, mask]
        mean = active.mean()
        std = active.std()

        return (spec - mean) / (std + 1e-6)

    def forward(self, waveform):
        #waveform = self.normalize_amplitude(waveform)
        spec = self.spectrogram(waveform)
        spec = self.normalize_spectrogram(spec)

        return spec