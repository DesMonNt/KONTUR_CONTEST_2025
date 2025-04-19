import torch
import torch.nn as nn
import torchaudio


class MorsePreprocessing(nn.Module):
    def __init__(self, sr=8000, window_duration=0.02, n_fft=1024, hop_length=256, power=2.0, alpha=0.1, margin=5):
        super().__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=power,
        )

        self.alpha = alpha
        self.margin = margin

    def normalize_amplitude(self, waveform):
        max_val = waveform.abs().max()

        return waveform / max_val if max_val > 0 else waveform

    def threshold_spectrogram(self, spec):
        energy_per_bin = spec.mean(dim=2)
        peak_bin = torch.argmax(energy_per_bin, dim=1)
        peak_mean = spec[0, peak_bin.item(), :].mean()

        start_bin = max(0, peak_bin.item() - self.margin)
        end_bin = min(spec.shape[1], peak_bin.item() + self.margin)

        spec = spec[:, start_bin:end_bin, :]

        return torch.where(spec >= peak_mean * self.alpha, spec, 0.0)

    def forward(self, waveform):
        spec = self.spectrogram(waveform)
        spec = self.threshold_spectrogram(spec)
        spec = self.normalize_amplitude(spec)

        return spec