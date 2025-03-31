import torch
import torch.nn as nn
import torchaudio


class LogMelFilterBanks(nn.Module):
    def __init__(self, n_fft=400, hop_length=160, n_mels=80, sr=16000, f_min=0.0, f_max=None, mel_scale='htk'):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sr = sr
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sr / 2
        self.mel_scale = mel_scale

        window = torch.hann_window(self.n_fft)
        self.register_buffer("window", window, persistent=False)

        fbanks = self._init_melscale_fbanks()
        self.register_buffer("fbanks", fbanks, persistent=False)

    def _init_melscale_fbanks(self):
        fbanks = torchaudio.functional.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            n_mels=self.n_mels,
            sample_rate=self.sr,
            f_min=self.f_min,
            f_max=self.f_max,
            mel_scale=self.mel_scale
        )
        return fbanks

    def spectrogram(self, x):
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )
        return stft.abs().pow(2)

    def forward(self, x):
        spec = self.spectrogram(x)
        mel = torch.matmul(spec.transpose(1, 2), self.fbanks)
        mel = mel.transpose(1, 2)
        return torch.log(mel + 1e-6)
