# Copyright 2021 Tomoki Hayashi
# MIT License (https://opensource.org/licenses/MIT)
# Adapted by Florian Lux 2021


import librosa
import torch
import torch.nn.functional as F


class MelSpectrogram(torch.nn.Module):

    def __init__(self,
                 fs=48000,
                 fft_size=1536,
                 hop_size=384,
                 win_length=None,
                 window="hann",
                 num_mels=100,
                 fmin=80,
                 fmax=None,
                 center=True,
                 normalized=False,
                 onesided=True,
                 eps=1e-10,
                 log_base=10.0, ):
        super().__init__()
        self.fft_size = fft_size
        if win_length is None:
            self.win_length = fft_size
        else:
            self.win_length = win_length
        self.hop_size = hop_size
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        if window is not None and not hasattr(torch, f"{window}_window"):
            raise ValueError(f"{window} window is not implemented")
        self.window = window
        self.eps = eps

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        melmat = librosa.filters.mel(sr=fs,
                                     n_fft=fft_size,
                                     n_mels=num_mels,
                                     fmin=fmin,
                                     fmax=fmax, )
        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())
        self.stft_params = {
            "n_fft": self.fft_size,
            "win_length": self.win_length,
            "hop_length": self.hop_size,
            "center": self.center,
            "normalized": self.normalized,
            "onesided": self.onesided,
        }
        self.stft_params["return_complex"] = False

        self.log_base = log_base
        if self.log_base is None:
            self.log = torch.log
        elif self.log_base == 2.0:
            self.log = torch.log2
        elif self.log_base == 10.0:
            self.log = torch.log10
        else:
            raise ValueError(f"log_base: {log_base} is not supported.")

    def forward(self, x):
        """
        Calculate Mel-spectrogram.

        Args:
            x (Tensor): Input waveform tensor (B, T) or (B, 1, T).

        Returns:
            Tensor: Mel-spectrogram (B, #mels, #frames).
        """
        if x.dim() == 3:
            # (B, C, T) -> (B*C, T)
            x = x.reshape(-1, x.size(2))

        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(self.win_length, dtype=x.dtype, device=x.device)
        else:
            window = None

        x_stft = torch.stft(x, window=window, **self.stft_params)
        # (B, #freqs, #frames, 2) -> (B, $frames, #freqs, 2)
        x_stft = x_stft.transpose(1, 2)
        x_power = x_stft[..., 0] ** 2 + x_stft[..., 1] ** 2
        x_amp = torch.sqrt(torch.clamp(x_power, min=self.eps))

        x_mel = torch.matmul(x_amp, self.melmat)
        x_mel = torch.clamp(x_mel, min=self.eps)

        return self.log(x_mel).transpose(1, 2)


class MelSpectrogramLoss(torch.nn.Module):

    def __init__(self,
                 fs=48000,
                 fft_size=1536,
                 hop_size=384,
                 win_length=None,
                 window="hann",
                 num_mels=100,
                 fmin=80,
                 fmax=None,
                 center=True,
                 normalized=False,
                 onesided=True,
                 eps=1e-10,
                 log_base=10.0, ):
        super().__init__()
        self.mel_spectrogram = MelSpectrogram(fs=fs,
                                              fft_size=fft_size,
                                              hop_size=hop_size,
                                              win_length=win_length,
                                              window=window,
                                              num_mels=num_mels,
                                              fmin=fmin,
                                              fmax=fmax,
                                              center=center,
                                              normalized=normalized,
                                              onesided=onesided,
                                              eps=eps,
                                              log_base=log_base, )

    def forward(self, y_hat, y):
        """
        Calculate Mel-spectrogram loss.

        Args:
            y_hat (Tensor): Generated single tensor (B, 1, T).
            y (Tensor): Groundtruth single tensor (B, 1, T).

        Returns:
            Tensor: Mel-spectrogram loss value.
        """
        mel_hat = self.mel_spectrogram(y_hat)
        mel = self.mel_spectrogram(y)
        mel_loss = F.l1_loss(mel_hat, mel)

        return mel_loss
