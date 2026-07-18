import torch
import torch.nn as nn
import torch.nn.functional as F


class MelSpectrogram(nn.Module):

    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=160,
        win_length=1024,
        n_mels=128,
        fmin=30,
        fmax=8000,
        clamp=1e-5,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin =  fmin
        self.fmax = fmax
        self.clamp = clamp

        self.register_buffer(
            "window",
            torch.hann_window(win_length)
        )

        self.register_buffer(
            "mel_basis",
            self._build_mel_filterbank()
        )

    def _hz_to_mel(self, hz):
        hz = torch.as_tensor(hz, dtype=torch.float32)
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel):
        mel = torch.as_tensor(mel, dtype=torch.float32)
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)



    def _build_mel_filterbank(self):
        n_freqs = self.n_fft // 2 + 1

        # Mel scale endpoints
        mel_min = self._hz_to_mel(self.fmin)
        mel_max = self._hz_to_mel(self.fmax)

        # Evenly spaced Mel points
        mel_points = torch.linspace(
            mel_min,
            mel_max,
            self.n_mels + 2
        )

        # Convert back to Hz
        hz_points = self._mel_to_hz(mel_points)

        # Convert Hz to FFT bin numbers
        bins = torch.floor(
            (self.n_fft + 1) * hz_points / self.sample_rate
        ).long()

        filterbank = torch.zeros(self.n_mels, n_freqs)

        for m in range(self.n_mels):

            left = bins[m].item()
            center = bins[m + 1].item()
            right = bins[m + 2].item()

            # Avoid zero-width triangles
            if center == left:
                center += 1
            if right == center:
                right += 1

            center = min(center, n_freqs - 1)
            right = min(right, n_freqs)

            # Rising slope
            if center > left:
                filterbank[m, left:center] = (
                    torch.arange(left, center) - left
                ) / (center - left)

            # Falling slope
            if right > center:
                filterbank[m, center:right] = (
                    right - torch.arange(center, right)
                ) / (right - center)

        return filterbank

    def forward(self, audio):

        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)

        audio = audio.float()
        audio = audio.flatten()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        #print("In Mel Audio Shape", audio.shape)
        stft = torch.stft(audio, 
                        n_fft=self.n_fft, 
                        hop_length=self.hop_length, 
                        win_length=self.win_length, 
                        window=self.window,
                        center=True,
                        return_complex=True)

        magnitude = stft.abs()

        mel = torch.matmul(self.mel_basis, magnitude)

        mel = torch.clamp(mel, min=self.clamp)

        return torch.log(mel)