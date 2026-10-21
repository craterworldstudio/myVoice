from utils.config import config
import os

import numpy as np
import torch

from backend.onnx_model import ONNX_Model
from backend.mel import MelSpectrogram

class RMVPE(ONNX_Model):

    def __init__(self, rmvpe_path):
        super().__init__(rmvpe_path)
        self.rmvpe_pth = rmvpe_path


        target_sr = 16000
        try:
            target_sr = config.dsp.resampler.rspl_target
        except Exception:
            pass

        self.mel = MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            hop_length=160,
            n_mels=128,
            fmin=30,
            fmax=8000,
        )

        cents = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents, (4, 4))

    def extract(self, audio, pitch_shift=0):

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio.squeeze(1)

        mel = self.mel(audio)
    
        hidden = self._infer(mel)
        f0 = self.decode(hidden)

        if pitch_shift != 0:
            f0 = f0 * (2.0 ** (pitch_shift / 12.0))

        return self.f0_to_coarse(f0), f0

    
    def decode(self, hidden, threshold=0.05):
        

        if hidden.ndim == 3:
            hidden = hidden.squeeze(0)

        cents = self.to_local_average_cents(hidden, threshold)

        f0 = 10.0 * (2.0 ** (cents / 1200.0))
        f0[f0 == 10.0] = 0.0

        return f0


    def to_local_average_cents(self, salience, threshold=0.05):
        """
        Decode the RMVPE salience map using local weighted averaging. nothing else...
        """

        center = np.argmax(salience, axis=1)

        salience = np.pad(
            salience,
            ((0, 0), (4, 4)),
            mode="constant"
        )

        center += 4

        local_salience = []
        local_cents = []

        for i in range(salience.shape[0]):

            start = center[i] - 4
            end = center[i] + 5

            local_salience.append(
                salience[i, start:end]
            )

            local_cents.append(
                self.cents_mapping[start:end]
            )

        local_salience = np.asarray(local_salience)
        local_cents = np.asarray(local_cents)

        weighted_sum = np.sum(
            local_salience * local_cents,
            axis=1
        )

        weight = np.sum(
            local_salience,
            axis=1
        )

        cents = weighted_sum / weight

        confidence = np.max(salience, axis=1)

        cents[confidence <= threshold] = 0.0

        return cents
    
    def f0_to_coarse(self, f0):
        f0_mel = 1127.0 * np.log1p(f0 / 700.0)

        f0_mel_min = 1127.0 * np.log1p(50.0 / 700.0)
        f0_mel_max = 1127.0 * np.log1p(1100.0 / 700.0)

        voiced = f0 > 0
        f0_coarse = np.zeros_like(f0, dtype=np.int64)

        if np.any(voiced):
            coarse_vals = (
                (f0_mel[voiced] - f0_mel_min)
                * 254.0
                / (f0_mel_max - f0_mel_min)
                + 1.0
            )
            f0_coarse[voiced] = np.clip(np.rint(coarse_vals), 1, 255).astype(np.int64)

        return f0_coarse