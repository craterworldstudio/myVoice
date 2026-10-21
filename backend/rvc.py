import sys, os, threading

import numpy as np
import torch
import torch.nn.functional as F

from core.stage import Stage
from backend.loader import RVCLoader
from backend.infer import RVCInfer
from backend.feature_extractor import FeatureExtractor
from backend.rmvpe import RMVPE

import soundfile as sf

from utils.config import config

class RVC(Stage):
    def __init__(self, model_path=None, pitch=None):
        super().__init__("rvc")

        self.lock = threading.Lock()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) # Works for ROCm (HIP) and CUDA

        self.hop_size = int(getattr(config.rvc, "hop_size", 3840)) # 240ms of new audio per hop at 16kHz
        self.context_size = int(getattr(config.rvc, "context_size", 2560)) # 160ms context window
        self.window_size = self.hop_size + self.context_size # 6400 samples (400ms)


        self.input_acc = np.empty((0, 1), dtype=np.float32)
        self.history = np.zeros((self.context_size, 1), dtype=np.float32)
        self.output_queue = np.empty((0, 1), dtype=np.float32)
        self.last_sample = None

        self.path_pth = model_path or config.rvc.model
        self.index_pth = getattr(config.rvc, "index", "")
        self.pitch = pitch if pitch is not None else getattr(config.rvc, "pitch", 0)

        self.loader = RVCLoader(self.path_pth, self.device)
        loaded = self.loader.load()

        self.model = loaded["net_g"]
        self.sr = int(loaded["sr"])
        self.version = loaded["version"]
        self.f0 = loaded["f0"]

        self.feature_extractor = FeatureExtractor(config.rvc.vec, self.device)
        if getattr(config.rvc, "use_onnx", True):
            self.rmvpe_path = config.rvc.rmvpe_onnx
        else:
            self.rmvpe_path = config.rvc.rmvpe_pt

        self.rmvpe = RMVPE(self.rmvpe_path)
        self.inference = RVCInfer(self.model, self.device)

        # Warm up GPU kernel compilation so live audio streaming starts immediately without delay
        try:
            warm_audio = np.zeros(self.window_size, dtype=np.float32)
            w_p = self.feature_extractor.extract(warm_audio)
            w_pitch, w_pitchf = self.rmvpe.extract(warm_audio, pitch_shift=self.pitch)
            _ = self.inference.infer(w_p, w_pitch, w_pitchf, sid=0)
        except Exception:
            pass

    def reset_buffers(self):
        with self.lock:
            self.input_acc = np.empty((0, 1), dtype=np.float32)
            self.history = np.zeros((self.context_size, 1), dtype=np.float32)
            self.output_queue = np.empty((0, 1), dtype=np.float32)
            self.last_sample = None

    def load_model(self, model_path, index_path=None):
        with self.lock:
            self.path_pth = model_path
            self.loader = RVCLoader(self.path_pth, self.device)
            loaded = self.loader.load()

            self.model = loaded["net_g"]
            self.sr = int(loaded["sr"])
            self.version = loaded["version"]
            self.f0 = loaded["f0"]

            if index_path is not None:
                self.index_pth = index_path
            else:
                candidate_index = os.path.splitext(model_path)[0] + ".index"
                if os.path.exists(candidate_index):
                    self.index_pth = candidate_index
                else:
                    self.index_pth = ""

            self.inference = RVCInfer(self.model, self.device)

            # Reset audio buffers
            self.input_acc = np.empty((0, 1), dtype=np.float32)
            self.history = np.zeros((self.context_size, 1), dtype=np.float32)
            self.output_queue = np.empty((0, 1), dtype=np.float32)
            self.last_sample = None

            # Warm up GPU kernel compilation
            try:
                warm_audio = np.zeros(self.window_size, dtype=np.float32)
                w_p = self.feature_extractor.extract(warm_audio)
                w_pitch, w_pitchf = self.rmvpe.extract(warm_audio, pitch_shift=self.pitch)
                _ = self.inference.infer(w_p, w_pitch, w_pitchf, sid=0)
            except Exception:
                pass

    def set_parameters(self, pitch=None):
        with self.lock:
            if pitch is not None:
                self.pitch = pitch

    def get_parameters(self):
        with self.lock:
            return {
                "pitch": self.pitch,
                "sr": self.sr,
                "model": self.path_pth,
            }

    def process(self, audio):
        with self.lock:
            if not self.enabled:
                return audio

            audio = np.asarray(audio, dtype=np.float32).reshape(-1, 1)
            in_len = len(audio)
            if in_len <= 0:
                return audio

            # Downsample incoming 48kHz chunk to 16kHz (exact 3:1 integer decimation)
            target_16k_len = int(round(in_len / 3.0))
            t_in = torch.from_numpy(audio.T).unsqueeze(0)
            down_16k = F.interpolate(t_in, size=target_16k_len, mode='linear', align_corners=False).squeeze().numpy().reshape(-1, 1)

            self.input_acc = np.concatenate((self.input_acc, down_16k), axis=0)

            # Process when we accumulate enough new audio for an inference hop
            while len(self.input_acc) >= self.hop_size:
                new_chunk = self.input_acc[:self.hop_size]
                self.input_acc = self.input_acc[self.hop_size:]

                # Build window: [past_context, new_chunk]
                window = np.concatenate((self.history, new_chunk), axis=0)
                self.history = window[-self.context_size:]

                if len(window) < self.window_size:
                    pad_needed = self.window_size - len(window)
                    window = np.pad(window, ((pad_needed, 0), (0, 0)), mode="constant")

                chunk_rms = float(np.sqrt(np.mean(new_chunk ** 2)))
                out_hop = int(round(self.hop_size * (self.sr / 16000.0)))

                if chunk_rms < 0.002:
                    # Silence or quiet room noise: output clean silence without synthesizing phantom buzz
                    out_slice = np.zeros((out_hop, 1), dtype=np.float32)
                    if self.last_sample is not None and abs(self.last_sample) > 1e-4:
                        fade_len = min(64, out_hop // 2)
                        ramp = np.linspace(self.last_sample, 0.0, fade_len, dtype=np.float32).reshape(-1, 1)
                        out_slice[:fade_len] = ramp
                    self.last_sample = 0.0
                else:
                    target_frames = self.window_size // 160  # e.g. 6400 // 160 = 40 frames
                    phone = self.feature_extractor.extract(window)
                    pitch, pitchf = self.rmvpe.extract(window, pitch_shift=self.pitch)
                    
                    # Ensure exact frame alignment across ContentVec and RMVPE
                    n_frames = min(phone.shape[1], pitch.shape[0], pitchf.shape[0], target_frames)
                    phone = phone[:, :n_frames]
                    pitch = pitch[:n_frames]
                    pitchf = pitchf[:n_frames]

                    audiop = self.inference.infer(phone, pitch, pitchf, sid=0)
                    out_slice = audiop[-out_hop:].reshape(-1, 1)

                # Smooth boundary transition
                if self.last_sample is not None:
                    fade_len = min(64, len(out_slice) // 2)
                    diff = self.last_sample - out_slice[0, 0]
                    if abs(diff) > 1e-4:
                        ramp = np.linspace(1.0, 0.0, fade_len, dtype=np.float32).reshape(-1, 1)
                        out_slice[:fade_len] += diff * ramp

                self.last_sample = out_slice[-1, 0]

                # Upsample the native SR hop slice directly to 48kHz
                out_hop_48k = int(round(self.hop_size * 3.0)) # 3840 * 3 = 11520 samples
                t_nat = torch.from_numpy(out_slice.T).unsqueeze(0)
                out_48k = F.interpolate(t_nat, size=out_hop_48k, mode='linear', align_corners=False).squeeze().numpy().reshape(-1, 1)
                self.output_queue = np.concatenate((self.output_queue, out_48k), axis=0)

            # Serve exactly in_len (e.g. 960) samples at 48kHz
            if len(self.output_queue) >= in_len:
                ret = self.output_queue[:in_len]
                self.output_queue = self.output_queue[in_len:]
                return ret
            else:
                # During initial buffer filling (~0.24s), return available samples padded with silence
                avail = len(self.output_queue)
                if avail > 0:
                    ret = np.pad(self.output_queue, ((0, in_len - avail), (0, 0)), mode="constant")
                    self.output_queue = np.empty((0, 1), dtype=np.float32)
                    return ret
                return np.zeros((in_len, 1), dtype=np.float32)