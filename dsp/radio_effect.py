import numpy as np
from scipy import signal
from core.stage import Stage

class RadioEffect(Stage):
    """
    Broadcast-grade 1930s vintage radio demon effect processor for Alastor.
    Combines:
    1. Bandpass telephone EQ (350 Hz to 3800 Hz)
    2. Megaphone horn resonance peak (~1850 Hz, Q=3.0)
    3. Asymmetrical warm tube overdrive/saturation
    4. Subtle radio carrier amplitude modulation (32 Hz)
    5. Vintage slapback / spring reflection (25-30ms)
    """

    def __init__(self, sample_rate=48000):
        super().__init__("radio_effect")
        self.sample_rate = sample_rate
        self.enabled = False

        # Bandpass filter (350 Hz to 3800 Hz)
        self.bp_b, self.bp_a = signal.butter(
            4,
            [350.0 / (sample_rate / 2.0), 3800.0 / (sample_rate / 2.0)],
            btype="bandpass"
        )
        self.bp_zi = signal.lfilter_zi(self.bp_b, self.bp_a)

        # Peaking filter at 1850 Hz (megaphone horn resonance)
        f0 = 1850.0
        q = 3.0
        gain_db = 7.0
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * f0 / sample_rate
        alpha = np.sin(w0) / (2.0 * q)
        b0 = 1.0 + alpha * A
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha / A

        self.peak_b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float32)
        self.peak_a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float32)
        self.peak_zi = signal.lfilter_zi(self.peak_b, self.peak_a)

        # Carrier ring mod phase
        self.carrier_phase = 0.0
        self.carrier_freq = 32.0  # 32 Hz vintage radio hum

        # Slapback delay buffer (~26ms delay)
        self.delay_samples = int(round(0.026 * sample_rate))  # 1248 samples at 48k
        self.delay_buf = np.zeros(self.delay_samples, dtype=np.float32)
        self.delay_pos = 0

        # Output makeup gain
        self.drive = 2.2
        self.makeup = 1.4

    def reset(self):
        self.bp_zi = signal.lfilter_zi(self.bp_b, self.bp_a)
        self.peak_zi = signal.lfilter_zi(self.peak_b, self.peak_a)
        self.carrier_phase = 0.0
        self.delay_buf.fill(0.0)
        self.delay_pos = 0

    def process(self, audio):
        if not self.enabled:
            return audio

        audio = np.asarray(audio, dtype=np.float32)
        orig_shape = audio.shape
        x = audio.flatten()
        n = len(x)
        if n == 0:
            return audio

        # 1. Bandpass Telephone Filtering
        x_bp, self.bp_zi = signal.lfilter(self.bp_b, self.bp_a, x, zi=self.bp_zi)

        # 2. Megaphone Peak Resonance
        x_peak, self.peak_zi = signal.lfilter(self.peak_b, self.peak_a, x_bp, zi=self.peak_zi)

        # 3. Asymmetric Tube Overdrive & Harmonic Distortion
        x_driven = x_peak * self.drive
        sat = np.tanh(x_driven) + 0.12 * np.tanh(2.5 * (x_driven ** 2) * np.sign(x_driven))

        # 4. Subtle 32 Hz Radio Carrier Ring Modulation
        t = np.arange(n, dtype=np.float32)
        carrier = 1.0 + 0.10 * np.sin(self.carrier_phase + 2.0 * np.pi * self.carrier_freq * t / self.sample_rate)
        self.carrier_phase = float((self.carrier_phase + 2.0 * np.pi * self.carrier_freq * n / self.sample_rate) % (2.0 * np.pi))
        x_mod = sat * carrier

        # 5. Vintage Slapback Cabinet Delay
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            delayed_sample = self.delay_buf[self.delay_pos]
            current_sample = x_mod[i]
            self.delay_buf[self.delay_pos] = current_sample + delayed_sample * 0.28
            self.delay_pos = (self.delay_pos + 1) % self.delay_samples
            out[i] = current_sample + delayed_sample * 0.40

        out = out * self.makeup
        return out.reshape(orig_shape)
