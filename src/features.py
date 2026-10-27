"""Sound -> log-mel spectrogram, written out by hand so every step can be explained (lesson 02).

Pipeline for one 5-second clip at 32 kHz (160,000 samples):
  1. STFT: cut the signal into overlapping windows (4096 samples = 128 ms, moved 320 samples = 10 ms
     each step) and take the FFT of each -> how much energy at each frequency, over time.
  2. Power: |FFT|^2.
  3. Mel filterbank: merge the 2049 FFT frequency bins into 128 bands that are narrow at low
     frequencies and wide at high ones (ship tonals live at low frequencies, so we want detail there).
  4. Decibels: 10*log10(power). Loudness becomes additive, and the network sees a sane value range.

We deliberately do NOT normalise each clip separately: absolute loudness is the main clue for distance.
"""
import io

import numpy as np
import soundfile as sf
import torch

SR = 32_000
CLIP_SECONDS = 5.0
CLIP_SAMPLES = int(SR * CLIP_SECONDS)       # 160,000
N_FFT = 4096                                # 128 ms window -> 7.8 Hz frequency resolution
HOP = 320                                   # 10 ms between frames
N_MELS = 128
FMIN, FMAX = 10.0, SR / 2
N_FRAMES = 1 + CLIP_SAMPLES // HOP          # 501 frames (center=True)
POWER_FLOOR = 1e-12                         # -120 dB, avoids log(0)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    """Triangular filters, equally spaced on the mel scale. Shape (n_mels, n_fft//2 + 1)."""
    fft_freqs = np.linspace(0.0, sr / 2, n_fft // 2 + 1)
    edges_hz = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, fft_freqs.size), dtype=np.float32)
    for m in range(n_mels):
        lo, centre, hi = edges_hz[m], edges_hz[m + 1], edges_hz[m + 2]
        rising = (fft_freqs - lo) / (centre - lo)
        falling = (hi - fft_freqs) / (hi - centre)
        fb[m] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb


class LogMel(torch.nn.Module):
    """Batch of waveforms (B, samples) -> log-mel in dB (B, N_MELS, frames)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        self.register_buffer("fb", torch.from_numpy(mel_filterbank()), persistent=False)

    @torch.no_grad()
    def forward(self, wave):
        spec = torch.stft(wave, N_FFT, HOP, window=self.window, center=True,
                          pad_mode="reflect", return_complex=True)
        power = spec.real ** 2 + spec.imag ** 2                   # (B, 2049, frames)
        mel = torch.matmul(self.fb, power)                         # (B, 128, frames)
        return 10.0 * torch.log10(mel.clamp_min(POWER_FLOOR))


def decode_wav(data):
    """WAV bytes (or a path) -> mono float32 waveform in [-1, 1], plus sample rate and sample format."""
    src = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data
    with sf.SoundFile(src) as f:
        wave = f.read(dtype="float32", always_2d=True)
        sr, subtype = f.samplerate, f.subtype
    return wave.mean(axis=1), sr, subtype


def resample(wave, sr_in, sr_out=SR):
    if sr_in == sr_out:
        return wave
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(wave, sr_out // g, sr_in // g).astype(np.float32)


def fix_length(wave, n=CLIP_SAMPLES):
    """Crop to n samples, or pad a short clip by repeating it (better than silence for ocean noise)."""
    if wave.size >= n:
        return wave[:n]
    return np.pad(wave, (0, n - wave.size), mode="wrap")
