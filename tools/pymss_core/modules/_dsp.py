"""Small DSP helpers needed by model definitions."""

from __future__ import annotations

import numpy as np


def hz_to_midi(hz):
    """Convert frequencies in Hz to MIDI note numbers."""
    hz = np.asarray(hz)
    return 69.0 + 12.0 * np.log2(hz / 440.0)


def midi_to_hz(midi):
    """Convert MIDI note numbers to frequencies in Hz."""
    midi = np.asarray(midi)
    return 440.0 * np.power(2.0, (midi - 69.0) / 12.0)


def _hz_to_mel(frequencies, *, htk=False):
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    f_sp = 200.0 / 3
    mels = frequencies / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = frequencies >= min_log_hz
    mels = np.array(mels, copy=True)
    mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels, *, htk=False):
    mels = np.asarray(mels, dtype=np.float64)
    if htk:
        return 700.0 * (np.power(10.0, mels / 2595.0) - 1.0)

    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs = np.array(freqs, copy=True)
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


def mel_frequencies(n_mels, *, fmin=0.0, fmax=11025.0, htk=False):
    """Return center frequencies on the mel scale, including endpoints."""
    min_mel = _hz_to_mel(fmin, htk=htk)
    max_mel = _hz_to_mel(fmax, htk=htk)
    return _mel_to_hz(np.linspace(min_mel, max_mel, int(n_mels)), htk=htk)


def fft_frequencies(*, sr, n_fft):
    """Return FFT bin center frequencies."""
    return np.linspace(0.0, float(sr) / 2.0, int(1 + n_fft // 2), endpoint=True)


def mel_filterbank(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm="slaney", dtype=np.float32):
    """Create a triangular mel filterbank for model initialization."""
    if fmax is None:
        fmax = float(sr) / 2.0

    mel_f = mel_frequencies(int(n_mels) + 2, fmin=fmin, fmax=fmax, htk=htk)
    fft_f = fft_frequencies(sr=sr, n_fft=n_fft)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_f)
    lower = -ramps[:-2] / fdiff[:-1, np.newaxis]
    upper = ramps[2:] / fdiff[1:, np.newaxis]
    weights = np.maximum(0.0, np.minimum(lower, upper))

    if norm == "slaney":
        enorm = 2.0 / (mel_f[2 : int(n_mels) + 2] - mel_f[: int(n_mels)])
        weights *= enorm[:, np.newaxis]
    elif norm is not None:
        raise ValueError(f"Unsupported mel filterbank norm: {norm!r}")

    return weights.astype(dtype, copy=False)
