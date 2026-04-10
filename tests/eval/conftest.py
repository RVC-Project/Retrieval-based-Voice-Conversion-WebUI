import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def sine_wav(tmp_path):
    """Generate a sine wave WAV file."""

    def _make(freq=440, duration=2.0, sr=48000, filename="sine.wav"):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        path = tmp_path / filename
        sf.write(str(path), audio, sr)
        return str(path)

    return _make


@pytest.fixture
def harmonic_wav(tmp_path):
    """Generate a harmonic-rich WAV file that pyworld.harvest can detect as voiced."""

    def _make(freq=440, duration=2.0, sr=48000, n_harmonics=8, filename="harmonic.wav"):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.zeros_like(t)
        for h in range(1, n_harmonics + 1):
            audio += (0.5 / h) * np.sin(2 * np.pi * freq * h * t)
        audio = audio.astype(np.float32)
        path = tmp_path / filename
        sf.write(str(path), audio, sr)
        return str(path)

    return _make
