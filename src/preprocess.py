"""M2 - Pre-processing: resampling, band-pass filtering, noise reduction,
amplitude normalization and frame-based segmentation. Domain-agnostic."""
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
from scipy import signal as sps
from scipy.io import wavfile


AU_ENCODINGS = {1: (np.int8, 8), 2: (np.int8, 8), 3: (">i2", 16),
                4: (">i4", 24), 5: (">i4", 32), 6: (">f4", 32), 7: (">f8", 64)}


def read_au(path):
    """Sun AU reader. GTZAN ships as .au and Python removed `sunau` in 3.13."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:4] != b".snd":
        raise ValueError(f"{path}: not a Sun AU file")
    offset, _, encoding, sr, ch = (
        int.from_bytes(raw[i:i + 4], "big") for i in range(4, 24, 4))
    if encoding not in AU_ENCODINGS:
        raise ValueError(f"{path}: unsupported AU encoding {encoding}")
    dtype, _ = AU_ENCODINGS[encoding]
    x = np.frombuffer(raw[offset:], dtype=dtype)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    x = x.astype(np.float64)
    if np.issubdtype(np.dtype(dtype), np.integer):
        x /= float(np.iinfo(np.dtype(dtype)).max)
    return x, int(sr)


COMPRESSED = {".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}


def read_compressed(path):
    """Decodes MP3 and friends by piping raw float32 PCM out of ffmpeg, so the
    GUI can open any song the examiner hands it, not just .wav/.au."""
    ff = shutil.which("ffmpeg")
    if ff is None:
        raise RuntimeError(
            f"{Path(path).suffix} needs ffmpeg on PATH. Install it with "
            "`winget install Gyan.FFmpeg`, or convert the file to .wav first.")

    sr = 44100
    probe = shutil.which("ffprobe")
    if probe:
        out = subprocess.run(
            [probe, "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=sample_rate", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip().isdigit():
            sr = int(out.stdout.strip())

    proc = subprocess.run(
        [ff, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
         "-ar", str(sr), "-"],
        capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg could not decode {path}: "
                           f"{proc.stderr.decode(errors='replace')[:200]}")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float64), sr


def read_audio(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".au":
        x, sr = read_au(path)
    elif suffix in COMPRESSED:
        x, sr = read_compressed(path)
    else:
        # Some WAV writers add metadata chunks SciPy does not parse; it skips
        # them with a warning, and the audio itself is read correctly.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", wavfile.WavFileWarning)
            sr, x = wavfile.read(str(path))
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    return x.astype(np.float64), int(sr)


read_wav = read_audio


def to_mono(x):
    return x.mean(axis=1) if x.ndim > 1 else x


def resample_to(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    g = np.gcd(int(sr_in), int(sr_out))
    return sps.resample_poly(x, sr_out // g, sr_in // g)


def bandpass(x, sr, band, order=4):
    lo, hi = band
    nyq = sr / 2.0
    hi = min(hi, 0.99 * nyq)
    lo = max(lo, 1.0)
    if lo >= hi:
        return x
    sos = sps.butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return sps.sosfiltfilt(sos, x)


def denoise(x, sr, frame_len=1024, hop=256, over_sub=1.5, floor_gain=0.05):
    """Spectral-subtraction noise reduction; noise floor from the quietest
    10% of frames, so it does not assume a leading silent segment."""
    if len(x) < frame_len * 2:
        return x
    f, t, Z = sps.stft(x, fs=sr, nperseg=frame_len, noverlap=frame_len - hop)
    mag, phase = np.abs(Z), np.angle(Z)
    frame_energy = mag.sum(axis=0)
    k = max(1, int(0.10 * len(frame_energy)))
    quiet = np.argsort(frame_energy)[:k]
    noise = mag[:, quiet].mean(axis=1, keepdims=True)
    clean = np.maximum(mag - over_sub * noise, floor_gain * mag)
    _, y = sps.istft(clean * np.exp(1j * phase), fs=sr,
                     nperseg=frame_len, noverlap=frame_len - hop)
    return y[:len(x)]


def normalize(x, mode="peak"):
    if mode == "rms":
        r = np.sqrt(np.mean(x ** 2))
        return x / (r + 1e-12)
    return x / (np.max(np.abs(x)) + 1e-12)


def frame_signal(x, frame_len, hop_len, window="hamming"):
    """Frame-based segmentation with a Hamming window (M2 output -> M3 input)."""
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n = 1 + (len(x) - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + hop_len * np.arange(n)[:, None]
    frames = x[idx]
    if window == "hamming":
        frames = frames * np.hamming(frame_len)[None, :]
    return frames


def fix_length(x, sr, seconds):
    n = int(round(sr * seconds))
    if len(x) >= n:
        start = (len(x) - n) // 2
        return x[start:start + n]
    return np.pad(x, (0, n - len(x)))


def preprocess(x, sr_in, cfg, do_denoise=True):
    """Full M2 chain for one recording; returns (signal, sr)."""
    sr = cfg["sr"]
    x = to_mono(np.asarray(x, dtype=np.float64))
    x = resample_to(x, sr_in, sr)
    if do_denoise:
        x = denoise(x, sr)
    x = bandpass(x, sr, cfg["band"])
    x = normalize(x, "peak")
    x = fix_length(x, sr, cfg["clip_s"])
    return x, sr
