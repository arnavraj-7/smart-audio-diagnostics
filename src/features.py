"""M3 - Feature extraction. MFCC is implemented from first principles
(pre-emphasis -> framing -> windowing -> FFT -> Mel filter bank -> log -> DCT)
rather than called from a library, since it is the core DSP deliverable.
Same code path is used by the music branch and the medical branch."""
import numpy as np
from scipy.fft import dct

from .preprocess import frame_signal

EPS = 1e-12


def pre_emphasis(x, alpha=0.97):
    return np.append(x[0], x[1:] - alpha * x[:-1])


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    fmax = fmax if fmax is not None else sr / 2.0
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c == l:
            c = min(l + 1, n_fft // 2)
        if r == c:
            r = min(c + 1, n_fft // 2)
        if c > l:
            fb[m - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[m - 1, c:r] = (r - np.arange(c, r)) / (r - c)
    # Slaney-style area normalization keeps filters comparable across widths
    enorm = 2.0 / (hz[2:n_mels + 2] - hz[:n_mels] + EPS)
    return fb * enorm[:, None]


def stft_power(x, sr, frame_ms, hop_ms, n_fft=None):
    """Framed magnitude/power spectra - the shared FFT stage (Objective O2)."""
    frame_len = int(round(sr * frame_ms / 1000.0))
    hop_len = max(1, int(round(sr * hop_ms / 1000.0)))
    n_fft = n_fft or int(2 ** np.ceil(np.log2(frame_len)))
    frames = frame_signal(x, frame_len, hop_len)
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1))
    power = (spec ** 2) / n_fft
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return spec, power, freqs, frames, n_fft


def mfcc(x, sr, cfg, n_fft=None):
    """13 MFCCs per frame -> shape (n_frames, n_mfcc)."""
    xp = pre_emphasis(x)
    _, power, _, _, _ = stft_power(xp, sr, cfg["frame_ms"], cfg["hop_ms"], n_fft)
    fb = mel_filterbank(sr, (power.shape[1] - 1) * 2, cfg["n_mels"],
                        fmin=cfg["band"][0], fmax=min(cfg["band"][1], sr / 2))
    mel_energy = power @ fb.T
    log_mel = np.log(mel_energy + EPS)
    return dct(log_mel, type=2, axis=1, norm="ortho")[:, :cfg["n_mfcc"]]


def delta(feat, width=2):
    if feat.shape[0] < 3:
        return np.zeros_like(feat)
    pad = np.pad(feat, ((width, width), (0, 0)), mode="edge")
    n = np.arange(-width, width + 1)[:, None]
    out = sum(n[i] * pad[i:i + feat.shape[0]] for i in range(len(n)))
    return out / (2.0 * np.sum(np.arange(1, width + 1) ** 2))


def spectral_centroid(power, freqs):
    p = power.sum(axis=1) + EPS
    return (power @ freqs) / p


def spectral_bandwidth(power, freqs, centroid):
    p = power.sum(axis=1) + EPS
    dev = (freqs[None, :] - centroid[:, None]) ** 2
    return np.sqrt((power * dev).sum(axis=1) / p)


def spectral_rolloff(power, freqs, roll=0.85):
    c = np.cumsum(power, axis=1)
    thr = roll * c[:, -1:]
    idx = np.argmax(c >= thr, axis=1)
    return freqs[idx]


def spectral_flatness(power):
    gm = np.exp(np.mean(np.log(power + EPS), axis=1))
    am = np.mean(power, axis=1) + EPS
    return gm / am


def zero_crossing_rate(frames):
    return np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)


def short_time_energy(frames):
    return np.sum(frames ** 2, axis=1) / frames.shape[1]


# Feature groups, kept explicit so the ablation study (O5) can drop one at a time.
GROUPS = ("mfcc", "spectral", "temporal")


def extract(x, sr, cfg):
    """Returns (vector, names). Frame-level features are summarised by mean and
    standard deviation to give one fixed-length vector per recording."""
    spec, power, freqs, frames, n_fft = stft_power(
        x, sr, cfg["frame_ms"], cfg["hop_ms"])

    m = mfcc(x, sr, cfg, n_fft=n_fft)
    dm = delta(m)

    cen = spectral_centroid(power, freqs)
    bw = spectral_bandwidth(power, freqs, cen)
    roll = spectral_rolloff(power, freqs)
    flat = spectral_flatness(power)

    raw_frames = frame_signal(x, frames.shape[1],
                              max(1, int(round(sr * cfg["hop_ms"] / 1000.0))),
                              window=None)
    zcr = zero_crossing_rate(raw_frames)
    ste = short_time_energy(raw_frames)

    vals, names = [], []

    def add(group, name, arr):
        vals.extend([np.mean(arr), np.std(arr)])
        names.extend([f"{group}::{name}_mean", f"{group}::{name}_std"])

    for i in range(m.shape[1]):
        add("mfcc", f"mfcc{i + 1}", m[:, i])
    for i in range(dm.shape[1]):
        vals.append(np.mean(dm[:, i]))
        names.append(f"mfcc::dmfcc{i + 1}_mean")

    add("spectral", "centroid", cen)
    add("spectral", "bandwidth", bw)
    add("spectral", "rolloff", roll)
    add("spectral", "flatness", flat)

    add("temporal", "zcr", zcr)
    add("temporal", "energy", ste)
    add("temporal", "rms", np.sqrt(ste))

    v = np.asarray(vals, dtype=np.float64)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), names


def group_mask(names, keep):
    """Boolean mask selecting the feature groups in `keep`."""
    return np.array([n.split("::")[0] in keep for n in names])
