"""Rate estimation from the amplitude envelope - one algorithm, three meanings.

    heart        -> heart rate        (beats per minute)
    respiratory  -> breathing rate    (breaths per minute)
    music        -> tempo             (beats per minute)

The chain is the same for all three, which is the point of a domain-agnostic
DSP front end:

    signal -> envelope (rectify / onset flux) -> low-pass -> decimate
           -> autocorrelation -> strongest peak inside a physiological /
              musical lag window -> parabolic interpolation -> rate

The envelope is what differs: a homomorphic (log-magnitude, low-passed)
envelope for heart and lung sounds, and a spectral-flux onset envelope for
music, where beats are marked by sudden broadband energy rather than by
loudness alone.
"""
import numpy as np
from scipy import signal as sps

from .features import stft_power

# Search window (rate units per minute) and envelope rate per domain.
RATE_SPECS = {
    "heart":       {"name": "Heart rate", "unit": "bpm", "lo": 40, "hi": 160,
                    "env_fs": 50.0, "env_lp": 8.0, "min_s": 3.0},
    "respiratory": {"name": "Breathing rate", "unit": "breaths/min", "lo": 8, "hi": 40,
                    "env_fs": 10.0, "env_lp": 1.0, "min_s": 8.0},
    "music":       {"name": "Tempo", "unit": "bpm", "lo": 60, "hi": 180,
                    "env_fs": 100.0, "env_lp": None, "min_s": 3.0},
}


def homomorphic_envelope(x, sr, lp_hz, out_fs):
    """exp(low-pass(log|x|)) - the standard heart-sound segmentation envelope
    (Schmidt et al. 2010). Taking the log first turns the multiplicative
    envelope x carrier model into an additive one the low-pass can separate."""
    a = np.abs(x) + 1e-6
    sos = sps.butter(2, lp_hz / (sr / 2), btype="low", output="sos")
    env = np.exp(sps.sosfiltfilt(sos, np.log(a)))
    step = max(1, int(round(sr / out_fs)))
    return env[::step], sr / step


def onset_envelope(x, sr, hop_ms=10.0):
    """Spectral flux: half-wave rectified frame-to-frame rise in log power,
    summed over frequency. Peaks where a note or drum hit starts."""
    _, power, _, _, _ = stft_power(x, sr, 2 * hop_ms + 20.0, hop_ms)
    logp = np.log(power + 1e-10)
    flux = np.maximum(np.diff(logp, axis=0), 0.0).sum(axis=1)
    flux = flux - sps.medfilt(flux, 31)               # remove the slow loudness trend
    flux = np.convolve(np.maximum(flux, 0.0), np.hanning(5) / 2.0, mode="same")
    return flux, 1000.0 / hop_ms


def autocorr(e):
    e = e - e.mean()
    n = len(e)
    ac = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * n)) ** 2)[:n]
    return ac / (ac[0] + 1e-12)


def _parabolic(ac, i):
    if 0 < i < len(ac) - 1:
        a, b, c = ac[i - 1], ac[i], ac[i + 1]
        d = a - 2 * b + c
        if d != 0:
            return i + 0.5 * (a - c) / d
    return float(i)


def estimate_rate(x, sr, domain):
    """Returns a dict with the rate, a 0-1 periodicity score and the
    intermediate signals, so the GUI can plot exactly what was measured."""
    spec = RATE_SPECS[domain]
    dur = len(x) / sr
    out = {"name": spec["name"], "unit": spec["unit"], "rate": None,
           "periodicity": 0.0, "duration": dur}
    if dur < spec["min_s"]:
        out["reason"] = f"needs at least {spec['min_s']:.0f} s of audio"
        return out

    if domain == "music":
        env, fs = onset_envelope(x, sr)
    else:
        env, fs = homomorphic_envelope(x, sr, spec["env_lp"], spec["env_fs"])
    ac = autocorr(env)
    lags = np.arange(len(ac)) / fs

    lo_lag = 60.0 / spec["hi"]
    hi_lag = min(60.0 / spec["lo"], lags[-1] * 0.66)
    win = np.where((lags >= lo_lag) & (lags <= hi_lag))[0]
    if len(win) < 3:
        out["reason"] = "recording too short for this rate range"
        return out

    weight = np.ones(len(win))
    if domain == "music":
        # Mild log-Gaussian preference for ~120 bpm against octave errors.
        bpm = 60.0 / lags[win]
        weight = np.exp(-0.5 * (np.log2(bpm / 120.0) / 1.0) ** 2)
    peaks, _ = sps.find_peaks(ac[win])
    if len(peaks) == 0:
        out["reason"] = "no periodicity found"
        return out
    score = ac[win][peaks] * weight[peaks]
    if domain == "music":
        best = peaks[np.argmax(score)]
    else:
        # A periodic envelope correlates with itself at every multiple of the
        # period, and those multiples are almost as tall as the first one.
        # Take the shortest lag that is nearly as strong as the best, so a
        # 120 bpm heart is not reported as 60.
        best = peaks[np.argmax(score >= 0.85 * score.max())]
    i = win[best]
    lag = _parabolic(ac, i) / fs

    out.update(rate=60.0 / lag, periodicity=float(np.clip(ac[i], 0, 1)),
               env=env, env_fs=fs, lags=lags, ac=ac,
               window=(lo_lag, hi_lag), lag=lag)

    # Event markers (beats / breaths) for the plot: envelope peaks at least
    # 60 % of a period apart.
    dist = max(1, int(0.6 * lag * fs))
    pk, _ = sps.find_peaks(env, distance=dist,
                           height=np.percentile(env, 60))
    out["events"] = pk / fs
    return out
