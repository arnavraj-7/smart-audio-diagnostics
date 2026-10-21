"""Physically-motivated synthetic signal generators.

These stand in for the Kaggle datasets so the whole pipeline can be run,
tested and demonstrated before the audio downloads are in place. Every
generator is built from the acoustic property that actually separates the
classes (harmonicity, murmur noise, wheeze tonality, crackle transients),
so the reported numbers are a pipeline sanity check, NOT dataset results.
"""
import numpy as np
from scipy import signal as sps


def _env(n, sr, attack=0.01, decay=0.15):
    t = np.arange(n) / sr
    return np.minimum(t / attack, 1.0) * np.exp(-t / decay)


def degrade(x, sr, rng, snr_db=(3.0, 22.0), lp=(0.25, 0.95)):
    """Recording-condition variability: microphone colouration, additive
    background noise at a random SNR, random gain and a random time shift.
    Without this the synthetic classes are trivially separable and the
    pipeline check reports a meaningless 100%. `lp` is kept above the
    diagnostic passband so degradation does not simply erase the class cue."""
    b, a = sps.butter(2, min(0.98, rng.uniform(*lp)), btype="low")
    x = sps.lfilter(b, a, x)
    if rng.random() < 0.5:
        bh, ah = sps.butter(1, min(0.9, rng.uniform(0.005, 0.05)), btype="high")
        x = sps.lfilter(bh, ah, x)

    snr = rng.uniform(*snr_db)
    p_sig = np.mean(x ** 2) + 1e-12
    noise = rng.standard_normal(len(x))
    noise *= np.sqrt(p_sig / (10 ** (snr / 10.0)) / (np.mean(noise ** 2) + 1e-12))
    x = x + noise

    x = np.roll(x, int(rng.integers(0, len(x))))
    x *= rng.uniform(0.3, 1.0)
    return x / (np.max(np.abs(x)) + 1e-12)


# ---------------------------------------------------------------- music
def music_clip(genre, sr, dur, rng):
    n = int(sr * dur)
    t = np.arange(n) / sr
    x = np.zeros(n)

    if genre == "classical":
        f0 = rng.uniform(150, 400)
        for k in range(1, 9):
            x += (1.0 / k ** 1.3) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28))
        x *= 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.3, 0.8) * t)
        x += 0.02 * rng.standard_normal(n)

    elif genre == "jazz":
        f0 = rng.uniform(180, 500)
        for k in range(1, 7):
            det = rng.uniform(-3, 3)
            x += (1.0 / k ** 1.1) * np.sin(2 * np.pi * (f0 * k + det) * t)
        bpm = rng.uniform(100, 160)
        for b in np.arange(0, dur, 60.0 / bpm):
            i = int(b * sr)
            m = min(n - i, int(0.08 * sr))
            if m > 0:
                x[i:i + m] += 0.5 * rng.standard_normal(m) * _env(m, sr, 0.002, 0.03)
        x += 0.05 * rng.standard_normal(n)

    elif genre == "rock":
        f0 = rng.uniform(90, 260)
        for k in range(1, 12):
            x += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t)
        x = np.tanh(rng.uniform(1.2, 3.0) * x)                       # distortion -> broadband
        x += 0.25 * rng.standard_normal(n)
        bpm = rng.uniform(120, 180)
        for b in np.arange(0, dur, 60.0 / bpm):
            i = int(b * sr)
            m = min(n - i, int(0.05 * sr))
            if m > 0:
                x[i:i + m] += 0.9 * rng.standard_normal(m) * _env(m, sr, 0.001, 0.02)

    elif genre == "pop":
        f0 = rng.uniform(200, 480)
        for k in (1, 2, 3, 4):
            x += (1.0 / k ** 1.2) * np.sin(2 * np.pi * f0 * k * t)
        x += 0.35 * np.sin(2 * np.pi * f0 * 1.5 * t)
        bpm = rng.uniform(100, 140)
        for b in np.arange(0, dur, 60.0 / bpm):
            i = int(b * sr)
            m = min(n - i, int(0.06 * sr))
            if m > 0:
                x[i:i + m] += 0.6 * rng.standard_normal(m) * _env(m, sr, 0.001, 0.025)
        x += 0.08 * rng.standard_normal(n)

    else:  # hiphop - strong low-frequency beat, sparse mid content
        bpm = rng.uniform(75, 115)
        for b in np.arange(0, dur, 60.0 / bpm):
            i = int(b * sr)
            m = min(n - i, int(0.25 * sr))
            if m > 0:
                tt = np.arange(m) / sr
                x[i:i + m] += 1.4 * np.sin(2 * np.pi * 55 * tt) * _env(m, sr, 0.002, 0.12)
        for b in np.arange(60.0 / bpm / 2, dur, 60.0 / bpm):
            i = int(b * sr)
            m = min(n - i, int(0.04 * sr))
            if m > 0:
                x[i:i + m] += 0.8 * rng.standard_normal(m) * _env(m, sr, 0.001, 0.015)
        x += 0.06 * rng.standard_normal(n)

    return degrade(x, sr, rng)


# ---------------------------------------------------------------- heart
def heart_clip(label, sr, dur, rng):
    """S1/S2 sequence; 'abnormal' adds a systolic murmur (band-limited noise
    between S1 and S2), which is what the PhysioNet task keys on."""
    n = int(sr * dur)
    x = np.zeros(n)
    hr = rng.uniform(60, 100)
    period = 60.0 / hr

    def thump(idx, f, amp, dec):
        m = min(n - idx, int(0.09 * sr))
        if m <= 0:
            return
        tt = np.arange(m) / sr
        x[idx:idx + m] += amp * np.sin(2 * np.pi * f * tt) * np.exp(-tt / dec)

    for beat in np.arange(0.05, dur, period):
        j = rng.uniform(-0.01, 0.01)
        i1 = int((beat + j) * sr)
        i2 = int((beat + j + 0.32 * period) * sr)
        thump(i1, rng.uniform(40, 60), 1.0, 0.02)
        thump(i2, rng.uniform(60, 90), 0.7, 0.015)
        if label == "abnormal":
            a, b = i1 + int(0.06 * sr), i2
            m = max(0, min(n, b) - a)
            if m > 0:
                nz = rng.standard_normal(m)
                sos = sps.butter(4, [100 / (sr / 2), min(0.45, 400 / (sr / 2))],
                                 btype="bandpass", output="sos")
                x[a:a + m] += rng.uniform(0.08, 0.40) * sps.sosfilt(sos, nz) * np.hanning(m)

    return degrade(x, sr, rng, snr_db=(8.0, 25.0), lp=(0.55, 0.95))


# ---------------------------------------------------- respiratory sounds
def respiratory_clip(label, sr, dur, rng):
    """Breath-modulated noise; 'wheeze' adds a sustained narrowband tone,
    'crackle' adds short explosive transients - the ICBHI class definitions."""
    n = int(sr * dur)
    t = np.arange(n) / sr
    rate = rng.uniform(0.25, 0.4)
    breath = 0.5 * (1 + np.sin(2 * np.pi * rate * t - np.pi / 2)) ** 2

    nz = rng.standard_normal(n)
    sos = sps.butter(4, [100 / (sr / 2), min(0.45, 1200 / (sr / 2))],
                     btype="bandpass", output="sos")
    x = sps.sosfilt(sos, nz) * breath

    if label == "wheeze":
        f = rng.uniform(300, 700)
        seg = (t > dur * 0.25) & (t < dur * 0.75)
        x += rng.uniform(0.30, 0.80) * np.sin(2 * np.pi * f * t) * seg * breath

    elif label == "crackle":
        for _ in range(rng.integers(8, 20)):
            i = int(rng.uniform(0, dur * 0.95) * sr)
            m = min(n - i, int(0.006 * sr))
            if m > 0:
                tt = np.arange(m) / sr
                x[i:i + m] += rng.uniform(0.5, 1.2) * np.sin(
                    2 * np.pi * rng.uniform(600, 1500) * tt) * np.exp(-tt / 0.0015)

    return degrade(x, sr, rng, snr_db=(15.0, 30.0), lp=(0.80, 0.99))


GENERATORS = {"music": music_clip, "heart": heart_clip, "respiratory": respiratory_clip}
