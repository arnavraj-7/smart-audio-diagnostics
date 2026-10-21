"""Unit checks on the DSP front end. Run: python tests/test_dsp.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.features import (delta, extract, mel_filterbank, mfcc, spectral_bandwidth,
                          spectral_centroid, stft_power, zero_crossing_rate)
from src.preprocess import bandpass, frame_signal, normalize, resample_to

CFG = config.DOMAINS["music"]
SR = CFG["sr"]
PASS = []


def check(name, cond, detail=""):
    PASS.append(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


def tone(f, sr=SR, dur=2.0):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * f * t)


# --- M2 -----------------------------------------------------------------
x = tone(1000)
check("resample keeps duration", len(resample_to(x, SR, 8000)) == int(len(x) * 8000 / SR),
      f"{len(resample_to(x, SR, 8000))}")

mix = tone(200) + tone(6000)
y = bandpass(mix, SR, (100.0, 1000.0))
X = np.abs(np.fft.rfft(y)); f = np.fft.rfftfreq(len(y), 1 / SR)
r = X[np.argmin(abs(f - 6000))] / X[np.argmin(abs(f - 200))]
check("band-pass rejects out-of-band tone", r < 1e-3, f"6k/200Hz ratio={r:.2e}")

check("peak normalisation", abs(np.abs(normalize(3.7 * x)).max() - 1) < 1e-9)

fr = frame_signal(np.arange(1000.0), 100, 50, window=None)
check("framing shape", fr.shape == (19, 100), str(fr.shape))
check("framing hop", fr[1, 0] == 50.0)

# --- M3 -----------------------------------------------------------------
_, power, freqs, _, _ = stft_power(tone(1000), SR, CFG["frame_ms"], CFG["hop_ms"])
c = spectral_centroid(power, freqs).mean()
check("centroid locks onto 1 kHz tone", abs(c - 1000) < 60, f"{c:.1f} Hz")

c2 = spectral_centroid(*[*stft_power(tone(4000), SR, CFG["frame_ms"], CFG["hop_ms"])[1:3]]).mean()
check("centroid tracks frequency upward", c2 > c + 2500, f"{c2:.1f} Hz")

cen = spectral_centroid(power, freqs)
bw_tone = spectral_bandwidth(power, freqs, cen).mean()
rng = np.random.default_rng(0)
_, pn, fn, _, _ = stft_power(rng.standard_normal(SR * 2), SR, CFG["frame_ms"], CFG["hop_ms"])
bw_noise = spectral_bandwidth(pn, fn, spectral_centroid(pn, fn)).mean()
check("bandwidth: noise wider than tone", bw_noise > 3 * bw_tone,
      f"tone={bw_tone:.0f} noise={bw_noise:.0f} Hz")

fb = mel_filterbank(SR, 1024, 40, 0, SR / 2)
check("mel bank shape", fb.shape == (40, 513), str(fb.shape))
check("mel filters non-negative", (fb >= 0).all())
peaks = np.argmax(fb, axis=1)
check("mel filter centres increase", np.all(np.diff(peaks) > 0))
widths = (fb > 0).sum(axis=1)
check("mel filters widen with frequency", widths[-1] > widths[0], f"{widths[0]} -> {widths[-1]}")

zc = zero_crossing_rate(frame_signal(tone(100), 2205, 2205, window=None)).mean()
check("ZCR matches 2f/fs for a tone", abs(zc - 2 * 100 / SR) < 2e-3,
      f"{zc:.5f} vs {2 * 100 / SR:.5f}")

m_loud = mfcc(tone(440), SR, CFG)[:, 0].mean()
m_quiet = mfcc(0.01 * tone(440), SR, CFG)[:, 0].mean()
check("MFCC-1 tracks log energy", m_loud > m_quiet, f"{m_loud:.2f} > {m_quiet:.2f}")

m = mfcc(tone(440), SR, CFG)
check("MFCC shape is (frames, 13)", m.shape[1] == 13, str(m.shape))

d = delta(np.tile(np.arange(10.0)[:, None], (1, 3)))
check("delta of a ramp is constant", np.allclose(d[3:-3], d[4], atol=1e-9), f"{d[5, 0]:.3f}")

v, names = extract(tone(440), SR, CFG)
check("feature vector length 53", len(v) == 53 and len(names) == 53, str(len(v)))
check("feature vector is finite", np.isfinite(v).all())
groups = {n.split("::")[0] for n in names}
check("three feature groups present", groups == {"mfcc", "spectral", "temporal"}, str(groups))

print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
sys.exit(0 if all(PASS) else 1)
