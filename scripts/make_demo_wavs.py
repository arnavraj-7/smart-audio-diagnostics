"""Writes demo clips into data/demo_wavs/ for the GUI and for sa_verify.

    python scripts/make_demo_wavs.py              # 4 clips per class
    python scripts/make_demo_wavs.py --per-class 6

Files are named <domain>_<class>_<NN>.wav, e.g. music_classical_01.wav,
music_classical_02.wav, so the GUI dropdown offers several examples of each
class instead of one.

Music and heart demos are whole held-out recordings (capped at 30 s / 20 s)
so the GUI can classify every segment and measure tempo / heart rate.
Respiratory demos are single labelled breathing cycles, plus two complete
recordings (respiratory_recording_NN.wav) for the breathing-rate display.

Two rules, both of which matter for an honest demo:

1. Clips are cut from the REAL dataset whenever that branch has one downloaded,
   falling back to the synthetic generator only otherwise. A model trained on
   real GTZAN misclassifies a synthetic clip, because the synthetic signals are
   nowhere near the real distribution.
2. Every clip comes from the HELD-OUT TEST split - the same group-disjoint
   split used to report accuracy - so no demo shows off a recording the model
   was fitted on. This applies to the synthetic branches too.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.datasets import REAL_LOADERS, _first_existing, load_domain
from src.pipeline import build_features
from src.preprocess import fix_length, normalize, read_audio, resample_to
from src.train import split

out = config.DATA / "demo_wavs"
out.mkdir(parents=True, exist_ok=True)

# Branches whose demo files are whole recordings, capped at this many seconds.
WHOLE_FILE_S = {"music": 30.0, "heart": 20.0}


def _write(path, sr, x, group):
    wavfile.write(path, sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    print(f"{path.name:<34} {str(group):<20} {len(x) / sr:5.1f} s  held-out")


def test_groups(domain):
    """Group ids held out of training, so demo clips can be drawn from them."""
    try:
        X, y, groups, _names, _source = build_features(domain, verbose=False)
    except Exception:
        return None
    *_, gte = split(X, y, groups)
    return set(gte.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--keep", action="store_true",
                    help="do not delete existing demo clips first")
    a = ap.parse_args()

    if not a.keep:
        for f in out.glob("*.wav"):
            f.unlink()

    rng = np.random.default_rng(config.RANDOM_STATE)
    for domain, cfg in config.DOMAINS.items():
        held_out = test_groups(domain)
        root = _first_existing(config.DATASET_PATHS[domain])
        items = REAL_LOADERS[domain](root, cfg) if root else []

        if items and domain in WHOLE_FILE_S:
            # Whole held-out recordings (capped), not 5 s snippets: the GUI
            # classifies every segment and needs several beats for the rate.
            by_label = defaultdict(list)
            for path, label, group in items:
                if held_out is None or group in held_out:
                    by_label[label].append((path, group))
            for label in cfg["classes"]:
                pool = by_label.get(label, [])
                pick = rng.choice(len(pool), size=min(a.per_class, len(pool)), replace=False)
                for i, j in enumerate(sorted(pick), 1):
                    path, group = pool[j]
                    x, sr_in = read_audio(path)
                    x = resample_to(np.asarray(x, dtype=np.float64), sr_in, cfg["sr"])
                    x = normalize(x[:int(WHOLE_FILE_S[domain] * cfg["sr"])]) * 0.95
                    _write(out / f"{domain}_{label}_{i:02d}.wav", cfg["sr"], x, group)
            continue

        records, source = load_domain(domain)
        chosen = defaultdict(list)
        seen_groups = defaultdict(set)
        order = rng.permutation(len(records))
        for k in order:
            sig, sr_in, label, group = records[k]
            if len(chosen[label]) >= a.per_class:
                continue
            if held_out is not None and group not in held_out:
                continue
            if group in seen_groups[label]:
                continue          # one clip per track/patient, for variety
            seen_groups[label].add(group)
            chosen[label].append((sig, sr_in, group))

        for label in cfg["classes"]:
            for i, (sig, sr_in, group) in enumerate(chosen.get(label, []), 1):
                x = resample_to(np.asarray(sig, dtype=np.float64), sr_in, cfg["sr"])
                x = fix_length(normalize(x) * 0.95, cfg["sr"], cfg["clip_s"])
                _write(out / f"{domain}_{label}_{i:02d}.wav", cfg["sr"], x, group)

        if domain == "respiratory" and items:
            # Two complete held-out ICBHI recordings (~20 s) for the
            # breathing-rate demo; a 3 s cycle is too short to measure it.
            files = sorted({it[0][0]: it[2] for it in items
                            if held_out is None or it[2] in held_out}.items())
            for i, j in enumerate(rng.choice(len(files), size=min(2, len(files)),
                                             replace=False), 1):
                path, group = files[j]
                x, sr_in = read_audio(path)
                x = normalize(resample_to(x, sr_in, cfg["sr"])) * 0.95
                _write(out / f"respiratory_recording_{i:02d}.wav", cfg["sr"], x, group)

    n = len(list(out.glob("*.wav")))
    print(f"\n{n} demo clips in {out}")


if __name__ == "__main__":
    main()
