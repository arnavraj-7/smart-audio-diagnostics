"""Does augmenting the REAL audio help the music branch?

Not the same thing as training on synthetic signals. Here every training clip
is a real GTZAN recording; augmentation makes perturbed copies of it (mic
colouration, additive noise at a random SNR, random gain) so the classifier
sees the same music under different recording conditions.

Augmented copies keep the group id of the track they came from, so they can
never land on the opposite side of a train/test split from their source.

    python scripts/augmentation_test.py [--copies 1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.datasets import load_domain
from src.features import extract
from src.preprocess import preprocess
from src.synth import degrade
from src.train import cv_splitter, make_models

DOMAIN = "music"


def featurise(records, cfg, rng=None, label_suffix=""):
    X, y, g = [], [], []
    for i, (sig, sr_in, lab, grp) in enumerate(records):
        s = np.asarray(sig, dtype=np.float64)
        if rng is not None:
            s = degrade(s, sr_in, rng, snr_db=(10.0, 30.0), lp=(0.7, 0.99))
        x, sr = preprocess(s, sr_in, cfg)
        v, _ = extract(x, sr, cfg)
        X.append(v)
        y.append(lab)
        g.append(grp)
        if (i + 1) % 500 == 0:
            print(f"    {label_suffix}{i + 1}/{len(records)}")
    return np.vstack(X), np.asarray(y), np.asarray(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=1,
                    help="augmented copies per real clip")
    a = ap.parse_args()

    cfg = config.DOMAINS[DOMAIN]
    records, source = load_domain(DOMAIN)
    if not source.startswith("real:"):
        sys.exit("This experiment needs the real GTZAN data in data/gtzan/.")
    print(f"{len(records)} real clips from {source}\n")

    print("  baseline features (real only)")
    X0, y0, g0 = featurise(records, cfg)

    rng = np.random.default_rng(config.RANDOM_STATE)
    Xa, ya, ga = [X0], [y0], [g0]
    for c in range(a.copies):
        print(f"  augmented copy {c + 1}")
        Xc, yc, gc = featurise(records, cfg, rng=rng, label_suffix=f"aug{c + 1} ")
        Xa.append(Xc)
        ya.append(yc)
        ga.append(gc)
    X1 = np.vstack(Xa)
    y1 = np.concatenate(ya)
    g1 = np.concatenate(ga)

    print(f"\nbaseline  {X0.shape}   augmented {X1.shape}"
          f"   (groups unchanged: {len(set(g0))} -> {len(set(g1))})\n")

    print(f"{'classifier':<16}{'real only':<14}{'+ augmented':<14}delta")
    for name in make_models():
        s0 = cross_val_score(make_models()[name], X0, y0, groups=g0,
                             cv=cv_splitter(g0), scoring="accuracy",
                             n_jobs=-1).mean()
        s1 = cross_val_score(make_models()[name], X1, y1, groups=g1,
                             cv=cv_splitter(g1), scoring="accuracy",
                             n_jobs=-1).mean()
        print(f"{name:<16}{s0:<14.3f}{s1:<14.3f}{s1 - s0:+.3f}")


if __name__ == "__main__":
    main()
