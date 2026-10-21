"""Glue: M1 -> M2 -> M3. Builds and caches the feature matrix for one branch."""
import time

import numpy as np

import config
from src.datasets import load_domain
from src.features import extract
from src.preprocess import preprocess


CACHE_KEYS = ("X", "y", "groups", "names", "source")


def build_features(domain, limit_per_class=None, use_cache=True, verbose=True):
    cfg = config.DOMAINS[domain]
    cache = config.CACHE / f"{domain}.npz"
    if use_cache and cache.exists():
        d = np.load(cache, allow_pickle=True)
        if all(k in d for k in CACHE_KEYS):
            return d["X"], d["y"], d["groups"], list(d["names"]), str(d["source"])
        if verbose:
            print(f"[{domain}] cache predates the current format, rebuilding")

    records, source = load_domain(domain, limit_per_class)
    if verbose:
        print(f"[{domain}] {len(records)} clips  source={source}")

    X, y, groups, names, t0 = [], [], [], None, time.time()
    for i, (sig, sr_in, lab, grp) in enumerate(records):
        x, sr = preprocess(sig, sr_in, cfg)
        v, names = extract(x, sr, cfg)
        X.append(v)
        y.append(lab)
        groups.append(grp)
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(records)}")
    X = np.vstack(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if verbose:
        print(f"[{domain}] feature matrix {X.shape} from "
              f"{len(set(groups))} groups in {time.time() - t0:.1f}s")

    np.savez_compressed(cache, X=X, y=y, groups=groups,
                        names=np.array(names), source=source)
    return X, y, groups, names, source
