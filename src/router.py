"""Signal-type router: music vs heart sound vs lung sound.

The user should not have to know which branch a recording belongs to. The
router is a fourth classifier that sits *in front of* the two branches and
decides where a file goes. It reuses the same M2/M3 front end, run at one
common setting so the three kinds of signal are described on the same scale:

    any file -> resample to 4 kHz -> band-pass 20-1900 Hz -> 53 features
             -> Random Forest -> {music, heart, respiratory}

4 kHz is enough: the router only has to separate broadband, harmonic music
from heart sounds (energy below ~400 Hz) and breath noise (100-1800 Hz). It
never sees the branch labels, so genre and clinical labels stay unmixed.

    python -m src.router          # train and save models/router.joblib
"""
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.datasets import load_domain
from src.features import extract
from src.preprocess import preprocess, read_audio, resample_to
from src.train import for_inference

ROUTER_CFG = {"sr": 4000, "band": (20.0, 1900.0), "frame_ms": 50.0,
              "hop_ms": 25.0, "n_mels": 30, "n_mfcc": 13, "clip_s": 3.0}
DOMAINS = ("music", "heart", "respiratory")
# A fourth, "none of these" class: a classifier must always answer one of
# its classes, so without it a door bell or a voice would be confidently
# called a lung sound. Trained on ESC-50 (Piczak 2015): 2 000 five-second
# environmental clips in 50 categories - speech-like vocal sounds, animals,
# machines, alarms, clapping, typing, rain...
OTHER = "other"
CLASSES = DOMAINS + (OTHER,)
ESC50 = config.DATA / "esc50" / "ESC-50-master"
ESC_PER_CATEGORY = 16
# Clips drawn at random per class of each branch, so every recording
# device / sub-database in the branch is represented, not just the first.
PER_CLASS = {"music": 140, "heart": 300, "respiratory": 220}
MODEL_PATH = config.ROOT / "models" / "router.joblib"


def _clips(x, sr):
    """Cuts a recording into 3 s router clips (at most 4, from the middle)."""
    x = resample_to(np.asarray(x, dtype=np.float64), sr, ROUTER_CFG["sr"])
    n = int(ROUTER_CFG["sr"] * ROUTER_CFG["clip_s"])
    k = max(1, min(4, len(x) // n))
    start = max(0, (len(x) - k * n) // 2)
    return [x[start + i * n:start + (i + 1) * n] for i in range(k)]


def features(x, sr):
    out = []
    for clip in _clips(x, sr):
        y, s = preprocess(clip, ROUTER_CFG["sr"], ROUTER_CFG)
        out.append(extract(y, s, ROUTER_CFG)[0])
    return np.vstack(out)


def build_dataset(verbose=True):
    X, y, groups, sources = [], [], [], {}
    rng = np.random.default_rng(config.RANDOM_STATE)
    for d in DOMAINS:
        records, source = load_domain(d)
        sources[d] = source
        labels = np.array([r[2] for r in records])
        pick = np.concatenate([
            rng.permutation(np.where(labels == c)[0])[:PER_CLASS[d]]
            for c in np.unique(labels)])
        for k in pick:
            sig, sr, _label, grp = records[k]
            for v in features(sig, sr):
                X.append(v)
                y.append(d)
                groups.append(f"{d}:{grp}")
        if verbose:
            print(f"  router/{d:<12} {sum(1 for t in y if t == d):>5} clips  {source}")

    meta = ESC50 / "meta" / "esc50.csv"
    if meta.exists():
        rows = list(csv.DictReader(open(meta, newline="")))
        by_cat = {}
        for r in rows:
            by_cat.setdefault(r["category"], []).append(r)
        for cat, rs in sorted(by_cat.items()):
            for j in rng.permutation(len(rs))[:ESC_PER_CATEGORY]:
                r = rs[j]
                sig, sr = read_audio(ESC50 / "audio" / r["filename"])
                if np.max(np.abs(sig)) < 1e-4:
                    continue
                for v in features(sig, sr):
                    X.append(v)
                    y.append(OTHER)
                    groups.append(f"esc:{r['src_file']}")   # same source -> same side
        sources[OTHER] = f"real:{ESC50}"
        if verbose:
            print(f"  router/{OTHER:<12} {sum(1 for t in y if t == OTHER):>5} clips  ESC-50")
    return np.vstack(X), np.asarray(y), np.asarray(groups), sources


def _nn_distance(ref, Q):
    """Mean distance from each row of Q to its 3 nearest rows of ref."""
    nn = NearestNeighbors(n_neighbors=3).fit(ref)
    return nn.kneighbors(Q)[0].mean(axis=1)


def train(verbose=True, use_cache=True):
    cache = config.CACHE / "router.npz"
    if use_cache and cache.exists():
        d = np.load(cache, allow_pickle=True)
        X, y, groups, sources = d["X"], d["y"], d["groups"], d["sources"].item()
    else:
        X, y, groups, sources = build_dataset(verbose)
        np.savez_compressed(cache, X=X, y=y, groups=groups,
                            sources=np.array(sources, dtype=object))
    model = Pipeline([("sc", StandardScaler()),
                      ("clf", RandomForestClassifier(
                          n_estimators=300, class_weight="balanced",
                          random_state=config.RANDOM_STATE, n_jobs=-1))])
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=config.RANDOM_STATE)
                  .split(X, y, groups))
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    acc = accuracy_score(y[te], pred)
    labels = [c for c in CLASSES if c in set(y)]
    cm = confusion_matrix(y[te], pred, labels=labels).tolist()

    # Novelty threshold. A classifier always answers one of its classes, even
    # for a door bell. So we also measure how far a clip is from the nearest
    # training clips, and calibrate the cut-off on held-out recordings (never
    # their own neighbours): 99 % of genuine music / heart / lung clips fall
    # below it, anything beyond is reported as "unlike the training data".
    sc = model.named_steps["sc"]
    known_tr = tr[y[tr] != OTHER]
    known_te = te[y[te] != OTHER]
    d_te = _nn_distance(sc.transform(X[known_tr]), sc.transform(X[known_te]))
    threshold = float(np.percentile(d_te, 99))
    if verbose:
        print(f"  router held-out accuracy = {acc:.3f}  (n_test={len(te)})")
        print(f"  confusion (rows=true {labels}): {cm}")
        print(f"  novelty threshold = {threshold:.2f} (99th pct of held-out NN distance)")

    model.fit(X, y)
    meta = {"classes": labels, "holdout_accuracy": float(acc),
            "confusion_matrix": cm, "n_samples": int(len(y)),
            "novelty_threshold": threshold,
            "sources": sources, "cfg": {k: list(v) if isinstance(v, tuple) else v
                                        for k, v in ROUTER_CFG.items()}}
    ref = model.named_steps["sc"].transform(X[y != OTHER]).astype(np.float32)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": for_inference(model), "meta": meta, "reference": ref},
                MODEL_PATH)
    (MODEL_PATH.parent / "router_meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"  saved {MODEL_PATH}")
    return meta


def analyse(x, sr, bundle=None):
    """Router decision plus a novelty score: distance to the nearest training
    clips divided by the calibrated threshold (> 1 means 'unknown sound')."""
    bundle = bundle or joblib.load(MODEL_PATH)
    model = for_inference(bundle["model"])
    F = features(x, sr)
    p = model.predict_proba(F).mean(axis=0)
    probs = {c: float(v) for c, v in zip(model.classes_, p)}
    novelty = None
    if "reference" in bundle:
        d = _nn_distance(bundle["reference"], model.named_steps["sc"].transform(F))
        novelty = float(np.median(d) / bundle["meta"]["novelty_threshold"])
    top = max(probs, key=probs.get)
    # The branch to analyse with is always one of the three; "unknown" is a
    # separate flag so the app can still show the signal while warning.
    domain = max(DOMAINS, key=lambda c: probs.get(c, 0.0))
    unknown = top == OTHER or (novelty is not None and novelty > 1.0)
    return {"domain": domain, "probs": probs, "novelty": novelty,
            "unknown": bool(unknown), "top": top}


def detect(x, sr, bundle=None):
    """Returns (domain, {domain: probability}) averaged over the file's clips."""
    r = analyse(x, sr, bundle)
    return r["domain"], r["probs"]


if __name__ == "__main__":
    train(use_cache="--no-cache" not in sys.argv)
