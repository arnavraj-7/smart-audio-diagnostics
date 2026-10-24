"""Writes demo clips into data/demo_wavs/ for the GUI and for sa_verify.

    python scripts/make_demo_wavs.py              # 4 clips per class
    python scripts/make_demo_wavs.py --per-class 6
    python scripts/make_demo_wavs.py --per-class 10 --verified   # demo set for the viva

Files are named <domain>_<class>_<NN>.wav, e.g. music_classical_01.wav,
music_classical_02.wav, so the GUI dropdown offers several examples of each
class instead of one.

Music and heart demos are whole held-out recordings (capped at 30 s / 20 s)
so the GUI can classify every segment and measure tempo / heart rate.
Respiratory demos are single labelled breathing cycles (never zero-padded),
plus complete recordings (respiratory_<class>_recNN.wav, labelled by the
majority of their annotated cycles) for the breathing-rate display.

--verified keeps only clips the app gets right in Auto-detect mode (right
branch, right class with a clear margin, no warning, plausible heart or
breathing rate), so the demo dropdown offers only clips that work.

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
    ap.add_argument("--verified", action="store_true",
                    help="keep only clips the app classifies correctly in Auto-detect mode")
    ap.add_argument("--keep-per-class", type=int, default=3,
                    help="with --verified: clips kept per class")
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
                # Never zero-pad a short cycle: the hard stop becomes a click
                # after filtering that the model reads as a crackle. Longer
                # units are centre-cropped exactly as in training.
                x = normalize(x) * 0.95
                if len(x) > int(round(cfg["sr"] * cfg["clip_s"])):
                    x = fix_length(x, cfg["sr"], cfg["clip_s"])
                _write(out / f"{domain}_{label}_{i:02d}.wav", cfg["sr"], x, group)

        if domain == "respiratory" and items:
            # Complete held-out ICBHI recordings (~20 s) for the breathing-rate
            # demo; a 3 s cycle is too short to measure it. A recording's
            # label is the class of most of its annotated cycles, and only
            # recordings where one class clearly dominates (>= 60 %) are used,
            # named respiratory_<class>_recNN.wav.
            cycles = defaultdict(list)
            for (path, _s, _e), lab, group in items:
                if held_out is None or group in held_out:
                    cycles[(path, group)].append(lab)
            files = []
            for (path, group), labs in sorted(cycles.items()):
                top = max(set(labs), key=labs.count)
                if labs.count(top) / len(labs) >= 0.6:
                    files.append((path, group, top))
            count = defaultdict(int)
            for j in rng.permutation(len(files)):
                path, group, top = files[j]
                if count[top] >= max(2, a.per_class // 3):
                    continue
                count[top] += 1
                x, sr_in = read_audio(path)
                x = normalize(resample_to(x, sr_in, cfg["sr"])) * 0.95
                _write(out / f"respiratory_{top}_rec{count[top]:02d}.wav", cfg["sr"], x, group)

    if a.verified:
        verify_and_prune(a.keep_per_class)

    n = len(list(out.glob("*.wav")))
    print(f"\n{n} demo clips in {out}")


def verify_and_prune(keep):
    """Runs every candidate through exactly what the app does in Auto-detect
    mode (router -> branch model, all segments averaged) and keeps only clips
    that are routed to the right branch, are not flagged unknown, and get
    the right class with a clear margin. Survivors are renumbered 01..N so
    the demo dropdown only offers clips that are known to work. The accuracy
    figures in the reports come from the full test split, not from these."""
    import joblib
    from app_gui import segment_probs, true_label_from_name
    from src import router
    from src.preprocess import trim_digital_silence
    from src.rhythm import estimate_rate
    from src.preprocess import preprocess
    from src.train import for_inference

    rb = joblib.load(router.MODEL_PATH)
    models = {d: for_inference(joblib.load(config.ROOT / "models" / f"{d}.joblib")["model"])
              for d in config.DOMAINS}
    min_conf = {"music": 0.60, "heart": 0.70, "respiratory": 0.60}
    kept = defaultdict(list)
    print("\nverifying candidates (Auto-detect mode, as in the app):")
    for f in sorted(out.glob("*.wav")):
        domain = f.stem.split("_")[0]
        x, sr = read_audio(f)
        x = trim_digital_silence(np.asarray(x, dtype=np.float64))
        r = router.analyse(x, sr, rb)
        # Same rules as the app's status-bar warnings: right branch, not
        # "unknown", and the router at least 60 % sure.
        ok_route = (r["domain"] == domain and not r["unknown"]
                    and r["probs"][domain] >= 0.6)
        cfg = config.DOMAINS[domain]
        full = resample_to(x, sr, cfg["sr"])
        truth = true_label_from_name(f, domain)
        classes, _, probs = segment_probs(models[domain], full, cfg)
        j = int(np.argmax(probs))
        ok = ok_route and classes[j] == truth and probs[j] >= min_conf[domain]
        verdict = f"{classes[j]} {probs[j] * 100:.0f}%"
        key = truth
        if "_rec" in f.stem or domain == "heart":
            # The rate shown must be unambiguous too: inside the normal
            # resting range and from a clearly periodic envelope (an evenly
            # spaced S1/S2 pattern can otherwise read as double the rate).
            y, s_ = preprocess(full, cfg["sr"], dict(cfg, clip_s=len(full) / cfg["sr"]))
            rr = estimate_rate(y, s_, domain)
            rate, per = rr.get("rate"), rr.get("periodicity", 0.0)
            lo, hi = (45, 110) if domain == "heart" else (8, 30)
            min_per = 0.5 if domain == "heart" else 0.3     # breathing is less regular
            ok = ok and rate is not None and lo <= rate <= hi and per >= min_per
            verdict += f", {rate:.0f}/min p={per:.2f}" if rate else ", no rate"
            if "_rec" in f.stem:
                key = (truth, "rec")
        limit = 1 if isinstance(key, tuple) else keep     # one recording per class
        status = "keep" if ok and len(kept[(domain, key)]) < limit else "drop"
        print(f"  {f.name:<32} route={r['domain']:<12} {verdict:<22} {status}")
        if status == "keep":
            kept[(domain, key)].append(f)
        else:
            f.unlink()
    def final_name(domain, key, i):
        if isinstance(key, tuple):                      # whole lung recording
            return f"{domain}_{key[0]}_rec{i:02d}.wav"
        return f"{domain}_{key}_{i:02d}.wav"

    for (domain, key), files in kept.items():
        for i, f in enumerate(files, 1):
            f.rename(f.with_name("_" + final_name(domain, key, i)))
    for (domain, key), files in kept.items():
        for i in range(1, len(files) + 1):
            name = final_name(domain, key, i)
            (out / ("_" + name)).rename(out / name)


if __name__ == "__main__":
    main()
