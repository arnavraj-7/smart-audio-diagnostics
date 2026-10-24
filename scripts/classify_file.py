"""Classify any audio file (or a whole folder) from the command line.

    python scripts/classify_file.py song.mp3
    python scripts/classify_file.py ~/Downloads --domain music
    python scripts/classify_file.py data/demo_wavs --domain auto   # router picks the branch

Averages over every clip-length segment of the file, which is far more stable
than judging a song from one 5-second window.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.features import extract
from src.preprocess import preprocess, read_audio, resample_to, trim_digital_silence
from src import router
from src.train import for_inference

EXTS = {".wav", ".au", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


def scores(path, domain, model):
    cfg = config.DOMAINS[domain]
    x, sr = read_audio(path)
    x = trim_digital_silence(np.asarray(x, dtype=np.float64))
    x = resample_to(np.asarray(x, dtype=np.float64), sr, cfg["sr"])
    n = int(round(cfg["sr"] * cfg["clip_s"]))
    k = max(1, len(x) // n)
    acc, classes = None, None
    for i in range(k):
        v, _ = extract(*preprocess(x[i * n:(i + 1) * n], cfg["sr"], cfg), cfg)
        classes, p = _proba(model, v[None, :])
        acc = p if acc is None else acc + p
    return classes, acc / k, k


def _proba(model, x):
    classes = list(model.classes_)
    if hasattr(model, "predict_proba"):
        try:
            return classes, np.asarray(model.predict_proba(x))[0]
        except (AttributeError, NotImplementedError):
            pass
    d = np.asarray(model.decision_function(x))
    d = np.array([[-d[0], d[0]]]) if d.ndim == 1 else d
    e = np.exp(d[0] - d[0].max())
    return classes, e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="audio file or folder")
    ap.add_argument("--domain", default="auto", choices=["auto"] + list(config.DOMAINS))
    ap.add_argument("--top", type=int, default=3)
    a = ap.parse_args()

    models = {d: for_inference(joblib.load(config.ROOT / "models" / f"{d}.joblib")["model"])
              for d in config.DOMAINS}
    rb = joblib.load(router.MODEL_PATH) if a.domain == "auto" else None
    t = Path(a.target)
    files = sorted(p for p in t.iterdir() if p.suffix.lower() in EXTS) \
        if t.is_dir() else [t]
    if not files:
        sys.exit(f"No audio files in {t}")

    for f in files:
        try:
            d, tag = a.domain, ""
            if rb is not None:
                r = router.analyse(*read_audio(f), rb)
                d = r["domain"]
                tag = f"{'UNKNOWN~' if r['unknown'] else ''}{d}"
            classes, probs, k = scores(f, d, models[d])
        except Exception as e:
            print(f"{f.name[:44]:<46} FAILED: {str(e)[:50]}")
            continue
        order = np.argsort(probs)[::-1][:a.top]
        top = "  ".join(f"{classes[i]} {probs[i] * 100:.0f}%" for i in order)
        print(f"{f.name[:44]:<46} {tag:<20}[{k:>2} seg]  {top}")


if __name__ == "__main__":
    main()
