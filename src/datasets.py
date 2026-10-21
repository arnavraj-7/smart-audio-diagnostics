"""M1 - Data layer. Each branch loads the real dataset if it has been
downloaded into data/, otherwise it falls back to the synthetic generator so
the pipeline stays runnable. The record format is identical either way.

Every record carries a `group` id (track, patient or recording). Splitting is
done on that id so segments cut from the same source file can never appear in
both the training and the test set - the classic way GTZAN and ICBHI scores
get inflated.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src import synth
from src.preprocess import read_audio, resample_to

AUDIO_SUFFIXES = {".wav", ".au"}


def _audio_files(root):
    # Matched on the suffix rather than by globbing each case variant: Windows
    # globs are case-insensitive, so "*.au" and "*.AU" return the same files
    # and every recording would be loaded twice.
    return sorted(p for p in Path(root).rglob("*")
                  if p.suffix.lower() in AUDIO_SUFFIXES)


def _first_existing(paths):
    for p in paths:
        if Path(p).is_dir() and _audio_files(p):
            return Path(p)
    return None


# --------------------------------------------------------------- real loaders
def load_music_real(root, cfg, limit_per_class=None):
    items = []
    for genre in cfg["classes"]:
        files = [p for p in _audio_files(root) if p.parent.name.lower() == genre]
        if not files:
            files = [p for p in _audio_files(root) if genre in p.name.lower()]
        for f in files[:limit_per_class]:
            items.append((f, genre, f.stem))
    return items


def load_heart_real(root, cfg, limit_per_class=None):
    """PhysioNet/CinC 2016: REFERENCE.csv maps record -> 1 (abnormal) / -1 (normal)."""
    ref = {}
    for csvf in Path(root).rglob("REFERENCE*.csv"):
        with open(csvf, newline="") as fh:
            for row in csv.reader(fh):
                if len(row) >= 2:
                    ref[row[0].strip()] = "abnormal" if row[1].strip() == "1" else "normal"
    if not ref:
        return []
    counts = {c: 0 for c in cfg["classes"]}
    items = []
    for f in _audio_files(root):
        lab = ref.get(f.stem)
        if lab is None:
            continue
        if limit_per_class and counts[lab] >= limit_per_class:
            continue
        counts[lab] += 1
        items.append((f, lab, f.stem))
    return items


def load_respiratory_real(root, cfg, limit_per_class=None):
    """ICBHI 2017: each .txt gives respiratory cycles as
    start end crackles(0/1) wheezes(0/1). Cycles are the labelled units and
    the patient id (the leading number in the filename) is the group."""
    counts = {c: 0 for c in cfg["classes"]}
    items = []
    for txt in sorted(Path(root).rglob("*.txt")):
        wav = txt.with_suffix(".wav")
        if not wav.exists():
            continue
        patient = txt.stem.split("_")[0]
        for line in txt.read_text().splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            start, end, cr, wh = float(parts[0]), float(parts[1]), int(parts[2]), int(parts[3])
            if cr and wh:
                continue                      # 'both' is outside this 3-class task
            lab = "crackle" if cr else "wheeze" if wh else "normal"
            if limit_per_class and counts[lab] >= limit_per_class:
                continue
            counts[lab] += 1
            items.append(((wav, start, end), lab, patient))
    return items


REAL_LOADERS = {
    "music": load_music_real,
    "heart": load_heart_real,
    "respiratory": load_respiratory_real,
}


_LAST_FILE = {}


def _read_item(item, sr_out):
    """Reads one labelled unit and resamples it to the branch rate straight
    away: ICBHI is mostly 44.1 kHz, and holding 6 000 cycles at that rate
    would take several GB for a branch that is analysed at 4 kHz."""
    if isinstance(item, tuple):                # ICBHI (wav, start, end)
        path, start, end = item
        # Consecutive cycles come from the same recording; decode it once.
        if _LAST_FILE.get("path") != path:
            x, sr = read_audio(path)
            _LAST_FILE.update(path=path, data=(resample_to(x, sr, sr_out), sr_out))
        x, sr = _LAST_FILE["data"]
        return x[int(start * sr):int(end * sr)], sr
    x, sr = read_audio(item)
    return resample_to(x, sr, sr_out), sr_out


def _segment(x, sr, cfg):
    """Cuts a long recording into up to n_segments non-overlapping clips.
    GTZAN files are 30 s, so one clip per file would throw most of it away."""
    n = int(round(sr * cfg["clip_s"]))
    k = min(cfg.get("n_segments", 1), max(1, len(x) // n))
    if k <= 1:
        return [x]
    start = (len(x) - k * n) // 2
    return [x[start + i * n:start + (i + 1) * n] for i in range(k)]


# ------------------------------------------------------------------- public
def load_domain(domain, limit_per_class=None, seed=config.RANDOM_STATE):
    """Yields (signal, sr_in, label, group) and the source actually used."""
    cfg = config.DOMAINS[domain]
    root = _first_existing(config.DATASET_PATHS[domain])
    items = REAL_LOADERS[domain](root, cfg, limit_per_class) if root else []

    if items:
        records = []
        for item, lab, group in items:
            try:
                x, sr = _read_item(item, cfg["sr"])
            except Exception:
                continue
            if len(x) < 512:
                continue
            for seg in _segment(x, sr, cfg):
                records.append((seg, sr, lab, group))
        if records:
            return records, f"real:{root}"

    rng = np.random.default_rng(seed)
    n = limit_per_class or config.SYNTH_PER_CLASS
    gen = synth.GENERATORS[domain]
    records = [(gen(c, cfg["sr"], cfg["clip_s"], rng), cfg["sr"], c, f"{c}_{i}")
               for c in cfg["classes"] for i in range(n)]
    return records, "synthetic"
