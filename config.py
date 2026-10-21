"""Global configuration: paths, DSP parameters, per-domain settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
CACHE = RESULTS / "cache"
for _p in (DATA, RESULTS, FIGURES, TABLES, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Where real datasets are expected once downloaded from Kaggle.
DATASET_PATHS = {
    "music":       [DATA / "gtzan" / "genres_original", DATA / "gtzan" / "genres", DATA / "gtzan"],
    "heart":       [DATA / "physionet2016"],
    "respiratory": [DATA / "icbhi" / "audio_and_txt_files", DATA / "icbhi"],
}

# M2/M3 parameters. One pipeline, per-domain sample rate and passband only.
DOMAINS = {
    "music": {
        "sr": 22050, "band": (30.0, 10000.0),
        "frame_ms": 25.0, "hop_ms": 10.0, "n_mels": 40, "n_mfcc": 13,
        "clip_s": 5.0, "n_segments": 6,
        "classes": ["classical", "jazz", "rock", "pop", "hiphop"],
    },
    "heart": {
        "sr": 2000, "band": (25.0, 400.0),
        "frame_ms": 50.0, "hop_ms": 25.0, "n_mels": 26, "n_mfcc": 13,
        "clip_s": 5.0, "n_segments": 3,
        "classes": ["normal", "abnormal"],
    },
    "respiratory": {
        "sr": 4000, "band": (80.0, 1800.0),
        "frame_ms": 50.0, "hop_ms": 25.0, "n_mels": 30, "n_mfcc": 13,
        "clip_s": 3.0, "n_segments": 1,
        "classes": ["normal", "wheeze", "crackle"],
    },
}

# Number of synthetic clips per class when a real dataset is absent.
SYNTH_PER_CLASS = 120
