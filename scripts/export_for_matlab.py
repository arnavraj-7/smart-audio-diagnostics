"""Exports the trained Python model to plain CSV so the MATLAB app can use it,
and dumps a reference feature vector for the cross-toolchain check.

    python scripts/export_for_matlab.py
    python scripts/export_for_matlab.py --verify path/to/file.wav --domain heart
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.features import extract
from src.pipeline import build_features
from src.preprocess import preprocess, read_wav

OUT = config.ROOT / "model_export"
OUT.mkdir(exist_ok=True)


def export_all():
    for domain in config.DOMAINS:
        X, y, _groups, names, source = build_features(domain, verbose=False)
        sc = StandardScaler().fit(X)
        np.savetxt(OUT / f"{domain}_scaler_mean.csv", sc.mean_[None, :], delimiter=",")
        np.savetxt(OUT / f"{domain}_scaler_scale.csv", sc.scale_[None, :], delimiter=",")
        np.savetxt(OUT / f"{domain}_train_X.csv", sc.transform(X), delimiter=",", fmt="%.6f")
        with open(OUT / f"{domain}_train_y.csv", "w") as fh:
            fh.write("label\n")
            fh.writelines(f"{v}\n" for v in y)
        (OUT / f"{domain}_feature_names.csv").write_text("\n".join(names))
        print(f"{domain:<12} exported {X.shape[0]} x {X.shape[1]}  (source={source})")


def export_verify(wav, domain, do_denoise=False):
    cfg = config.DOMAINS[domain]
    x, sr_in = read_wav(wav)
    y, sr = preprocess(x, sr_in, cfg, do_denoise=do_denoise)
    v, names = extract(y, sr, cfg)
    p = config.TABLES / "verify_python.csv"
    np.savetxt(p, v[:, None], delimiter=",", fmt="%.10f")
    tag = "_dn" if do_denoise else ""
    np.savetxt(config.TABLES / f"verify_python_{domain}{tag}.csv", v[:, None],
               delimiter=",", fmt="%.10f")
    print(f"Wrote {p} ({len(v)} features from {wav})")
    print("Now run in MATLAB:  sa_verify('%s', '%s')" % (str(wav).replace("\\", "/"), domain))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify")
    ap.add_argument("--domain", default="heart", choices=list(config.DOMAINS))
    ap.add_argument("--denoise", action="store_true",
                    help="include spectral subtraction in the verified chain")
    a = ap.parse_args()
    if a.verify:
        export_verify(a.verify, a.domain, a.denoise)
    else:
        export_all()
