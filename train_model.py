"""Trains the final classifier for each branch and saves it to models/.

The models saved here are what the GUI loads, so training and demonstration
are separate steps: train once, demo many times.

    python train_model.py                 # all branches
    python train_model.py --branch heart  # one branch
    python train_model.py --tune          # grid-search hyper-parameters
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src.pipeline import build_features
from src.train import (best_model_name, cv_splitter, evaluate_branch, for_inference,
                       make_models,
                       recording_level, screening_scores, split)

MODELS = config.ROOT / "models"
MODELS.mkdir(exist_ok=True)

GRIDS = {
    "SVM (RBF)": {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001]},
    "k-NN (k=5)": {"clf__n_neighbors": [3, 5, 7, 11]},
    "Random Forest": {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 12]},
}


def train_branch(domain, tune=False, use_cache=True):
    cfg = config.DOMAINS[domain]
    X, y, groups, names, source = build_features(
        domain, use_cache=use_cache, verbose=True)
    labels = [c for c in cfg["classes"] if c in set(y)]

    results, _ = evaluate_branch(X, y, labels, groups=groups)
    best_name = best_model_name(results)
    print(f"  classifier comparison -> best is {best_name}")

    Xtr, Xte, ytr, yte, gtr, gte = split(X, y, groups)
    model = make_models()[best_name]
    params = {}

    if tune:
        gs = GridSearchCV(model, GRIDS[best_name], cv=cv_splitter(groups),
                          scoring="balanced_accuracy", n_jobs=-1)
        gs.fit(Xtr, ytr, groups=gtr)
        model, params = gs.best_estimator_, gs.best_params_
        print(f"  tuned: {params}  (cv={gs.best_score_:.3f})")
    else:
        model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    bacc = balanced_accuracy_score(yte, pred)
    print(f"  held-out test accuracy = {acc:.3f}  balanced = {bacc:.3f}")
    print(classification_report(yte, pred, labels=labels, zero_division=0))
    rec = recording_level(model, Xte, yte, gte, labels)
    if rec:
        print(f"  per-recording ({rec['n_recordings']} held-out recordings, segments "
              f"averaged): accuracy = {rec['accuracy']:.3f}  "
              f"balanced = {rec['balanced_accuracy']:.3f}")

    # Refit on everything for the model that ships to the GUI.
    final = make_models()[best_name]
    if params:
        final.set_params(**params)
    final.fit(X, y)

    meta = {
        "domain": domain, "classifier": best_name, "params": params,
        "source": source, "n_samples": int(X.shape[0]),
        "n_groups": int(len(set(groups))),
        "n_features": int(X.shape[1]), "classes": labels,
        "holdout_accuracy": float(acc),
        "holdout_balanced_accuracy": float(bacc),
        "screening": screening_scores(yte, pred) if "normal" in labels else None,
        "recording_level": rec,
        "comparison": {k: {"cv_mean": v["cv_mean"], "test_accuracy": v["test_accuracy"],
                           "balanced_accuracy": v["balanced_accuracy"],
                           "macro_f1": v["macro_f1"]} for k, v in results.items()},
        "confusion_matrix": confusion_matrix(yte, pred, labels=labels).tolist(),
        "feature_names": names,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump({"model": for_inference(final), "meta": meta}, MODELS / f"{domain}.joblib")
    (MODELS / f"{domain}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  saved {MODELS / f'{domain}.joblib'}\n")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", choices=list(config.DOMAINS), action="append")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    metas = []
    for d in (a.branch or list(config.DOMAINS)):
        print(f"=== training {d} ===")
        metas.append(train_branch(d, a.tune, use_cache=not a.no_cache))

    print("Summary")
    print(f"{'branch':<14}{'classifier':<16}{'acc':<8}{'n':<7}source")
    for m in metas:
        print(f"{m['domain']:<14}{m['classifier']:<16}"
              f"{m['holdout_accuracy']:<8.3f}{m['n_samples']:<7}{m['source']}")


if __name__ == "__main__":
    main()
