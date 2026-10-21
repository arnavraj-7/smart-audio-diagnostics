"""O5 - feature-ablation study: drop one feature group at a time and measure
the accuracy lost, per branch."""
import numpy as np
from sklearn.model_selection import cross_val_score

import config
from src.features import GROUPS, group_mask
from src.train import cv_splitter, make_models


def run_ablation(X, y, names, model_name, groups=None, cv_folds=5):
    model = make_models()[model_name]
    cv = cv_splitter(groups, cv_folds)

    def score(mask):
        if mask.sum() == 0:
            return 0.0
        return float(cross_val_score(model, X[:, mask], y, groups=groups, cv=cv,
                                     scoring="accuracy", n_jobs=-1).mean())

    full = score(np.ones(len(names), dtype=bool))
    rows = [{"configuration": "All features", "n_features": int(len(names)),
             "cv_accuracy": full, "delta": 0.0}]

    for g in GROUPS:
        keep = [k for k in GROUPS if k != g]
        m = group_mask(names, keep)
        s = score(m)
        rows.append({"configuration": f"without {g}", "n_features": int(m.sum()),
                     "cv_accuracy": s, "delta": s - full})

    for g in GROUPS:
        m = group_mask(names, [g])
        s = score(m)
        rows.append({"configuration": f"{g} only", "n_features": int(m.sum()),
                     "cv_accuracy": s, "delta": s - full})
    return rows
