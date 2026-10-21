"""M4/M5 - classifier training, comparison and evaluation.
SVM, k-NN and Random Forest are trained independently per branch."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config

RS = config.RANDOM_STATE


def make_models():
    # class_weight="balanced" re-weights each class by 1/frequency. It changes
    # nothing on balanced GTZAN, and stops the classifier on PhysioNet (~80 %
    # normal) and ICBHI (~55 % normal) from buying accuracy by always
    # answering "normal".
    return {
        "SVM (RBF)": Pipeline([("sc", StandardScaler()),
                               ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                                           class_weight="balanced", random_state=RS))]),
        "k-NN (k=5)": Pipeline([("sc", StandardScaler()),
                                ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "Random Forest": Pipeline([("sc", StandardScaler()),
                                   ("clf", RandomForestClassifier(
                                       n_estimators=300, class_weight="balanced",
                                       random_state=RS, n_jobs=-1))]),
    }


def split(X, y, groups=None, test_size=0.25):
    """Group-aware hold-out: no group (track / patient / recording) appears in
    both halves, so segments of the same source cannot leak across the split."""
    if groups is None:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=RS)
        return Xtr, Xte, ytr, yte, None, None
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RS)
    tr, te = next(gss.split(X, y, groups))
    return X[tr], X[te], y[tr], y[te], groups[tr], groups[te]


def cv_splitter(groups, cv_folds=5):
    if groups is None:
        return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RS)
    return StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=RS)


def evaluate_branch(X, y, labels, groups=None, test_size=0.25, cv_folds=5):
    Xtr, Xte, ytr, yte, gtr, _ = split(X, y, groups, test_size)
    cv = cv_splitter(groups, cv_folds)

    out = {}
    for name, model in make_models().items():
        cvs = cross_val_score(model, Xtr, ytr, groups=gtr, cv=cv,
                              scoring="accuracy", n_jobs=-1)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rep = classification_report(yte, pred, labels=labels,
                                    output_dict=True, zero_division=0)
        res = {
            "cv_mean": float(cvs.mean()), "cv_std": float(cvs.std()),
            "test_accuracy": float(accuracy_score(yte, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
            "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(yte, pred, labels=labels).tolist(),
            "report": rep,
            "model": model,
        }
        if len(labels) == 2:
            cm = np.array(res["confusion_matrix"])   # rows follow `labels` order
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            res["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
            res["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
        elif "normal" in labels:
            res.update(screening_scores(yte, pred))
        out[name] = res
    return out, (Xtr, Xte, ytr, yte)


def screening_scores(y_true, y_pred, normal="normal"):
    """ICBHI-challenge convention for a multi-class screening task:
    sensitivity = abnormal cycles given the right abnormal label,
    specificity = normal cycles called normal, score = their mean."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ab = y_true != normal
    se = float(np.mean(y_pred[ab] == y_true[ab])) if ab.any() else 0.0
    sp = float(np.mean(y_pred[~ab] == normal)) if (~ab).any() else 0.0
    return {"sensitivity": se, "specificity": sp, "icbhi_score": (se + sp) / 2}


def for_inference(model):
    """Random Forests train with n_jobs=-1, but predicting one file needs no
    parallelism, and joblib worker threads inside the Tk GUI let Python free
    Tk objects on the wrong thread, which aborts the app ("Tcl_AsyncDelete").
    Every shipped model is switched to single-threaded prediction."""
    for step in getattr(model, "named_steps", {}).values():
        if hasattr(step, "n_jobs"):
            step.n_jobs = 1
    return model


def proba(model, X):
    """Class probabilities from any of the three classifiers (an SVC without
    probability=True only has a decision function; softmax it)."""
    try:
        return np.asarray(model.predict_proba(X))
    except AttributeError:
        d = np.asarray(model.decision_function(X))
        d = np.column_stack([-d, d]) if d.ndim == 1 else d
        e = np.exp(d - d.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


def recording_level(model, X, y, groups, labels):
    """One decision per recording - the mean class probability over all of
    its segments, which is exactly what the GUI does. Only meaningful where a
    group carries one label (a track, a heart recording), not for ICBHI where
    the group is a patient with cycles of several classes."""
    classes = list(model.classes_)
    P = proba(model, X)
    yt, yp = [], []
    for g in np.unique(groups):
        m = groups == g
        if len(set(y[m])) != 1:
            return None
        yt.append(y[m][0])
        yp.append(classes[int(np.argmax(P[m].mean(axis=0)))])
    yt, yp = np.asarray(yt), np.asarray(yp)
    out = {"n_recordings": int(len(yt)),
           "accuracy": float(accuracy_score(yt, yp)),
           "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
           "confusion_matrix": confusion_matrix(yt, yp, labels=labels).tolist()}
    if "normal" in labels:
        out.update(screening_scores(yt, yp))
    return out


def best_model_name(results):
    # Balanced accuracy, not raw accuracy: on imbalanced medical data a model
    # that always says "normal" would otherwise win.
    return max(results, key=lambda k: results[k]["balanced_accuracy"])
