"""Over-/under-fitting diagnostics per branch.

Reports training accuracy, cross-validation accuracy and held-out test accuracy
side by side, and plots a learning curve. Overfitting shows up as CV and test
falling well below training for a classifier that does not interpolate by
design; underfitting shows up as a CV score close to chance.

    python analyze_fit.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, learning_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src.pipeline import build_features
from src.train import best_model_name, cv_splitter, evaluate_branch, make_models, split

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})


# Random Forest grows trees until the leaves are pure and an RBF SVM at high C
# interpolates too, so training accuracy is 1.000 by construction for both.
# A train-minus-CV gap therefore says nothing on its own; what matters is
# whether CV sits well above chance and whether the held-out test agrees with
# it. Both are reported, and the verdict is based on the second pair.
INTERPOLATING = ("Random Forest", "SVM (RBF)")


def verdict(train, cv, test, n_classes, classifier):
    chance = 1.0 / n_classes
    gap = train - cv
    drift = abs(cv - test)

    if cv < chance + 0.10:
        return "UNDERFITTING - barely above chance"
    if drift > 0.10:
        return (f"unstable - CV and test differ by {drift:.3f}; "
                "too few groups to judge")
    if classifier in INTERPOLATING and train > 0.999:
        note = "train=1.000 is inherent to this classifier, not evidence of overfitting"
        if cv > chance + 0.45:
            return f"generalises well - CV {cv:.3f} vs test {test:.3f}; {note}"
        return f"generalises, with headroom - CV {cv:.3f} vs test {test:.3f}; {note}"
    if gap > 0.20:
        return f"OVERFITTING - train exceeds CV by {gap:.3f}"
    if gap > 0.10:
        return f"mild overfitting - gap {gap:.3f}"
    return f"well fitted - gap {gap:.3f}"


def analyse(domain):
    cfg = config.DOMAINS[domain]
    X, y, groups, names, source = build_features(domain, verbose=False)
    labels = [c for c in cfg["classes"] if c in set(y)]

    # Analyse the model that actually ships: same classifier and the
    # hyper-parameters chosen by `train_model.py --tune`, if it was run.
    meta_p = config.ROOT / "models" / f"{domain}_meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        best, params = meta["classifier"], meta.get("params") or {}
    else:
        results, _ = evaluate_branch(X, y, labels, groups=groups)
        best, params = best_model_name(results), {}

    def fresh():
        return make_models()[best].set_params(**params)

    model = fresh()

    Xtr, Xte, ytr, yte, gtr, _ = split(X, y, groups)
    model.fit(Xtr, ytr)
    train_acc = accuracy_score(ytr, model.predict(Xtr))
    test_acc = accuracy_score(yte, model.predict(Xte))
    cv = cross_val_score(fresh(), Xtr, ytr, groups=gtr,
                         cv=cv_splitter(groups), scoring="accuracy", n_jobs=-1)

    row = {
        "domain": domain, "source": source, "classifier": best, "params": params,
        "n_samples": int(X.shape[0]), "n_groups": int(len(set(groups))),
        "n_classes": len(labels), "chance": 1.0 / len(labels),
        "train_accuracy": float(train_acc),
        "cv_accuracy": float(cv.mean()), "cv_std": float(cv.std()),
        "test_accuracy": float(test_acc),
        "train_minus_cv": float(train_acc - cv.mean()),
        "cv_minus_test": float(cv.mean() - test_acc),
        "verdict": verdict(train_acc, cv.mean(), test_acc, len(labels), best),
    }

    sizes, tr_sc, va_sc = learning_curve(
        fresh(), X, y, groups=groups, cv=cv_splitter(groups),
        train_sizes=np.linspace(0.15, 1.0, 6), scoring="accuracy", n_jobs=-1,
        shuffle=True, random_state=config.RANDOM_STATE)
    row["learning_curve"] = {
        "sizes": sizes.tolist(),
        "train": tr_sc.mean(axis=1).tolist(),
        "validation": va_sc.mean(axis=1).tolist(),
    }
    return row, (sizes, tr_sc, va_sc)


def main():
    rows, curves = [], {}
    for d in config.DOMAINS:
        print(f"analysing {d} ...")
        row, curve = analyse(d)
        rows.append(row)
        curves[d] = curve

    fig, axes = plt.subplots(1, len(curves), figsize=(4.6 * len(curves), 3.8))
    axes = np.atleast_1d(axes)
    for ax, (d, (sizes, tr, va)) in zip(axes, curves.items()):
        ax.plot(sizes, tr.mean(axis=1), "o-", label="training", color="#c05621")
        ax.fill_between(sizes, tr.mean(1) - tr.std(1), tr.mean(1) + tr.std(1),
                        alpha=0.15, color="#c05621")
        ax.plot(sizes, va.mean(axis=1), "o-", label="cross-validation",
                color="#2b6cb0")
        ax.fill_between(sizes, va.mean(1) - va.std(1), va.mean(1) + va.std(1),
                        alpha=0.15, color="#2b6cb0")
        n_cls = len(config.DOMAINS[d]["classes"])
        ax.axhline(1 / n_cls, ls="--", lw=0.8, color="#888",
                   label=f"chance ({1 / n_cls:.2f})")
        ax.set(title=d, xlabel="Training samples", ylabel="Accuracy", ylim=(0, 1.05))
        ax.legend(fontsize=7)
    fig.suptitle("Learning curves - training vs cross-validation",
                 fontweight="bold")
    fig.tight_layout()
    out = config.FIGURES / "learning_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'branch':<13}{'chance':<8}{'train':<8}{'CV':<8}{'test':<8}"
          f"{'CV-test':<9}verdict")
    for r in rows:
        print(f"{r['domain']:<13}{r['chance']:<8.2f}{r['train_accuracy']:<8.3f}"
              f"{r['cv_accuracy']:<8.3f}{r['test_accuracy']:<8.3f}"
              f"{r['cv_minus_test']:<+9.3f}{r['verdict']}")

    (config.TABLES / "fit_analysis.json").write_text(json.dumps(rows, indent=2))

    L = ["# Over-/under-fitting analysis\n",
         "Generated by `analyze_fit.py`. Training and test sets are split on the",
         "group id (track / patient / recording), so no segment of a source file",
         "appears in both.\n",
         "| Branch | Source | Classifier | Samples | Groups | Chance | Train | CV | Test | CV−Test | Verdict |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['domain']} | `{r['source']}` | {r['classifier']} | "
                 f"{r['n_samples']} | {r['n_groups']} | {r['chance']:.2f} | "
                 f"{r['train_accuracy']:.3f} | {r['cv_accuracy']:.3f} ± {r['cv_std']:.3f} | "
                 f"{r['test_accuracy']:.3f} | {r['cv_minus_test']:+.3f} | {r['verdict']} |")
    L.append("\nLearning curves: `results/figures/learning_curves.png`\n")
    (config.RESULTS / "FIT_ANALYSIS.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nSaved {config.RESULTS / 'FIT_ANALYSIS.md'} and {out}")


if __name__ == "__main__":
    main()
