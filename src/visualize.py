"""Figure generation for the report (waveform, FFT, spectrogram, mel filter
bank, MFCC map, feature separation, confusion matrices, classifier comparison)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from src.features import mel_filterbank, mfcc, stft_power

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})


def signal_overview(x, sr, cfg, title, path):
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
    t = np.arange(len(x)) / sr
    ax[0, 0].plot(t, x, lw=0.4)
    ax[0, 0].set(title="Time-domain waveform (after M2)", xlabel="Time (s)",
                 ylabel="Amplitude")

    X = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    ax[0, 1].semilogy(f, X + 1e-9, lw=0.5)
    ax[0, 1].set(title="FFT magnitude spectrum", xlabel="Frequency (Hz)",
                 ylabel="|X(f)|", xlim=(0, min(cfg["band"][1] * 1.2, sr / 2)))

    spec, power, freqs, _, _ = stft_power(x, sr, cfg["frame_ms"], cfg["hop_ms"])
    ax[1, 0].imshow(10 * np.log10(power.T + 1e-12), origin="lower", aspect="auto",
                    extent=[0, len(x) / sr, 0, sr / 2], cmap="magma")
    ax[1, 0].set(title="Spectrogram (dB)", xlabel="Time (s)", ylabel="Hz",
                 ylim=(0, min(cfg["band"][1] * 1.2, sr / 2)))
    ax[1, 0].grid(False)

    m = mfcc(x, sr, cfg)
    im = ax[1, 1].imshow(m.T, origin="lower", aspect="auto", cmap="viridis",
                         extent=[0, len(x) / sr, 1, m.shape[1]])
    ax[1, 1].set(title="MFCC (13 coefficients)", xlabel="Time (s)",
                 ylabel="Coefficient")
    ax[1, 1].grid(False)
    fig.colorbar(im, ax=ax[1, 1], fraction=0.04)

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def filterbank_figure(cfg, path):
    sr = cfg["sr"]
    n_fft = int(2 ** np.ceil(np.log2(sr * cfg["frame_ms"] / 1000.0)))
    fb = mel_filterbank(sr, n_fft, cfg["n_mels"], cfg["band"][0],
                        min(cfg["band"][1], sr / 2))
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for row in fb:
        ax.plot(f, row, lw=0.8)
    ax.set(title=f"Mel filter bank ({cfg['n_mels']} filters, fs={sr} Hz)",
           xlabel="Frequency (Hz)", ylabel="Weight",
           xlim=(0, min(cfg["band"][1] * 1.1, sr / 2)))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def confusion_grid(results, labels, title, path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.6))
    axes = np.atleast_1d(axes)
    for ax, (name, r) in zip(axes, results.items()):
        cm = np.array(r["confusion_matrix"], dtype=float)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
        ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
               xlabel="Predicted", ylabel="True",
               title=f"{name}\nacc={r['test_accuracy']:.3f}")
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.grid(False)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def classifier_comparison(all_results, path):
    branches = list(all_results)
    models = list(next(iter(all_results.values())))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    xpos = np.arange(len(branches))
    for i, m in enumerate(models):
        vals = [all_results[b][m]["test_accuracy"] for b in branches]
        bars = ax.bar(xpos + i * w, vals, w, label=m)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set(xticks=xpos + w, ylim=(0, 1.08), ylabel="Test accuracy",
           title="Classifier comparison per branch")
    ax.set_xticklabels(branches)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def feature_scatter(X, y, names, path, title):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Z = PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(
        StandardScaler().fit_transform(X))
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for lab in sorted(set(y)):
        m = y == lab
        ax.scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.75, label=lab)
    ax.set(title=title, xlabel="PC 1", ylabel="PC 2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def ablation_figure(rows_by_branch, path):
    fig, axes = plt.subplots(1, len(rows_by_branch),
                             figsize=(4.6 * len(rows_by_branch), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (branch, rows) in zip(axes, rows_by_branch.items()):
        labs = [r["configuration"] for r in rows]
        vals = [r["cv_accuracy"] for r in rows]
        ax.barh(range(len(labs)), vals,
                color=["#2b6cb0"] + ["#c05621"] * 3 + ["#2f855a"] * 3)
        ax.set(yticks=range(len(labs)), title=branch, xlim=(0, 1.05),
               xlabel="CV accuracy")
        ax.set_yticklabels(labs, fontsize=8)
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7)
    fig.suptitle("Feature-group ablation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
