"""One-page PDF report for a single analysed recording.

This is the deliverable a screening workflow needs: a health worker records a
heart or lung sound, the app classifies it, and the report goes to the doctor
with the evidence (waveform, spectrogram, rate, confidence) on one page.
For music it doubles as a track-analysis sheet.
"""
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .features import stft_power

DISCLAIMER = ("Screening / signal-classification output only. This is NOT a "
              "clinical diagnosis. Refer to a qualified clinician.")


def build_report(path, ctx):
    """ctx keys: file, domain, domain_label, prediction, probs (dict),
    classifier, source, holdout, x, sr, cfg, rate (dict or None),
    features (list of (name, value)), router (dict or None)."""
    x, sr, cfg = ctx["x"], ctx["sr"], ctx["cfg"]
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))          # A4 portrait
        fig.patch.set_facecolor("white")

        fig.text(0.06, 0.965, "Smart Audio Diagnostics - Analysis Report",
                 fontsize=16, fontweight="bold")
        fig.text(0.06, 0.945, "Group 09 · Digital Signal Processing · ECE, MNNIT Allahabad",
                 fontsize=9, color="#555")
        fig.text(0.94, 0.965, datetime.now().strftime("%d %b %Y, %H:%M"),
                 fontsize=9, ha="right", color="#555")

        lines = [
            ("File", ctx["file"]),
            ("Signal type", ctx["domain_label"]),
            ("Classifier", f"{ctx['classifier']}  (held-out accuracy "
                           f"{ctx['holdout'] * 100:.1f} %)"),
            ("Trained on", ctx["source"]),
        ]
        if ctx.get("router"):
            r = ctx["router"]
            lines.insert(2, ("Auto-detected as",
                             "UNKNOWN SOUND - result not meaningful" if r.get("unknown") else
                             f"{r['domain']} ({r['probs'][r['domain']] * 100:.0f} % confidence)"))
        y0 = 0.915
        for k, v in lines:
            fig.text(0.06, y0, k, fontsize=9, color="#555")
            fig.text(0.25, y0, str(v)[:95], fontsize=9)
            y0 -= 0.018

        # Result box
        top = ctx["prediction"]
        ax = fig.add_axes([0.06, 0.70, 0.40, 0.12])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#eef3fb", transform=ax.transAxes))
        ax.text(0.05, 0.70, "RESULT", fontsize=9, color="#2b6cb0",
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.28, str(top).upper(), fontsize=24, fontweight="bold",
                color="#1a7f5a" if top == "normal" or ctx["domain"] == "music" else "#c05621",
                transform=ax.transAxes)
        if ctx.get("rate") and ctx["rate"].get("rate"):
            rr = ctx["rate"]
            ax.text(0.97, 0.28, f"{rr['name']}\n{rr['rate']:.0f} {rr['unit']}",
                    fontsize=10, ha="right", transform=ax.transAxes)

        # Confidence bars
        ax = fig.add_axes([0.58, 0.70, 0.36, 0.12])
        probs = ctx["probs"]
        names = sorted(probs, key=probs.get)
        vals = [probs[n] * 100 for n in names]
        ax.barh(names, vals, color=["#2b6cb0" if n == top else "#a0aec0" for n in names])
        for i, v in enumerate(vals):
            ax.text(v + 1, i, f"{v:.1f} %", va="center", fontsize=8)
        ax.set_xlim(0, 115)
        ax.set_title("Class confidence", fontsize=9, loc="left")
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        # Waveform
        ax = fig.add_axes([0.08, 0.50, 0.86, 0.14])
        xf = ctx.get("x_full", x)
        t = np.arange(len(xf)) / sr
        ax.plot(t, xf, lw=0.4, color="#2b6cb0")
        rate = ctx.get("rate") or {}
        if rate.get("rate") and rate.get("events") is not None:
            for e in rate["events"]:
                ax.axvline(e, color="#c05621", lw=0.6, alpha=0.6)
        ax.set_xlim(0, t[-1] if len(t) else 1)
        ax.set_title("Pre-processed recording" +
                     ("  (orange = detected beats / breaths)" if rate.get("rate") else ""),
                     fontsize=9, loc="left")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=7)

        # Spectrogram
        ax = fig.add_axes([0.08, 0.29, 0.86, 0.15])
        _, power, _, _, _ = stft_power(x, sr, cfg["frame_ms"], cfg["hop_ms"])
        fmax = min(cfg["band"][1] * 1.15, sr / 2)
        ax.imshow(10 * np.log10(power.T + 1e-12), origin="lower", aspect="auto",
                  extent=[0, len(x) / sr, 0, sr / 2], cmap="magma")
        ax.set_ylim(0, fmax)
        ax.set_title("Spectrogram (dB)", fontsize=9, loc="left")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Hz", fontsize=8)
        ax.tick_params(labelsize=7)

        # Feature table
        ax = fig.add_axes([0.06, 0.07, 0.88, 0.17])
        ax.axis("off")
        feats = ctx["features"][:12]
        half = (len(feats) + 1) // 2
        cells = []
        for i in range(half):
            a = feats[i]
            b = feats[i + half] if i + half < len(feats) else ("", "")
            cells.append([a[0], a[1], b[0], b[1]])
        tbl = ax.table(cellText=cells, colLabels=["DSP feature", "Value", "DSP feature", "Value"],
                       loc="upper center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.25)

        if ctx["domain"] != "music":
            fig.text(0.5, 0.035, DISCLAIMER, ha="center", fontsize=8,
                     color="#c05621", style="italic")
        fig.text(0.5, 0.018, "Generated by Smart Audio Diagnostics (Python · NumPy · SciPy · "
                 "scikit-learn). MFCC computed from first principles.",
                 ha="center", fontsize=7, color="#888")
        pdf.savefig(fig)
        plt.close(fig)
    return path
