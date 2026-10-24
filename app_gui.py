"""Smart Audio Diagnostics - desktop demo UI.

Loads (or records) an audio file, runs the same M2/M3 DSP chain used for
training, and shows:

    Signal analysis   waveform, FFT spectrum, spectrogram, MFCC map
    DSP pipeline      raw vs processed signal, Butterworth response, mel bank
    Rhythm & timeline heart rate / breathing rate / tempo + per-segment votes
    Model performance confusion matrix and classifier comparison

then classifies it with the model saved by train_model.py. "Auto-detect"
routes the file to the right branch with the signal-type router.

    python train_model.py     # once
    python app_gui.py
"""
import csv
import re
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import joblib
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import signal as sps
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src import router
from src.features import extract, mel_filterbank, mfcc, stft_power
from src.preprocess import (denoise, preprocess, read_audio, resample_to,
                            trim_digital_silence)
from src.report import build_report
from src.rhythm import estimate_rate
from src.train import for_inference

try:
    import sounddevice as sd
except Exception:                                   # optional: record / play
    sd = None

BG = "#12161f"
PANEL = "#1b212e"
CARD = "#232b3b"
FG = "#e6ebf5"
MUTED = "#8d99b0"
ACCENT = "#5b9bff"
OK = "#38b48b"
WARN = "#e0803a"

AUTO = "Auto-detect"
# Medical classes keep fixed colours (normal = green, findings = warm) so the
# timeline reads the same way as the result card.
CLASS_COLOURS = {"normal": OK, "abnormal": WARN, "wheeze": "#d9b44a",
                 "crackle": "#e0607e"}
DOMAIN_LABEL = {
    "music": "Music - genre",
    "heart": "Medical - heart sound",
    "respiratory": "Medical - respiratory sound",
}
LABEL_DOMAIN = {v: k for k, v in DOMAIN_LABEL.items()}
AUDIO_EXTS = {".wav", ".au", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
REPORTS = config.RESULTS / "reports"
RECORDINGS = config.DATA / "recordings"
RATE_MAX_S = 60.0          # rate analysis looks at the first minute at most


def class_scores(model, x):
    """Confidence per class, whichever API the chosen classifier exposes."""
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


def segment_probs(model, full, cfg):
    """Classifies every clip-length segment of a recording. Returns the
    classes, the per-segment probabilities and their mean."""
    n = int(round(cfg["sr"] * cfg["clip_s"]))
    k = max(1, len(full) // n)
    rows, classes = [], None
    for i in range(k):
        xx, sr = preprocess(full[i * n:(i + 1) * n], cfg["sr"], cfg)
        v, _ = extract(xx, sr, cfg)
        classes, p = class_scores(model, v[None, :])
        rows.append(p)
    rows = np.vstack(rows)
    return classes, rows, rows.mean(axis=0)


def true_label_from_name(path, domain):
    """Demo clips are named <domain>_<class>_NN.wav (or _recNN for a whole
    lung recording, labelled by the majority of its annotated cycles), so the
    ground truth is known and the UI can show whether the prediction matched."""
    m = re.match(rf"{domain}_([a-z]+)_(?:rec)?\d+", Path(path).stem)
    if m and m.group(1) in config.DOMAINS[domain]["classes"]:
        return m.group(1)
    return None


def capture_window(win, out_path):
    """Grabs the window's own pixels via PrintWindow, so the capture is not
    spoiled by anything drawn on top of it and is immune to display scaling."""
    import ctypes
    from ctypes import wintypes

    from PIL import Image

    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    hwnd = int(win.wm_frame(), 16)

    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top

    hdc = user32.GetWindowDC(hwnd)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    gdi32.SelectObject(mem, bmp)
    user32.PrintWindow(hwnd, mem, 2)          # 2 = PW_RENDERFULLCONTENT

    class BMI(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                    ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                    ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD),
                    ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG),
                    ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]

    bi = BMI(ctypes.sizeof(BMI), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(bi), 0)

    img = Image.frombuffer("RGB", (w, h), buf, "raw", "BGRX", 0, 1)
    img.save(out_path)

    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(hwnd, hdc)
    return img.size


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Audio Diagnostics - Group 09 - DSP, ECE MNNIT Allahabad")
        w = min(1640, self.winfo_screenwidth() - 60)
        h = min(940, self.winfo_screenheight() - 100)
        self.geometry(f"{w}x{h}+15+10")
        self.minsize(1150, 720)
        self.configure(bg=BG)

        self.models = {}
        self.router = None
        self.audio = None
        self.sr_in = None
        self.path = None
        self.feature_names = None
        self.detected = None            # router output in Auto-detect mode
        self.result = None              # last classification
        self.rate = None
        self.recording = False

        self._style()
        self._header()
        self._toolbar()
        self._body()
        self._load_models()
        self._status("Ready. Open or record audio, or pick a demo signal. "
                     "Auto-detect works out which branch the file belongs to.")

    # ------------------------------------------------------------ chrome
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=CARD,
                    bordercolor=PANEL, font=("Segoe UI", 10))
        s.configure("TFrame", background=BG)
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("Card.TFrame", background=CARD)
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("Panel.TLabel", background=PANEL, foreground=FG)
        s.configure("Card.TLabel", background=CARD, foreground=FG)
        s.configure("Muted.TLabel", background=PANEL, foreground=MUTED,
                    font=("Segoe UI", 9))
        s.configure("CardMuted.TLabel", background=CARD, foreground=MUTED,
                    font=("Segoe UI", 9))
        s.configure("Title.TLabel", background=PANEL, foreground=FG,
                    font=("Segoe UI Semibold", 15))
        s.configure("Result.TLabel", background=CARD, foreground=OK,
                    font=("Segoe UI Semibold", 22))
        s.configure("Rate.TLabel", background=CARD, foreground=FG,
                    font=("Segoe UI Semibold", 18))
        s.configure("Section.TLabel", background=PANEL, foreground=ACCENT,
                    font=("Segoe UI Semibold", 10))
        s.configure("TButton", background=CARD, foreground=FG, borderwidth=0,
                    padding=(12, 7), font=("Segoe UI", 10))
        s.map("TButton", background=[("active", "#313a4e"), ("disabled", PANEL)],
              foreground=[("disabled", MUTED)])
        s.configure("Accent.TButton", background=ACCENT, foreground="#0b1020",
                    font=("Segoe UI Semibold", 10))
        s.map("Accent.TButton", background=[("active", "#7db0ff")])
        s.configure("Rec.TButton", background="#7a2e3a", foreground=FG)
        s.map("Rec.TButton", background=[("active", "#a03a4a")])
        s.configure("TCombobox", fieldbackground=CARD, background=CARD,
                    foreground=FG, arrowcolor=FG, padding=5)
        self.option_add("*TCombobox*Listbox.background", CARD)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        s.map("TCombobox", fieldbackground=[("readonly", CARD)],
              foreground=[("readonly", FG)])
        s.configure("Treeview", background=CARD, fieldbackground=CARD,
                    foreground=FG, rowheight=21, borderwidth=0)
        s.configure("Treeview.Heading", background=PANEL, foreground=MUTED,
                    font=("Segoe UI Semibold", 9), borderwidth=0)
        s.map("Treeview", background=[("selected", "#313a4e")])
        s.configure("Conf.Horizontal.TProgressbar", background=ACCENT,
                    troughcolor=PANEL, borderwidth=0, thickness=13)
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=PANEL, foreground=MUTED,
                    padding=(14, 6), font=("Segoe UI", 10))
        s.map("TNotebook.Tab", background=[("selected", CARD)],
              foreground=[("selected", FG)])

    def _header(self):
        h = ttk.Frame(self, style="Panel.TFrame", padding=(18, 10))
        h.pack(fill="x")
        left = ttk.Frame(h, style="Panel.TFrame")
        left.pack(side="left")
        ttk.Label(left, text="Smart Audio Diagnostics",
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(left, text="One DSP front end, two classification branches  ·  "
                             "music genre  ·  heart-sound screening  ·  lung-sound screening",
                  style="Muted.TLabel").pack(anchor="w")
        self.models_lbl = ttk.Label(h, text="", style="Muted.TLabel", justify="right")
        self.models_lbl.pack(side="right")

    def _toolbar(self):
        t = ttk.Frame(self, style="Panel.TFrame", padding=(18, 0, 18, 10))
        t.pack(fill="x")
        ttk.Label(t, text="Signal type", style="Muted.TLabel").pack(side="left", padx=(0, 6))
        self.domain_var = tk.StringVar(value=AUTO)
        cb = ttk.Combobox(t, textvariable=self.domain_var, state="readonly", width=25,
                          values=[AUTO] + list(DOMAIN_LABEL.values()))
        cb.pack(side="left", padx=(0, 10))
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_domain())

        ttk.Button(t, text="Open audio", command=self.open_file).pack(side="left", padx=3)
        self.rec_btn = ttk.Button(t, text="● Record 6 s", style="Rec.TButton",
                                  command=self.record)
        self.rec_btn.pack(side="left", padx=3)
        if sd is None:
            self.rec_btn.state(["disabled"])
        self.demo_var = tk.StringVar(value="Demo signal")
        self.demo_cb = ttk.Combobox(t, textvariable=self.demo_var, state="readonly", width=28)
        self.demo_cb.pack(side="left", padx=3)
        self.demo_cb.bind("<<ComboboxSelected>>", lambda e: self.load_demo())
        self.play_btn = ttk.Button(t, text="▶ Play", command=self.play)
        self.play_btn.pack(side="left", padx=3)

        ttk.Button(t, text="Batch folder…", command=self.batch_folder).pack(side="right", padx=3)
        ttk.Button(t, text="Export PDF report", command=self.export_report).pack(side="right", padx=3)
        ttk.Button(t, text="Classify", style="Accent.TButton",
                   command=self.classify).pack(side="right", padx=(3, 10))

    def _body(self):
        body = ttk.Frame(self, padding=(14, 10))
        body.pack(fill="both", expand=True)

        side = ttk.Frame(body, style="Panel.TFrame", padding=14, width=420)
        side.pack(side="right", fill="y", padx=(14, 0))
        side.pack_propagate(False)

        self.nb = ttk.Notebook(body)
        self.nb.pack(side="left", fill="both", expand=True)
        self.figs, self.canvases = {}, {}
        for key, title in (("analysis", "Signal analysis"), ("dsp", "DSP pipeline"),
                           ("rhythm", "Rhythm & timeline"), ("model", "Model performance")):
            frame = ttk.Frame(self.nb)
            self.nb.add(frame, text=title)
            fig = Figure(figsize=(8.4, 6.6), facecolor=BG)
            cv = FigureCanvasTkAgg(fig, master=frame)
            cv.get_tk_widget().pack(fill="both", expand=True)
            self.figs[key], self.canvases[key] = fig, cv
        self._blank_all()

        ttk.Label(side, text="FILE", style="Section.TLabel").pack(anchor="w")
        self.file_lbl = ttk.Label(side, text="(none loaded)", style="Muted.TLabel",
                                  wraplength=385)
        self.file_lbl.pack(anchor="w", pady=(2, 8))

        self.route_lbl = ttk.Label(side, text="", style="Muted.TLabel", wraplength=385)
        self.route_lbl.pack(anchor="w", pady=(0, 6))

        ttk.Label(side, text="PREDICTION", style="Section.TLabel").pack(anchor="w")
        card = ttk.Frame(side, style="Card.TFrame", padding=12)
        card.pack(fill="x", pady=(4, 8))
        self.result_lbl = ttk.Label(card, text="—", style="Result.TLabel")
        self.result_lbl.pack(anchor="w")
        self.truth_lbl = ttk.Label(card, text="", style="CardMuted.TLabel")
        self.truth_lbl.pack(anchor="w")
        self.model_lbl = ttk.Label(card, text="no model loaded", style="CardMuted.TLabel",
                                   wraplength=370)
        self.model_lbl.pack(anchor="w", pady=(2, 8))
        self.conf_rows = ttk.Frame(card, style="Card.TFrame")
        self.conf_rows.pack(fill="x")

        rate_card = ttk.Frame(side, style="Card.TFrame", padding=(12, 8))
        rate_card.pack(fill="x", pady=(0, 8))
        self.rate_name = ttk.Label(rate_card, text="RATE", style="CardMuted.TLabel")
        self.rate_name.pack(anchor="w")
        self.rate_lbl = ttk.Label(rate_card, text="—", style="Rate.TLabel")
        self.rate_lbl.pack(anchor="w")

        ttk.Label(side, text="DSP FEATURES (M3)", style="Section.TLabel").pack(anchor="w")
        self.tree = ttk.Treeview(side, columns=("v",), show="tree headings", height=11)
        self.tree.heading("#0", text="Feature")
        self.tree.heading("v", text="Value")
        self.tree.column("#0", width=195, anchor="w", stretch=False)
        self.tree.column("v", width=160, anchor="e", stretch=True)
        self.tree.pack(fill="both", expand=True, pady=(4, 8))

        self.disclaimer = ttk.Label(side, text="", style="Muted.TLabel", wraplength=385)
        self.disclaimer.pack(anchor="w")

        self.status = ttk.Label(self, text="", style="Muted.TLabel", anchor="w",
                                background=PANEL, padding=(18, 6))
        self.status.pack(fill="x", side="bottom")

    def _blank_all(self):
        for key, titles in (
                ("analysis", ["Time-domain waveform (after pre-processing)",
                              "FFT magnitude spectrum", "Spectrogram (dB)",
                              "MFCC - 13 coefficients"]),
                ("dsp", ["Raw vs processed waveform", "Raw vs processed spectrum",
                         "Butterworth band-pass response", "Mel filter bank"]),
                ("rhythm", ["Envelope and detected events", "Autocorrelation of envelope",
                            "Per-segment class probability", ""])):
            fig = self.figs[key]
            fig.clear()
            for i, t in enumerate(titles):
                if not t:
                    continue
                ax = fig.add_subplot(2, 2, i + 1) if key != "rhythm" else \
                    fig.add_subplot(3, 1, i + 1)
                self._axis_style(ax, t)
                ax.text(0.5, 0.5, "load a signal", ha="center", va="center",
                        color=MUTED, transform=ax.transAxes, fontsize=9)
            fig.tight_layout(pad=2.0)
            self.canvases[key].draw()
        self._plot_model()

    @staticmethod
    def _axis_style(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=FG, fontsize=10, pad=8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#33405a")
        ax.grid(alpha=0.15, color=MUTED)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)

    @staticmethod
    def _legend(ax):
        leg = ax.legend(fontsize=7, facecolor=CARD, edgecolor="#33405a", loc="upper right")
        for t in leg.get_texts():
            t.set_color(FG)

    # ------------------------------------------------------------- logic
    def _status(self, msg):
        self.status.config(text=msg)
        self.update_idletasks()

    def _load_models(self):
        missing = []
        for d in config.DOMAINS:
            p = config.ROOT / "models" / f"{d}.joblib"
            if p.exists():
                self.models[d] = joblib.load(p)
                for_inference(self.models[d]["model"])
            else:
                missing.append(d)
        if router.MODEL_PATH.exists():
            self.router = joblib.load(router.MODEL_PATH)
            for_inference(self.router["model"])
        else:
            missing.append("router")
        if missing:
            messagebox.showwarning(
                "Model not found",
                "No trained model for: " + ", ".join(missing) +
                "\n\nRun:  python train_model.py")
        parts = []
        for d, b in self.models.items():
            src = "real" if str(b["meta"]["source"]).startswith("real") else "synthetic"
            parts.append(f"{d}: {b['meta']['holdout_accuracy'] * 100:.0f}% ({src})")
        if self.router:
            parts.append(f"router: {self.router['meta']['holdout_accuracy'] * 100:.0f}%")
        self.models_lbl.config(text="Models loaded  ·  " + "  ·  ".join(parts))
        self._on_domain()

    @property
    def domain(self):
        """The branch actually in use: chosen, or found by the router."""
        v = self.domain_var.get()
        if v == AUTO:
            return self.detected or "music"
        return LABEL_DOMAIN[v]

    def _on_domain(self):
        auto = self.domain_var.get() == AUTO
        if auto:
            demos = sorted(p.name for p in (config.DATA / "demo_wavs").glob("*.wav"))
        else:
            demos = sorted(p.name for p in
                           (config.DATA / "demo_wavs").glob(f"{self.domain}_*.wav"))
        self.demo_cb["values"] = demos or ["(run scripts/make_demo_wavs.py)"]
        self.demo_var.set("Demo signal")
        if self.audio is not None:
            if auto:
                self._route()
            else:
                self.route_lbl.config(text="")
            self._analyse()
        self._refresh_branch_info()

    def _refresh_branch_info(self):
        d = self.domain
        m = self.models.get(d)
        if m:
            meta = m["meta"]
            self.model_lbl.config(
                text=f"{meta['classifier']} · {meta['n_samples']} clips · "
                     f"held-out acc {meta['holdout_accuracy'] * 100:.1f}% · "
                     f"{'real' if str(meta['source']).startswith('real') else 'synthetic'} data")
        self.disclaimer.config(
            text="" if d == "music" else
            "Screening / signal-classification demonstration only. "
            "Not a clinical diagnosis.")
        self._plot_model()

    def open_file(self):
        p = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio", "*.wav *.au *.mp3 *.flac *.m4a *.ogg"),
                       ("WAV audio", "*.wav"), ("All files", "*.*")])
        if p:
            self._load(Path(p))

    def load_demo(self):
        name = self.demo_var.get()
        p = config.DATA / "demo_wavs" / name
        if p.exists():
            self._load(p)

    # ---------------------------------------------------------- record
    def record(self, seconds=6.0, sr=22050):
        """Records from the default microphone. The audio driver fills the
        buffer in the background; the countdown and the hand-over to _load
        run on the Tk thread via after(), never from a worker thread."""
        if sd is None or self.recording:
            return
        try:
            self._rec = sd.rec(int(seconds * sr), samplerate=sr, channels=1,
                               dtype="float32")
        except Exception as e:
            messagebox.showerror("Recording failed", str(e))
            return
        self.recording = True
        self._rec_sr = sr
        self._rec_end = datetime.now().timestamp() + seconds
        self.rec_btn.state(["disabled"])
        self._status(f"Recording {seconds:.0f} s from the default microphone…")
        self.after(100, self._record_tick)

    def _record_tick(self):
        left = self._rec_end - datetime.now().timestamp()
        if left > 0:
            self.rec_btn.config(text=f"● {left:.0f} s")
            self.after(100, self._record_tick)
            return
        try:
            sd.wait()
            RECORDINGS.mkdir(parents=True, exist_ok=True)
            out = RECORDINGS / f"rec_{datetime.now():%Y%m%d_%H%M%S}.wav"
            x = self._rec[:, 0]
            x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
            wavfile.write(out, self._rec_sr, (x * 32767).astype(np.int16))
        except Exception as e:
            out = None
            messagebox.showerror("Recording failed", str(e))
        finally:
            self.recording = False
            self.rec_btn.state(["!disabled"])
            self.rec_btn.config(text="● Record 6 s")
        if out:
            self._load(out)

    # ------------------------------------------------------------- load
    def _load(self, path):
        # Read into locals first: a file that fails must not wipe the one
        # already on screen.
        try:
            audio, sr_in = read_audio(path)
            audio = trim_digital_silence(np.asarray(audio, dtype=np.float64))
        except Exception as e:
            messagebox.showerror("Could not read file", str(e))
            return
        if len(audio) < 0.5 * sr_in:
            messagebox.showerror("Too short", "The recording is shorter than 0.5 s.")
            return
        if not np.any(audio):
            messagebox.showerror("Silent file", "The recording contains only silence.")
            return
        self.audio, self.sr_in = audio, sr_in
        self.path = Path(path)
        self.file_lbl.config(
            text=f"{self.path.name}\n{self.sr_in} Hz · "
                 f"{len(self.audio) / self.sr_in:.1f} s · {len(self.audio)} samples")
        self.result = None
        self.result_lbl.config(text="—")
        self.truth_lbl.config(text="")
        for w in self.conf_rows.winfo_children():
            w.destroy()
        if self.domain_var.get() == AUTO:
            self._route()
        else:
            self.route_lbl.config(text="")
        self._analyse()

    def _route(self):
        if not self.router:
            self.route_lbl.config(text="Auto-detect unavailable: run python -m src.router")
            return
        self._status("Auto-detecting signal type (router)…")
        r = router.analyse(self.audio, self.sr_in, self.router)
        d, probs = r["domain"], r["probs"]
        self.detected = d
        self.route_probs = probs
        self.route_unknown = r["unknown"]
        order = sorted(probs, key=probs.get, reverse=True)
        head = ("AUTO-DETECTED:  UNKNOWN SOUND - not music, heart or lung\n"
                f"(shown with the nearest branch: {d})" if r["unknown"] else
                "AUTO-DETECTED:  " + DOMAIN_LABEL[d].upper())
        self.route_lbl.config(
            text=head + "\n" + "   ".join(f"{k} {probs[k] * 100:.0f}%" for k in order),
            foreground=WARN if r["unknown"] else MUTED)
        self._refresh_branch_info()

    def _analyse(self):
        cfg = config.DOMAINS[self.domain]
        self._status("Pre-processing (M2) and extracting features (M3)...")
        x, sr = preprocess(self.audio, self.sr_in, cfg)
        self.x, self.sr = x, sr
        self.full = resample_to(np.asarray(self.audio, dtype=np.float64),
                                self.sr_in, cfg["sr"])
        self.vec, self.feature_names = extract(x, sr, cfg)
        self._plot(x, sr, cfg)
        self._plot_dsp(cfg)
        self._fill_table(cfg)
        self._compute_rate(cfg)
        self._plot_rhythm()
        self._status(f"Analysed as {DOMAIN_LABEL[self.domain]} with fs={sr} Hz, band-pass "
                     f"{cfg['band'][0]:.0f}-{cfg['band'][1]:.0f} Hz, "
                     f"{len(self.vec)} features. Press Classify.")

    def _compute_rate(self, cfg):
        seg = self.full[:int(RATE_MAX_S * cfg["sr"])]
        c = dict(cfg, clip_s=len(seg) / cfg["sr"])
        y, sr = preprocess(seg, cfg["sr"], c)
        self.x_full = y
        self.rate = estimate_rate(y, sr, self.domain)
        r = self.rate
        self.rate_name.config(text=r["name"].upper() + " (envelope autocorrelation)")
        if r.get("rate"):
            self.rate_lbl.config(text=f"{r['rate']:.0f} {r['unit']}   "
                                      f"· periodicity {r['periodicity']:.2f}")
        else:
            self.rate_lbl.config(text=f"— ({r.get('reason', 'n/a')})")

    # ------------------------------------------------------------ plots
    def _plot(self, x, sr, cfg):
        fig = self.figs["analysis"]
        fig.clear()
        fmax = min(cfg["band"][1] * 1.15, sr / 2)

        ax = fig.add_subplot(2, 2, 1)
        self._axis_style(ax, "Time-domain waveform (after pre-processing)")
        ax.plot(np.arange(len(x)) / sr, x, lw=0.4, color=ACCENT)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")

        ax = fig.add_subplot(2, 2, 2)
        self._axis_style(ax, "FFT magnitude spectrum")
        X = np.abs(np.fft.rfft(x)); f = np.fft.rfftfreq(len(x), 1 / sr)
        ax.semilogy(f, X + 1e-9, lw=0.5, color=OK)
        ax.set_xlim(0, fmax); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("|X(f)|")

        ax = fig.add_subplot(2, 2, 3)
        self._axis_style(ax, "Spectrogram (dB)")
        _, power, _, _, _ = stft_power(x, sr, cfg["frame_ms"], cfg["hop_ms"])
        ax.imshow(10 * np.log10(power.T + 1e-12), origin="lower", aspect="auto",
                  extent=[0, len(x) / sr, 0, sr / 2], cmap="magma")
        ax.set_ylim(0, fmax); ax.grid(False)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")

        ax = fig.add_subplot(2, 2, 4)
        self._axis_style(ax, "MFCC - 13 coefficients")
        m = mfcc(x, sr, cfg)
        ax.imshow(m.T, origin="lower", aspect="auto", cmap="viridis",
                  extent=[0, len(x) / sr, 1, m.shape[1]])
        ax.grid(False); ax.set_xlabel("Time (s)"); ax.set_ylabel("Coefficient")

        fig.tight_layout(pad=2.0)
        self.canvases["analysis"].draw()

    def _plot_dsp(self, cfg):
        """Shows each M2 stage, so the effect of the filtering is visible."""
        fig = self.figs["dsp"]
        fig.clear()
        sr = cfg["sr"]
        n = int(round(sr * cfg["clip_s"]))
        raw = self.full
        start = max(0, (len(raw) - n) // 2)
        raw = raw[start:start + n]
        raw = raw / (np.max(np.abs(raw)) + 1e-12)
        proc = self.x[:len(raw)]
        fmax = min(cfg["band"][1] * 1.4, sr / 2)

        ax = fig.add_subplot(2, 2, 1)
        self._axis_style(ax, "Raw vs processed waveform")
        t = np.arange(len(raw)) / sr
        ax.plot(t, raw, lw=0.4, color=MUTED, label="raw (resampled)")
        ax.plot(t[:len(proc)], proc, lw=0.4, color=ACCENT, alpha=0.85,
                label="denoised + band-passed")
        ax.set_xlabel("Time (s)")
        self._legend(ax)

        ax = fig.add_subplot(2, 2, 2)
        self._axis_style(ax, "Raw vs processed spectrum (Welch PSD)")
        nper = min(len(raw), 2048)
        f1, p1 = sps.welch(raw, sr, nperseg=nper)
        f2, p2 = sps.welch(proc, sr, nperseg=nper)
        ax.plot(f1, 10 * np.log10(p1 + 1e-14), lw=0.8, color=MUTED, label="raw")
        ax.plot(f2, 10 * np.log10(p2 + 1e-14), lw=0.8, color=OK, label="processed")
        lo, hi = cfg["band"]
        ax.axvspan(lo, min(hi, sr / 2), color=ACCENT, alpha=0.08)
        ax.set_xlim(0, fmax); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("dB")
        self._legend(ax)

        ax = fig.add_subplot(2, 2, 3)
        self._axis_style(ax, f"Butterworth band-pass, order 4 ×2 (zero-phase)")
        nyq = sr / 2
        hi_c = min(hi, 0.99 * nyq)
        sos = sps.butter(4, [max(lo, 1.0) / nyq, hi_c / nyq], btype="bandpass", output="sos")
        w, hresp = sps.sosfreqz(sos, worN=4096, fs=sr)
        mag = 20 * np.log10(np.abs(hresp) ** 2 + 1e-12)     # filtfilt squares |H|
        ax.semilogx(w[1:], mag[1:], color=WARN, lw=1.2)
        ax.axhline(-6, color=MUTED, lw=0.6, ls="--")
        ax.axvline(lo, color=MUTED, lw=0.6, ls=":")
        ax.axvline(hi_c, color=MUTED, lw=0.6, ls=":")
        ax.set_ylim(-80, 5); ax.set_xlim(max(1.0, lo / 10), sr / 2)
        ax.set_xlabel("Frequency (Hz, log scale)"); ax.set_ylabel("|H(f)|² (dB)")
        ax.text(0.98, 0.08, f"passband {lo:.0f}-{hi_c:.0f} Hz", transform=ax.transAxes,
                ha="right", color=FG, fontsize=8)

        ax = fig.add_subplot(2, 2, 4)
        self._axis_style(ax, f"Mel filter bank ({cfg['n_mels']} triangular filters)")
        n_fft = int(2 ** np.ceil(np.log2(sr * cfg["frame_ms"] / 1000.0)))
        fb = mel_filterbank(sr, n_fft, cfg["n_mels"], lo, min(hi, sr / 2))
        ff = np.fft.rfftfreq(n_fft, 1 / sr)
        cmap = matplotlib.colormaps["viridis"]
        for i, row in enumerate(fb):
            ax.plot(ff, row / (row.max() + 1e-12), lw=0.7, color=cmap(i / len(fb)))
        ax.set_xlim(0, fmax); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Weight")

        fig.tight_layout(pad=2.0)
        self.canvases["dsp"].draw()

    def _plot_rhythm(self):
        fig = self.figs["rhythm"]
        fig.clear()
        r = self.rate or {}

        ax = fig.add_subplot(3, 1, 1)
        self._axis_style(ax, f"{r.get('name', 'Rate')}: envelope and detected events")
        if r.get("rate"):
            env, fs = r["env"], r["env_fs"]
            t = np.arange(len(env)) / fs
            ax.plot(t, env / (env.max() + 1e-12), color=ACCENT, lw=0.9, label="envelope")
            for i, e in enumerate(r.get("events", [])):
                ax.axvline(e, color=WARN, lw=0.8, alpha=0.7,
                           label="detected event" if i == 0 else None)
            ax.set_xlabel("Time (s)")
            self._legend(ax)
        else:
            ax.text(0.5, 0.5, r.get("reason", "load a signal"), ha="center", va="center",
                    color=MUTED, transform=ax.transAxes, fontsize=9)

        ax = fig.add_subplot(3, 1, 2)
        self._axis_style(ax, "Autocorrelation of the envelope")
        if r.get("rate"):
            lags, ac = r["lags"], r["ac"]
            keep = lags <= min(lags[-1], r["window"][1] * 1.6)
            ax.plot(lags[keep], ac[keep], color=OK, lw=1.0)
            ax.axvspan(*r["window"], color=ACCENT, alpha=0.08, label="search window")
            ax.axvline(r["lag"], color=WARN, lw=1.2,
                       label=f"period {r['lag']:.3f} s → {r['rate']:.0f} {r['unit']}")
            ax.set_xlabel("Lag (s)")
            self._legend(ax)

        ax = fig.add_subplot(3, 1, 3)
        self._axis_style(ax, "Per-segment class probability" +
                         ("" if self.result is not None else " (press Classify)"))
        if self.result is not None:
            classes, rows = self.result["classes"], self.result["rows"]
            n_seg = rows.shape[0]
            clip = config.DOMAINS[self.domain]["clip_s"]
            t = np.arange(n_seg) * clip
            bottom = np.zeros(n_seg)
            cmap = matplotlib.colormaps["tab10"]
            for i, c in enumerate(classes):
                ax.bar(t, rows[:, i], width=clip * 0.92, bottom=bottom, align="edge",
                       color=CLASS_COLOURS.get(str(c), cmap(i)), label=c)
                bottom += rows[:, i]
            ax.set_xlim(0, n_seg * clip); ax.set_ylim(0, 1)
            ax.set_xlabel("Time (s)"); ax.set_ylabel("p")
            self._legend(ax)

        fig.tight_layout(pad=1.6)
        self.canvases["rhythm"].draw()

    def _plot_model(self):
        fig = self.figs["model"]
        fig.clear()
        bundle = self.models.get(self.domain)
        if not bundle:
            self.canvases["model"].draw()
            return
        meta = bundle["meta"]
        labels = meta["classes"]
        cm = np.asarray(meta["confusion_matrix"], dtype=float)

        ax = fig.add_subplot(1, 2, 1)
        self._axis_style(ax, f"Held-out confusion matrix - {self.domain}")
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j] * 100:.0f}%", ha="center",
                        va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "#0b1020")
        ax.set_xticks(range(len(labels)), labels, rotation=30)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.grid(False)

        ax = fig.add_subplot(1, 2, 2)
        self._axis_style(ax, "Classifier comparison (held-out test)")
        comp = meta.get("comparison")
        if comp:
            names = list(comp)
            acc = [comp[n]["test_accuracy"] for n in names]
            bal = [comp[n]["balanced_accuracy"] for n in names]
            xi = np.arange(len(names))
            ax.bar(xi - 0.2, acc, 0.4, color=ACCENT, label="accuracy")
            ax.bar(xi + 0.2, bal, 0.4, color=OK, label="balanced accuracy")
            for i in range(len(names)):
                ax.text(xi[i] - 0.2, acc[i] + 0.01, f"{acc[i]:.2f}", ha="center",
                        color=FG, fontsize=7)
                ax.text(xi[i] + 0.2, bal[i] + 0.01, f"{bal[i]:.2f}", ha="center",
                        color=FG, fontsize=7)
            ax.set_xticks(xi, names, fontsize=8)
            ax.set_ylim(0, 1.1)
            ax.axhline(1 / len(labels), color=MUTED, ls="--", lw=0.8)
            self._legend(ax)
        lines = [f"{meta['classifier']} · {meta['n_samples']} clips from "
                 f"{meta.get('n_groups', '?')} recordings · {meta['n_features']} features"]
        sc = meta.get("screening")
        if sc:
            lines.append(f"Per clip:  sensitivity {sc['sensitivity'] * 100:.1f}%   "
                         f"specificity {sc['specificity'] * 100:.1f}%   "
                         f"score {(sc['sensitivity'] + sc['specificity']) * 50:.1f}%")
        rec = meta.get("recording_level")
        if rec:
            line = (f"Per recording ({rec['n_recordings']} held-out, segments averaged):  "
                    f"accuracy {rec['accuracy'] * 100:.1f}%   "
                    f"balanced {rec['balanced_accuracy'] * 100:.1f}%")
            if "sensitivity" in rec:
                line += (f"   sens {rec['sensitivity'] * 100:.1f}%"
                         f"   spec {rec['specificity'] * 100:.1f}%")
            lines.append(line)
        lines.append(f"source: {meta['source']}")
        fig.text(0.02, 0.03, "\n".join(lines), color=MUTED, fontsize=8)
        fig.tight_layout(pad=2.0, rect=(0, 0.13, 1, 1))
        self.canvases["model"].draw()

    def _fill_table(self, cfg):
        self.tree.delete(*self.tree.get_children())
        idx = {n: i for i, n in enumerate(self.feature_names)}
        rows = [
            ("Sampling frequency", f"{self.sr} Hz"),
            ("Band-pass", f"{cfg['band'][0]:.0f}-{cfg['band'][1]:.0f} Hz"),
            ("Frame / hop", f"{cfg['frame_ms']:.0f} / {cfg['hop_ms']:.0f} ms"),
            ("Mel filters", f"{cfg['n_mels']}"),
            ("Spectral centroid", f"{self.vec[idx['spectral::centroid_mean']]:.1f} Hz"),
            ("Spectral bandwidth", f"{self.vec[idx['spectral::bandwidth_mean']]:.1f} Hz"),
            ("Spectral roll-off 85%", f"{self.vec[idx['spectral::rolloff_mean']]:.1f} Hz"),
            ("Spectral flatness", f"{self.vec[idx['spectral::flatness_mean']]:.4f}"),
            ("Zero-crossing rate", f"{self.vec[idx['temporal::zcr_mean']]:.4f}"),
            ("Short-time energy", f"{self.vec[idx['temporal::energy_mean']]:.5f}"),
            ("RMS", f"{self.vec[idx['temporal::rms_mean']]:.4f}"),
        ]
        for i in range(1, 4):
            rows.append((f"MFCC-{i} (mean)", f"{self.vec[idx[f'mfcc::mfcc{i}_mean']]:.3f}"))
        rows.append(("Feature vector length", f"{len(self.vec)}"))
        self.table_rows = rows
        for k, v in rows:
            self.tree.insert("", "end", text=k, values=(v,))

    # --------------------------------------------------------- classify
    def classify(self):
        if self.audio is None:
            messagebox.showinfo("No audio", "Load or record a file first.")
            return None
        bundle = self.models.get(self.domain)
        if not bundle:
            messagebox.showwarning("No model",
                                   "Run:  python train_model.py --branch " + self.domain)
            return None
        model, meta = bundle["model"], bundle["meta"]
        cfg = config.DOMAINS[self.domain]
        self._status("Classifying every segment…")

        # A single 5 s window from the middle of a 3-minute song is a fragile
        # sample of it. Classify every segment and average the probabilities.
        classes, rows, probs = segment_probs(model, self.full, cfg)
        self.n_segments = rows.shape[0]
        order = np.argsort(probs)[::-1]
        top = classes[order[0]]
        self.result = {"classes": classes, "rows": rows, "probs": probs, "top": top}

        colour = OK if (self.domain == "music" or top == "normal") else WARN
        if self.domain_var.get() == AUTO and getattr(self, "route_unknown", False):
            # Not music, heart or lung: a branch label would be meaningless,
            # so the card says so; the bars below stay as the nearest match.
            self.result["top"] = "unknown sound"
            colour, shown = MUTED, "UNKNOWN SOUND"
        else:
            shown = str(top).upper()
        ttk.Style(self).configure("Result.TLabel", foreground=colour)
        self.result_lbl.config(text=shown)
        truth = true_label_from_name(self.path, self.domain) if self.path else None
        if truth:
            ok = truth == top
            self.truth_lbl.config(text=f"Ground truth (held-out clip): {truth}  "
                                       f"{'✓ correct' if ok else '✗ wrong'}")
        else:
            self.truth_lbl.config(text="")
        for w in self.conf_rows.winfo_children():
            w.destroy()
        for i in order:
            row = ttk.Frame(self.conf_rows, style="Card.TFrame")
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=f"{classes[i]:<11}", style="Card.TLabel",
                      width=11).pack(side="left")
            pb = ttk.Progressbar(row, style="Conf.Horizontal.TProgressbar",
                                 length=170, maximum=1.0, value=float(probs[i]))
            pb.pack(side="left", padx=6)
            ttk.Label(row, text=f"{probs[i] * 100:5.1f}%",
                      style="CardMuted.TLabel").pack(side="left")

        conf = probs[order[0]]
        low = conf < 0.5 and len(classes) > 2 or conf < 0.65 and len(classes) == 2
        note = "" if self.domain == "music" else "  (screening result, not a diagnosis)"
        seg = (f" averaged over {self.n_segments} segments"
               if self.n_segments > 1 else "")
        auto = self.domain_var.get() == AUTO and self.detected
        if auto and getattr(self, "route_unknown", False):
            warn = ("  ⚠ unknown sound - this is not music, a heart or a lung recording; "
                    "the label above is not meaningful")
        elif auto and self.route_probs[self.detected] < 0.6:
            warn = "  ⚠ signal type uncertain - input may be outside all three domains"
        elif low:
            warn = ("  ⚠ low confidence - genre unclear or outside the 5 trained genres"
                    if self.domain == "music" else
                    "  ⚠ borderline result - repeat the recording or refer")
        else:
            warn = ""
        if self.result["top"] == "unknown sound":
            self._status("Unknown sound - not music, a heart or a lung recording, so no "
                         "branch result is given. Try a stethoscope recording or a song.")
        else:
            self._status(f"Predicted '{top}' with {conf * 100:.1f}% confidence "
                         f"using {meta['classifier']}{seg}{note}{warn}")
        self._plot_rhythm()
        return top, conf

    # ------------------------------------------------------- outputs
    def export_report(self, path=None):
        if self.result is None:
            if self.classify() is None:
                return None
        if path is None:
            REPORTS.mkdir(parents=True, exist_ok=True)
            path = filedialog.asksaveasfilename(
                title="Save PDF report", defaultextension=".pdf",
                initialdir=str(REPORTS),
                initialfile=f"{self.path.stem}_report.pdf",
                filetypes=[("PDF", "*.pdf")])
            if not path:
                return None
        meta = self.models[self.domain]["meta"]
        ctx = {
            "file": self.path.name, "domain": self.domain,
            "domain_label": DOMAIN_LABEL[self.domain],
            "prediction": self.result["top"],
            "probs": {c: float(p) for c, p in zip(self.result["classes"],
                                                  self.result["probs"])},
            "classifier": meta["classifier"], "source": meta["source"],
            "holdout": meta["holdout_accuracy"],
            "x": self.x, "sr": self.sr, "cfg": config.DOMAINS[self.domain],
            "x_full": self.x_full, "rate": self.rate,
            "features": self.table_rows,
            "router": ({"domain": self.detected, "probs": self.route_probs,
                        "unknown": getattr(self, "route_unknown", False)}
                       if self.domain_var.get() == AUTO and self.detected else None),
        }
        build_report(path, ctx)
        self._status(f"Report saved to {path}")
        return path

    def batch_folder(self, folder=None, out_csv=None):
        """Classifies every audio file in a folder - the 'tag my music library'
        and 'screen a day's recordings' use case."""
        if folder is None:
            folder = filedialog.askdirectory(title="Folder of audio files")
            if not folder:
                return None
        files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in AUDIO_EXTS)
        if not files:
            messagebox.showinfo("Batch", "No audio files in that folder.")
            return None
        rows = []
        for i, f in enumerate(files, 1):
            self._status(f"Batch {i}/{len(files)}: {f.name}")
            try:
                x, sr = read_audio(f)
                x = trim_digital_silence(np.asarray(x, dtype=np.float64))
                unknown = False
                if self.domain_var.get() == AUTO and self.router:
                    rr = router.analyse(x, sr, self.router)
                    d, unknown = rr["domain"], rr["unknown"]
                else:
                    d = self.domain
                cfg = config.DOMAINS[d]
                full = resample_to(np.asarray(x, dtype=np.float64), sr, cfg["sr"])
                classes, _rows, probs = segment_probs(self.models[d]["model"], full, cfg)
                j = int(np.argmax(probs))
                rows.append([f.name, f"unknown (~{d})" if unknown else d, classes[j],
                             f"{probs[j] * 100:.1f}", true_label_from_name(f, d) or ""])
            except Exception as e:
                rows.append([f.name, "-", f"error: {str(e)[:40]}", "", ""])
        REPORTS.mkdir(parents=True, exist_ok=True)
        out_csv = Path(out_csv or REPORTS / f"batch_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "branch", "prediction", "confidence_%", "true_label"])
            w.writerows(rows)

        scored = [r for r in rows if r[4]]
        acc = (sum(r[2] == r[4] for r in scored) / len(scored)) if scored else None
        top = tk.Toplevel(self)
        top.title(f"Batch results - {len(rows)} files")
        top.configure(bg=BG)
        top.geometry("820x520")
        tv = ttk.Treeview(top, columns=("b", "p", "c", "t"), show="tree headings")
        for col, name, wdt in (("#0", "File", 300), ("b", "Branch", 110),
                               ("p", "Prediction", 130), ("c", "Conf. %", 80),
                               ("t", "True label", 110)):
            tv.heading(col, text=name)
            tv.column(col, width=wdt, anchor="w")
        for r in rows:
            tv.insert("", "end", text=r[0], values=r[1:])
        tv.pack(fill="both", expand=True, padx=10, pady=10)
        summary = f"Saved {out_csv}"
        if acc is not None:
            summary += f"   ·   accuracy on files with a known label: {acc * 100:.1f}% " \
                       f"({len(scored)} files)"
        ttk.Label(top, text=summary, background=BG, foreground=MUTED).pack(
            anchor="w", padx=10, pady=(0, 10))
        self._status(summary)
        self.batch_window = top
        return out_csv, rows, acc

    def play(self):
        if self.audio is None:
            return
        if sd is not None:
            try:
                sd.stop()
                x = np.asarray(self.audio, dtype=np.float32)
                sd.play(x / (np.max(np.abs(x)) + 1e-9) * 0.9, self.sr_in)
                self._status(f"Playing {self.path.name}")
                return
            except Exception as e:
                self._status(f"Playback unavailable: {e}")
                return
        try:
            import winsound
            threading.Thread(
                target=lambda: winsound.PlaySound(str(self.path), winsound.SND_FILENAME),
                daemon=True).start()
            self._status(f"Playing {self.path.name}")
        except Exception as e:
            self._status(f"Playback unavailable: {e}")


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    app = App()
    if "--demo" in sys.argv:
        target = Path(_arg("--demo"))
        dom = _arg("--domain")
        if dom == "auto":
            app.domain_var.set(AUTO)
        else:
            dom = dom or target.stem.split("_")[0]
            app.domain_var.set(DOMAIN_LABEL[dom])
        app._on_domain()
        tab = int(_arg("--tab", 0))
        app.after(400, lambda: (app._load(target), app.classify(),
                                app.nb.select(tab)))
    if "--screenshot" in sys.argv:
        out = Path(_arg("--screenshot"))

        def shot():
            app.attributes("-topmost", True)
            app.lift()
            app.update()
            capture_window(app, out)
            print("saved", out)
            app.destroy()
        app.after(3500, shot)
    app.mainloop()
