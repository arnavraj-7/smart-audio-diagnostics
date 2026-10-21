"""System-level checks: loaders, splitting, rate estimation, trained models,
router, PDF report, file formats and the GUI itself.

    python tests/test_system.py           # everything
    python tests/test_system.py --no-gui  # skip the Tk window

Needs the trained models (python train_model.py, python -m src.router) and
the demo clips (python scripts/make_demo_wavs.py).
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src import router
from src.datasets import load_heart_real, load_respiratory_real
from src.preprocess import preprocess, read_audio
from src.report import build_report
from src.rhythm import estimate_rate
from src.train import for_inference, screening_scores, split

PASS = []
DEMO = config.DATA / "demo_wavs"


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")


def heart_signal(hr, sr=2000, dur=20.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    x = np.zeros(n)
    period = 60.0 / hr
    for beat in np.arange(0.05, dur, period):
        for off, f, a, d in ((0, 50, 1.0, 0.02), (min(0.32 * period, 0.3), 75, 0.7, 0.015)):
            i = int((beat + off + rng.uniform(-0.01, 0.01)) * sr)
            m = min(n - i, int(0.09 * sr))
            if m > 0:
                tt = np.arange(m) / sr
                x[i:i + m] += a * np.sin(2 * np.pi * f * tt) * np.exp(-tt / d)
    return x + 0.1 * rng.standard_normal(n)


def wav(path, sr, x):
    wavfile.write(path, sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))


# ------------------------------------------------------------ rate estimation
print("\n--- rate estimation (src/rhythm.py)")
for hr in (48, 72, 100, 140):
    cfg = dict(config.DOMAINS["heart"], clip_s=20.0)
    y, sr = preprocess(heart_signal(hr), 2000, cfg)
    r = estimate_rate(y, sr, "heart")
    check(f"heart rate {hr} bpm", r["rate"] and abs(r["rate"] - hr) < 3, f"got {r['rate']:.1f}")

rng = np.random.default_rng(1)
for bpm in (90, 128):
    sr, dur = 22050, 15
    x = 0.01 * rng.standard_normal(sr * dur)
    for b in np.arange(0, dur, 60 / bpm):
        i = int(b * sr)
        m = min(len(x) - i, 800)
        x[i:i + m] += rng.standard_normal(m) * np.exp(-np.arange(m) / 150)
    r = estimate_rate(x, sr, "music")
    check(f"music tempo {bpm} bpm", abs(r["rate"] - bpm) < 3, f"got {r['rate']:.1f}")

for br in (12, 24):
    sr, dur = 4000, 30
    t = np.arange(sr * dur) / sr
    x = rng.standard_normal(len(t)) * (0.5 * (1 + np.sin(2 * np.pi * br / 60 * t))) ** 2
    cfg = dict(config.DOMAINS["respiratory"], clip_s=dur)
    y, s = preprocess(x, sr, cfg)
    r = estimate_rate(y, s, "respiratory")
    check(f"breathing rate {br}/min", abs(r["rate"] - br) < 1.5, f"got {r['rate']:.1f}")

r = estimate_rate(np.zeros(4000 * 3), 4000, "respiratory")
check("too-short input reports a reason, not a number", r["rate"] is None and "reason" in r,
      r.get("reason", ""))

# ------------------------------------------------------------------ loaders
print("\n--- dataset loaders on synthetic fixtures")
tmp = Path(tempfile.mkdtemp())
try:
    pn = tmp / "physionet" / "training-a"
    pn.mkdir(parents=True)
    for name in ("a0001", "a0002", "a0003"):
        wav(pn / f"{name}.wav", 2000, 0.1 * np.random.default_rng(0).standard_normal(4000))
    (pn / "REFERENCE.csv").write_text("a0001,1\na0002,-1\na0003,-1\n")
    items = load_heart_real(tmp / "physionet", config.DOMAINS["heart"])
    labs = {g: l for _, l, g in items}
    check("PhysioNet: 1 -> abnormal, -1 -> normal",
          labs == {"a0001": "abnormal", "a0002": "normal", "a0003": "normal"}, str(labs))

    ic = tmp / "icbhi"
    ic.mkdir()
    wav(ic / "101_1b1_Al_sc_Meditron.wav", 4000,
        0.1 * np.random.default_rng(0).standard_normal(4000 * 10))
    (ic / "101_1b1_Al_sc_Meditron.txt").write_text(
        "0.0\t2.0\t0\t0\n2.0\t4.0\t1\t0\n4.0\t6.0\t0\t1\n6.0\t8.0\t1\t1\n")
    items = load_respiratory_real(ic, config.DOMAINS["respiratory"])
    labs = [l for _, l, _ in items]
    check("ICBHI: cycles labelled normal/crackle/wheeze", labs == ["normal", "crackle", "wheeze"],
          str(labs))
    check("ICBHI: 'both' cycle excluded", len(items) == 3)
    check("ICBHI: patient id is the group", {g for *_, g in items} == {"101"})
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ------------------------------------------------------------ split / metric
print("\n--- splitting and metrics")
X = np.arange(200)[:, None].astype(float)
y = np.array(["a", "b"] * 100)
g = np.repeat(np.arange(40), 5)
_, _, _, _, gtr, gte = split(X, y, g)
check("group split: no group in both train and test", not set(gtr) & set(gte),
      f"{len(set(gtr))} train / {len(set(gte))} test groups")

s = screening_scores(["normal", "normal", "wheeze", "crackle"],
                     ["normal", "wheeze", "wheeze", "normal"])
check("screening score: Se, Sp, mean", (s["sensitivity"], s["specificity"],
                                       s["icbhi_score"]) == (0.5, 0.5, 0.5), str(s))

# ----------------------------------------------------------- file formats
print("\n--- audio file formats")
tmp = Path(tempfile.mkdtemp())
try:
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(22050) / 22050)
    wav(tmp / "t.wav", 22050, tone)
    x, sr = read_audio(tmp / "t.wav")
    check("read .wav", sr == 22050 and len(x) == 22050)
    au = next((config.DATA / "gtzan" / "genres" / "jazz").glob("*.au"), None)
    if au:
        x, sr = read_audio(au)
        check("read GTZAN .au", sr == 22050 and len(x) > 22050 * 29, f"{len(x) / sr:.1f} s")
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp / "t.wav"),
                        str(tmp / "t.mp3")], check=True)
        x, sr = read_audio(tmp / "t.mp3")
        check("read .mp3 via ffmpeg", abs(len(x) / sr - 1.0) < 0.1, f"{len(x) / sr:.2f} s")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ------------------------------------------------------------- the models
print("\n--- trained models")
models = {}
for d, cfg in config.DOMAINS.items():
    p = config.ROOT / "models" / f"{d}.joblib"
    check(f"model file exists: {d}", p.exists())
    if p.exists():
        b = joblib.load(p)
        for_inference(b["model"])
        models[d] = b
        check(f"{d}: classes match config", sorted(b["model"].classes_) == sorted(cfg["classes"]),
              str(list(b["model"].classes_)))
        check(f"{d}: trained on real data", str(b["meta"]["source"]).startswith("real"),
              b["meta"]["source"])
        chance = 1 / len(cfg["classes"])
        check(f"{d}: held-out balanced accuracy well above chance",
              b["meta"]["holdout_balanced_accuracy"] > chance + 0.15,
              f"{b['meta']['holdout_balanced_accuracy']:.3f} vs chance {chance:.2f}")

sys.path.insert(0, str(config.ROOT))
from app_gui import segment_probs, true_label_from_name  # noqa: E402
from src.preprocess import resample_to  # noqa: E402

demos = sorted(DEMO.glob("*.wav"))
check("demo clips present", len(demos) >= 30, f"{len(demos)} files")
per = {}
for f in demos:
    d = f.stem.split("_")[0]
    truth = true_label_from_name(f, d)
    if truth is None or d not in models:
        continue
    x, sr = read_audio(f)
    cfg = config.DOMAINS[d]
    classes, _, probs = segment_probs(models[d]["model"],
                                      resample_to(x, sr, cfg["sr"]), cfg)
    per.setdefault(d, []).append(classes[int(np.argmax(probs))] == truth)
for d, hits in per.items():
    print(f"     demo accuracy {d:<12} {sum(hits)}/{len(hits)}")
    check(f"{d}: demo clips mostly correct", np.mean(hits) >= 0.5, f"{np.mean(hits):.2f}")

# ------------------------------------------------------------------ router
print("\n--- signal-type router")
check("router model exists", router.MODEL_PATH.exists())
if router.MODEL_PATH.exists():
    rb = joblib.load(router.MODEL_PATH)
    right = 0
    wrong = []
    for f in demos:
        x, sr = read_audio(f)
        d, _ = router.detect(x, sr, rb)
        if d == f.stem.split("_")[0]:
            right += 1
        else:
            wrong.append(f.name)
    check("router routes demo clips to the right branch", right / len(demos) >= 0.9,
          f"{right}/{len(demos)} {wrong[:4]}")
    check("router held-out accuracy (4 classes incl. 'other') >= 0.93",
          rb["meta"]["holdout_accuracy"] >= 0.93, f"{rb['meta']['holdout_accuracy']:.3f}")
    # Sounds the router never saw in training: Windows system sounds (chimes,
    # alarms, ring tones, synthetic speech). They must be flagged as unknown.
    media = sorted(Path(r"C:\Windows\Media").glob("*.wav"))
    if media:
        flagged = sum(router.analyse(*read_audio(f), rb)["unknown"] for f in media)
        check("unknown-sound rejection on Windows system sounds >= 80 %",
              flagged / len(media) >= 0.8, f"{flagged}/{len(media)}")

# --------------------------------------------------------------- PDF report
print("\n--- PDF report")
tmp = Path(tempfile.mkdtemp())
try:
    f = next(DEMO.glob("heart_abnormal_*.wav"))
    cfg = config.DOMAINS["heart"]
    x, sr = read_audio(f)
    y, s = preprocess(x, sr, cfg)
    yf, _ = preprocess(x, sr, dict(cfg, clip_s=len(x) / sr))
    out = build_report(tmp / "r.pdf", {
        "file": f.name, "domain": "heart", "domain_label": "Medical - heart sound",
        "prediction": "abnormal", "probs": {"normal": 0.3, "abnormal": 0.7},
        "classifier": "SVM", "source": "real", "holdout": 0.9, "x": y, "sr": s,
        "cfg": cfg, "x_full": yf, "rate": estimate_rate(yf, s, "heart"),
        "features": [("a", "1"), ("b", "2")], "router": None})
    data = Path(out).read_bytes()
    check("report is a non-trivial PDF", data[:4] == b"%PDF" and len(data) > 20000,
          f"{len(data)} bytes")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------------- GUI
if "--no-gui" not in sys.argv:
    print("\n--- GUI (Tk window, driven programmatically)")
    import app_gui
    app_gui.messagebox.showwarning = lambda *a, **k: None
    app = app_gui.App()
    app.update()
    check("GUI starts with all models", set(app.models) == set(config.DOMAINS)
          and app.router is not None)
    for f in [next(DEMO.glob(f"{d}_*_01.wav")) for d in config.DOMAINS] + \
             list(DEMO.glob("respiratory_recording_*.wav"))[:1]:
        app.domain_var.set(app_gui.AUTO)
        app._on_domain()
        app._load(f)
        app.update()
        expect = f.stem.split("_")[0]
        check(f"GUI auto-detect {f.name}", app.domain == expect, app.domain)
        top, conf = app.classify()
        app.update()
        check(f"GUI classify {f.name}", top in config.DOMAINS[app.domain]["classes"]
              and 0 < conf <= 1, f"{top} {conf:.2f}")
        for i in range(4):
            app.nb.select(i)
            app.update()
    # A file that cannot be used must not wipe the one on screen.
    app_gui.messagebox.showerror = lambda *a, **k: None
    before = app.path
    tmp = Path(tempfile.mkdtemp())
    wav(tmp / "short.wav", 8000, np.zeros(1000))
    app._load(tmp / "short.wav")
    check("GUI rejects a too-short file and keeps the current one",
          app.path == before and app.audio is not None)
    shutil.rmtree(tmp, ignore_errors=True)

    if app_gui.sd is not None and "--no-mic" not in sys.argv:
        import time
        try:
            app_gui.sd.check_input_settings(channels=1, samplerate=22050)
            has_mic = True
        except Exception:
            has_mic = False
        if has_mic:
            app.record(seconds=2)
            t0 = time.time()
            while app.recording and time.time() - t0 < 15:
                app.update()
                time.sleep(0.05)
            app.update()
            check("GUI records 2 s from the microphone and analyses it",
                  app.path is not None and app.path.name.startswith("rec_")
                  and abs(len(app.audio) / app.sr_in - 2.0) < 0.2, str(app.path))

    tmp = Path(tempfile.mkdtemp())
    try:
        p = app.export_report(tmp / "gui.pdf")
        check("GUI exports PDF", Path(p).exists() and Path(p).stat().st_size > 20000)
        sub = tmp / "batch"
        sub.mkdir()
        for f in demos[::5]:
            shutil.copy(f, sub)
        app.domain_var.set(app_gui.AUTO)
        out_csv, rows, acc = app.batch_folder(sub, tmp / "batch.csv")
        check("GUI batch mode classifies a folder", len(rows) == len(list(sub.iterdir()))
              and out_csv.exists(), f"{len(rows)} files, acc={acc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    app.destroy()

print(f"\n{sum(PASS)}/{len(PASS)} checks passed")
sys.exit(0 if all(PASS) else 1)
