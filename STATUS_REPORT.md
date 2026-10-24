# Status Report — 21 October 2026

**Smart Audio Diagnostics · Group 09 · DSP · ECE, MNNIT Allahabad**

All three branches are now trained and tested on **real datasets**. Every
number below was produced by the code in this folder; section 7 has the commands.
The teammate-facing explanation (use case, study plan, PPT outline, abstract) is
in **`TEAM_GUIDE.md`**.

---

## 1. Headline

| Branch | Dataset | Real? | Clips (recordings) | Classes | Shipped model | Test acc. | Balanced acc. | Notes |
|---|---|---|---|---|---|---|---|---|
| **Music** | GTZAN | **REAL** | 2 995 (500 tracks) | 5 | Random Forest (400 trees) | 0.822 | **0.812** | per song 0.832 |
| **Heart** | PhysioNet/CinC 2016 | **REAL** | 8 147 (3 240 recordings) | 2 | Random Forest (depth 12) | 0.857 | **0.833** | Se 0.790, Sp 0.876 |
| **Lungs** | ICBHI 2017 | **REAL** | 6 392 cycles (126 patients) | 3 | SVM (C=1, γ=0.001) | 0.505 | **0.504** | ICBHI score 0.512 |
| **Router** | all three + ESC-50 | **REAL** | 2 817 | 4 incl. *other* | Random Forest | **0.946** | — | 61/70 unseen sounds rejected |

Previously (8 Sep) only music was real. Heart and lungs ran on a synthetic
generator. That generator is still in `src/synth.py` but is no longer used
by any shipped model.

---

## 2. What changed since 8 September

### Completed items from the old "still needed" list

| Old item | Status |
|---|---|
| Download PhysioNet 2016 and ICBHI 2017 | **Done.** PhysioNet from the AWS open-data mirror (190 MB); ICBHI from the challenge site (1.98 GB; expired TLS certificate, fetched with `curl -k`). |
| Class imbalance (`class_weight='balanced'`, lead with Se/Sp) | **Done.** All SVM/RF models balanced; best model chosen by balanced accuracy; Se/Sp/ICBHI score reported. |
| Hyper-parameter search (`--tune`) | **Done** on all three branches. The lung SVM went from 0.426 → 0.504 balanced accuracy. |
| MATLAB GUI wiring | **Superseded.** `matlab/sa_demo.m` gives a complete MATLAB read → preprocess → features → classify → plot path, segment-averaged like the Python app. |
| Run `sa_verify` and record the error | **Done.** Max relative error ≤ 6×10⁻⁸ on all 53 features, all 3 branches, with and without denoising (`results/tables/matlab_verify.csv`). |

### Bugs found and fixed

1. **MATLAB/Python hop mismatch:** a 10 ms hop at 22 050 Hz is 220.5 samples.
   Python's `round` → 220 (banker's rounding), MATLAB's → 221. Every music feature
   drifted ~0.1 %. Fixed with `matlab/sa_round.m`.
2. **MATLAB denoise ≠ SciPy denoise:** MATLAB's `stft/istft` pad differently from
   `scipy.signal.stft`. Re-implemented to SciPy's framing. MATLAB demo-clip
   accuracy went from 12/40 to 32/40.
3. **GUI crash:** RF models predicting with `n_jobs=-1` inside Tk caused
   `Tcl_AsyncDelete: async handler deleted by the wrong thread`. Shipped models
   now predict single-threaded (`src/train.py: for_inference`).
4. **Recording thread** touched Tk widgets from a worker thread. Now polled on
   the Tk thread.
5. **Failed file load** wiped the file on screen. Now it is kept.
6. **ICBHI memory:** loading 6 000 cycles at 44.1 kHz would take several GB; audio is
   now resampled to the branch rate while loading, and each recording is decoded once.

### New features

| Feature | Where |
|---|---|
| Signal-type router (auto-detect music / heart / lung / **unknown**) | `src/router.py` |
| Heart-rate, breathing-rate and tempo estimation (envelope autocorrelation) | `src/rhythm.py` |
| Whole-recording classification (all segments, averaged) + per-segment timeline | `app_gui.py` |
| GUI tabs: Signal analysis · DSP pipeline · Rhythm & timeline · Model performance | `app_gui.py` |
| Live microphone recording, playback of any format | `app_gui.py` (sounddevice) |
| One-page PDF screening report | `src/report.py` |
| Batch folder classification → CSV | `app_gui.py` |
| Ground-truth ✓/✗ for demo clips | `app_gui.py` |
| Per-recording evaluation | `src/train.py: recording_level` |
| MATLAB end-to-end demo | `matlab/sa_demo.m` |
| System test suite (53 checks incl. GUI, mic, PDF) | `tests/test_system.py` |

---

## 3. Per-branch detail (held-out test split, shipped models)

### Music — GTZAN, 5 genres, chance 0.20

| Genre | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| classical | 0.94 | 0.95 | 0.95 | 192 |
| jazz | 0.69 | 0.80 | 0.75 | 102 |
| rock | 0.77 | 0.69 | 0.73 | 161 |
| pop | 0.83 | 0.84 | 0.84 | 180 |
| hiphop | 0.80 | 0.77 | 0.79 | 112 |

Rock/jazz remain the main confusion pair (broadband, percussive).

### Heart — PhysioNet 2016, normal vs abnormal, chance 0.50

| | Precision | Recall | Support |
|---|---|---|---|
| normal | 0.94 | 0.88 (= specificity) | 1 579 |
| abnormal | 0.64 | 0.79 (= sensitivity) | 438 |

Per recording (810 held-out recordings, segments averaged): accuracy 0.828,
balanced 0.811, Se 0.782, Sp 0.841. The data come from six sites and several
stethoscopes; the split is recording-disjoint.

### Lungs — ICBHI 2017, normal / wheeze / crackle, chance 0.33

| | Precision | Recall | Support |
|---|---|---|---|
| normal | 0.70 | 0.45 | 971 |
| wheeze | 0.29 | 0.43 | 228 |
| crackle | 0.44 | 0.64 | 538 |

ICBHI score (mean of Se and Sp) = 0.512. Patient-disjoint split (32 test
patients). Four pre-processing variants were tested; none beat the baseline
beyond noise (`results/RESPIRATORY_EXPERIMENTS.md`). The limit is clip-level
features averaging away < 20 ms crackles. Classical methods in the literature
sit in the same range.

### Router — 4 classes

Held-out confusion (rows = true; music, heart, lung, other):
`[[148,0,1,5],[0,162,0,0],[0,0,151,8],[18,0,5,193]]`. Heart is never confused.
On 70 Windows system sounds it never saw (chimes, alarms, synthetic speech),
61 were rejected as *unknown*.

---

## 4. Classifier comparison and ablation (`run_experiment.py`, default hyper-parameters)

| Branch | SVM (RBF) test / bal | k-NN test / bal | Random Forest test / bal |
|---|---|---|---|
| Music | 0.803 / 0.799 | 0.735 / 0.721 | **0.818 / 0.807** |
| Heart | 0.853 / 0.795 | 0.842 / 0.725 | **0.866 / 0.825** |
| Lungs | 0.525 / 0.426 | 0.533 / 0.398 | 0.537 / 0.426 |

| Ablation (CV acc., Δ vs all) | Music | Heart | Lungs |
|---|---|---|---|
| All 53 features | 0.790 | 0.873 | 0.538 |
| without MFCC | −0.127 | −0.016 | −0.073 |
| MFCC only | −0.020 | −0.012 | −0.030 |
| spectral only | −0.189 | −0.037 | −0.121 |
| temporal only | −0.218 | −0.036 | −0.101 |

---

## 5. Over/under-fitting (`analyze_fit.py`, shipped hyper-parameters)

| Branch | Chance | Train | CV | Test | Verdict |
|---|---|---|---|---|---|
| music | 0.20 | 1.000 | 0.772 | 0.822 | generalises; train = 1 is inherent to RF |
| heart | 0.50 | 0.958 | 0.868 | 0.857 | well fitted |
| lungs | 0.33 | 0.567 | 0.491 | 0.505 | well fitted; feature-limited, not over-fitted |

---

## 6. Testing done on 21 Oct

| Test | Result |
|---|---|
| `tests/test_dsp.py` (DSP maths) | 19/19 |
| `tests/test_system.py` (loaders, splits, rate estimator, models, router, PDF, file formats, GUI driven programmatically, live mic) | 53/53, run three times |
| Demo dropdown (31 held-out clips, kept only if the app gets them right; `make_demo_wavs.py --verified`) | 31/31 opened one by one in the GUI: right branch, right class, no warnings, PDF exported |
| Raw dataset files, MP3, stereo, too-short, silent, unseen genres, Windows sounds | all handled (see `DEMO_SCRIPT.md`) |
| MATLAB `sa_verify` × 3 branches × {denoise on, off} | max rel. error ≤ 6×10⁻⁸ |
| MATLAB `sa_demo` on the 31 demo clips | 26/31 (simpler k-NN on the exported model) |
| Every GUI tab inspected visually | screenshots in `results/figures/ui_*.png` |

---

## 7. Reproducing

```bash
python tests/test_dsp.py
python tests/test_system.py
python train_model.py --tune          # ~10 min, features are cached
python -m src.router                  # router (+ --no-cache to rebuild features)
python run_experiment.py              # metrics, ablation, all figures
python analyze_fit.py
python scripts/make_demo_wavs.py
python scripts/export_for_matlab.py
python app_gui.py
```

First-time feature extraction from raw audio: music ~1 min, heart ~2 min,
lungs ~1.5 min (then cached in `results/cache/`).

---

## 8. Scope statement

The medical branches perform acoustic signal classification and screening. They are
evaluated with signal-classification metrics on public research datasets. They
are not a diagnostic tool and no clinical claim is made.
