# Smart Audio Diagnostics

**Classifying Music and Medical Signals — a Unified DSP and Machine Learning Framework**
Group 09 · Digital Signal Processing · ECE, MNNIT Allahabad

One domain-agnostic DSP front end (FFT, MFCC, spectral centroid, bandwidth,
roll-off, flatness, zero-crossing rate, short-time energy) feeding two
independent classification branches:

- **Music branch** — genre classification (classical, jazz, rock, pop, hiphop)
- **Medical branch** — heart sounds (normal / abnormal) and respiratory sounds
  (normal / wheeze / crackle)

The two branches share the feature-extraction code and nothing else. Genre
labels and clinical-screening labels are never mixed into one classifier.

---

## Quick start

```bash
pip install -r requirements.txt
python app_gui.py               # the demo app (models are already trained)
python tests/test_dsp.py        # 19 checks on the DSP front end
python tests/test_system.py     # 52 checks: loaders, models, router, PDF, GUI, mic
```

### What is in this repository, and what is not

Code, trained models (`models/`), the MATLAB mirror, all result tables and
figures, sample PDF reports and the heart/lung demo clips are included, so
`python app_gui.py` works straight after cloning.

**Not included:** the raw datasets (GTZAN, PhysioNet 2016, ICBHI 2017,
ESC-50: about 7 GB, each under its own licence) and the GTZAN music demo
clips (copyrighted recordings). `DOWNLOADS.md` says where to get each dataset;
after that, `python scripts/make_demo_wavs.py` regenerates every demo clip and
`python train_model.py --tune` retrains from scratch. Until GTZAN is
downloaded, use **Open audio** with any song of your own for the music branch.

**New teammate? Read `TEAM_GUIDE.md` first.** It covers the use case, how
everything works, who studies what, the PPT outline and the abstract.

All three branches run on **real datasets**:

| Branch | Dataset | Balanced accuracy (held-out) |
|---|---|---|
| Music genre (5) | GTZAN | 81.2 % |
| Heart sound normal/abnormal | PhysioNet/CinC 2016 | 83.3 % (Se 79 %, Sp 88 %) |
| Lung sound normal/wheeze/crackle | ICBHI 2017 | 50.4 % (ICBHI score 51.2 %) |
| Signal-type router (+ unknown) | all three + ESC-50 | 94.6 % accuracy |

The app auto-detects the signal type, shows every DSP stage, estimates heart
rate / breathing rate / tempo, classifies the whole recording, records from
the microphone, exports a PDF report and batch-classifies folders.

---

## Documents

| File | Contents |
|---|---|
| `TEAM_GUIDE.md` | **Start here.** Use case, how it works, results, work split for the three of us, PPT outline, abstract |
| `STATUS_REPORT.md` | Latest results, what changed, bugs fixed, tests run |
| `DEMO_SCRIPT.md` | 6-minute demo for the examiner + answers to likely questions |
| `DSP_EXPLAINED.md` | Every DSP step with formulas, MATLAB and Python side by side |
| `HOW_IT_WORKS.md` | Plain-English walkthrough |
| `PROJECT_BRIEF.md` | Brief for the professor: objectives, tool split, stack |
| `DOWNLOADS.md` | Where each dataset came from and where it lives |
| `results/RESULTS.md`, `results/FIT_ANALYSIS.md`, `results/RESPIRATORY_EXPERIMENTS.md` | Generated result tables |
| `matlab/README_matlab.md` | The MATLAB side |

---

## Layout

```
config.py                 DSP parameters and paths for all three domains
run_experiment.py         end-to-end driver
train_model.py            trains + saves the model the UI loads
analyze_fit.py            over/underfitting diagnostics and learning curves
app_gui.py                desktop demo UI (4 tabs, auto-detect, record, PDF, batch)

src/preprocess.py         M2  resample, denoise, band-pass, normalise, frame
                              (+ a Sun AU reader, since GTZAN ships as .au)
src/features.py           M3  MFCC from scratch + spectral + temporal features
src/datasets.py           M1  GTZAN / PhysioNet2016 / ICBHI loaders, with
                              group ids so splits stay track/patient disjoint
src/synth.py                  synthetic test signals (pipeline checks; not used by shipped models)
src/pipeline.py               M1 -> M2 -> M3, with feature caching
src/train.py              M4  SVM, k-NN, Random Forest + metrics
src/ablation.py           M5  feature-group ablation study
src/visualize.py              all report figures
src/router.py                 signal-type router: music / heart / lung / unknown
src/rhythm.py                 heart rate, breathing rate, tempo (envelope autocorrelation)
src/report.py                 one-page PDF screening report

matlab/sa_preprocess.m    M2 in MATLAB
matlab/sa_features.m      M3 in MATLAB - same 53 features, same order
matlab/sa_classify.m      applies the model exported from Python
matlab/sa_verify.m        MATLAB vs Python feature cross-check
matlab/sa_demo.m          MATLAB end-to-end: read -> features -> classify -> plot
matlab/sa_round.m         Python-compatible rounding (keeps the two front ends identical)
matlab/PATCH_ClassifySignal.m   replaces the placeholder classifier in the GUI

scripts/export_for_matlab.py    model -> model_export/*.csv
scripts/make_demo_wavs.py       one example .wav per class
tests/test_dsp.py               unit checks on the DSP front end
tests/test_system.py            system checks incl. the GUI
```

---

## Command reference

```bash
python run_experiment.py --branch heart      # single branch
python run_experiment.py --no-cache          # force feature re-extraction
python run_experiment.py --limit 100         # cap recordings per class
python run_experiment.py --skip-figures      # numbers only, faster

python train_model.py --tune                 # grid-search hyper-parameters
python analyze_fit.py                        # fit diagnostics

python scripts/make_demo_wavs.py
python scripts/export_for_matlab.py
python scripts/export_for_matlab.py --verify data/demo_wavs/heart_abnormal_01.wav --domain heart
```

```matlab
addpath('matlab');
sa_verify('data/demo_wavs/heart_abnormal_01.wav', 'heart');
```

---

## Scope

The medical branches perform **acoustic signal classification / screening** and
are evaluated with signal-classification metrics. They are not a diagnostic
tool and no clinical claim is made.
