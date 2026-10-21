# Smart Audio Diagnostics — Implementation Brief

**Group 09 · Digital Signal Processing · ECE, MNNIT Allahabad**
Anushka Patil · Arin Saraf · Arjit Abhitaj · Arnav Raj
Submitted to: Dr. Rajeev Gupta

> One DSP front end, two classification branches. This document states what has
> been built, what each tool is used for and why, and what remains.

---

## 1. Status at a glance

| Synopsis objective | State | Evidence |
|---|---|---|
| **O1** Common pre-processing (resample, band-pass, denoise, normalise, framing) | **Done** | `src/preprocess.py`, `matlab/sa_preprocess.m` |
| **O2** Feature extraction (FFT, MFCC, centroid, bandwidth, ZCR, energy) | **Done** | `src/features.py`, `matlab/sa_features.m` — 53-D vector |
| **O3** Dataset assembly for both branches | **Done: GTZAN, PhysioNet 2016, ICBHI 2017 (all real)** | `src/datasets.py`, `DOWNLOADS.md` |
| **O4** Train / compare SVM, k-NN, Random Forest per branch | **Done** | `src/train.py`, `results/RESULTS.md` |
| **O5** Accuracy / precision / recall / F1 / confusion + feature ablation | **Done** | `src/ablation.py`, `results/figures/` |
| **O6** Modular pipeline, new domain addable without touching the front end | **Done** | the third branch was added by adding one `config.py` entry |

The full pipeline runs end-to-end on **real data for all three branches**
(21 Oct 2026): music 81.2 %, heart 83.3 % and lungs 50.4 % balanced accuracy on
held-out, group-disjoint test sets. Details in `STATUS_REPORT.md`; the
teammate guide is `TEAM_GUIDE.md`.

---

## 2. Tool split — what MATLAB does and what Python does

This was a deliberate decision, not a convenience.

```
                 +---------------- MATLAB -----------------+
   audio ------> |  M2 pre-processing | M3 features (MFCC  |
                 |                    | from scratch)      |
                 +------------+----------------------------+
                              |  53-element feature vector
                              |  (identical definition on both sides)
                 +------------v--------- Python -----------+
   audio ------> |  M2 pre-processing | M3 features        |
                 |  M4 SVM / k-NN / Random Forest          |
                 |  M5 metrics, confusion matrix, ablation |
                 +------------+----------------------------+
                              |  model exported as plain CSV
                              v
                 MATLAB GUI (SmartAudioDiagnostics.m) shows
                 waveform, FFT, spectrogram, MFCC + prediction
```

**MATLAB owns the signal processing and the demonstration.**
It is the natural tool for the course content — `butter`/`filtfilt`, `fft`,
`spectrogram`, `dct` — and the App Designer GUI our group already has. The
MATLAB app is the live artefact for the viva: load a `.wav`, see the waveform,
spectrum, spectrogram and MFCC map, get a prediction.

**Python owns the machine learning.**
`scikit-learn` gives SVM / k-NN / Random Forest, stratified cross-validation,
classification reports and confusion matrices as a few lines each, and it
scales to the ~3000 recordings in the real datasets without needing a toolbox
licence. Doing this in MATLAB would require the Statistics and Machine Learning
Toolbox and would still be more code for the same result.

**They are joined by a verification step, not by duplication.**
`sa_verify.m` computes the 53 features in MATLAB and compares them
element-by-element against the Python vector for the same file. Two independent
implementations of the same textbook equations agreeing to numerical precision
is the strongest evidence that the MFCC and spectral code is correct — and it
is a genuine DSP result to present, not glue code.

**The one thing we deliberately did not do:** call a library `mfcc()`.
MFCC is implemented from first principles on both sides — pre-emphasis →
framing → Hamming window → FFT → mel filter bank → log → DCT-II — because
Objective O2 *is* that computation. Everything else (filter design, FFT, DCT,
classifiers) uses standard library calls.

---

## 3. Full stack

| Layer | Tool | Used for |
|---|---|---|
| Language | Python 3.14, MATLAB R2023b+ | — |
| Numerics | NumPy 2.5 | arrays, FFT |
| DSP | SciPy 1.18 (`signal`, `fft`, `io.wavfile`) | Butterworth SOS design, zero-phase filtering, polyphase resampling, STFT, DCT-II |
| DSP (MATLAB) | Signal Processing Toolbox | `butter`, `filtfilt`, `spectrogram`, `resample`, `stft`, `dct` |
| ML | scikit-learn 1.9 | SVM (RBF), k-NN, Random Forest, `StandardScaler`, `StratifiedKFold`, metrics |
| Plots | Matplotlib 3.11 | all report figures |
| GUI | MATLAB App Designer | `SmartAudioDiagnostics.m` demo |

Deliberately **not** used: `librosa` (its `mfcc()` would replace the
deliverable), and any deep-learning framework — the synopsis specifies
classical classifiers, and the datasets are small enough that SVM and Random
Forest are the correct choice.

---

## 4. Module map (synopsis section 6 → files)

| Module | Synopsis function | Implementation |
|---|---|---|
| **M1** Data layer | labelled music + medical audio | `src/datasets.py` — GTZAN folder-per-genre, PhysioNet `REFERENCE.csv`, ICBHI per-cycle `.txt`; falls back to `src/synth.py` |
| **M2** Pre-processing | resample, band-pass, denoise, normalise, segment | `src/preprocess.py` / `matlab/sa_preprocess.m` |
| **M3** Feature extraction | FFT, 13 MFCC, centroid, bandwidth, ZCR, energy | `src/features.py` / `matlab/sa_features.m` |
| **M4a** Music branch | features → SVM / k-NN / RF → genre | `src/train.py` |
| **M4b** Medical branch | features → SVM / k-NN / RF → class | `src/train.py` (separate label space, never mixed) |
| **M5** Output / evaluation | accuracy, P/R/F1, confusion, ablation | `src/train.py`, `src/ablation.py`, `src/visualize.py` |

### The 53-element feature vector

| Group | Contents | Count |
|---|---|---|
| `mfcc` | 13 MFCC mean + 13 MFCC std + 13 Δ-MFCC mean | 39 |
| `spectral` | centroid, bandwidth, roll-off (85 %), flatness — mean + std each | 8 |
| `temporal` | ZCR, short-time energy, RMS — mean + std each | 6 |

Frame-level features are summarised by mean and standard deviation, so every
recording — whatever its length — yields one fixed-length vector. The group
name is carried inside each feature name (`mfcc::mfcc3_mean`), which is what
makes the ablation study a one-line mask instead of a rewrite.

### Per-domain DSP settings

Only the sample rate and the passband change between domains. The code path is identical.

| Branch | fs | Band-pass | Frame / hop | Mel filters | Classes |
|---|---|---|---|---|---|
| Music | 22 050 Hz | 30 Hz – 10 kHz | 25 / 10 ms | 40 | classical, jazz, rock, pop, hiphop |
| Heart | 2 000 Hz | 25 – 400 Hz | 50 / 25 ms | 26 | normal, abnormal |
| Respiratory | 4 000 Hz | 80 – 1800 Hz | 50 / 25 ms | 30 | normal, wheeze, crackle |

The heart passband comes from the physiology: S1/S2 energy sits below ~150 Hz
and murmurs extend to roughly 400 Hz, so 2 kHz sampling is sufficient and
discards noise the classifier would otherwise have to learn to ignore.

---

## 5. Datasets

| Branch | Dataset | Label format | Place in |
|---|---|---|---|
| Music | GTZAN Genre Collection | folder per genre, 100 × 30 s each (we use 5 genres) | `data/gtzan/genres_original/<genre>/*.wav` |
| Heart | PhysioNet/CinC Challenge 2016 | `REFERENCE.csv`: `1` = abnormal, `-1` = normal | `data/physionet2016/` |
| Respiratory | ICBHI 2017 Respiratory Sound DB | per-cycle `.txt`: `start end crackle wheeze` | `data/icbhi/audio_and_txt_files/` |

`src/datasets.py` already parses all three label formats. For ICBHI the
labelled unit is the **respiratory cycle**, not the file, so cycles are cut
using the annotation timestamps; cycles marked both crackle *and* wheeze are
excluded, since the synopsis defines a 3-class task.

---

## 6. How to run

```bash
pip install -r requirements.txt

python run_experiment.py                  # all three branches
python run_experiment.py --branch heart   # one branch
python run_experiment.py --no-cache       # force feature re-extraction

python scripts/make_demo_wavs.py          # one example .wav per class
python scripts/export_for_matlab.py       # model -> model_export/*.csv
```

Outputs: `results/RESULTS.md` (tables), `results/tables/results.json`, and
`results/figures/*.png` — waveform / FFT / spectrogram / MFCC per class, the
mel filter bank, PCA feature separation, confusion matrices, classifier
comparison and the ablation chart.

MATLAB, from the repo root:

```matlab
addpath('matlab');
sa_verify('data/demo_wavs/heart_abnormal_01.wav', 'heart');   % cross-toolchain check
```

---

## 7. What was left on 8 Sep — all done (21 Oct)

1. ✔ **Datasets downloaded and rerun.** PhysioNet 2016 and ICBHI 2017 are in use.
2. ✔ **Class imbalance:** `class_weight='balanced'`, model selection by balanced
   accuracy, sensitivity / specificity / ICBHI score reported.
3. ✔ **Patient-disjoint splitting:** ICBHI split on patient ID, PhysioNet on
   recording, GTZAN on track.
4. ✔ **Hyper-parameter search** run on all three branches (`train_model.py --tune`).
5. ✔ **MATLAB classification path:** `matlab/sa_demo.m` (end-to-end, segment-averaged).
6. ✔ **`sa_verify` run:** max relative error ≤ 6×10⁻⁸ (`results/tables/matlab_verify.csv`).

Added beyond the synopsis: signal-type router with unknown-sound rejection,
heart-rate/breathing-rate/tempo estimation, PDF screening report, live
recording, batch mode, 4-tab GUI. See `STATUS_REPORT.md` §2.

**Future work:** CNN on log-mel spectrograms; S1/S2 segmentation (HMM) for
murmur timing; event-level crackle detection; phone deployment.

---

## 8. Scope statement

The medical branches are **acoustic signal classification and screening
demonstrations**, evaluated with signal-classification metrics. They are not a
diagnostic tool, and no clinical claim is made.
