# Smart Audio Diagnostics — Team Guide

**Group 09 · Digital Signal Processing · ECE, MNNIT Allahabad**

This is the one document to read first. It explains **what we built, why it is
useful, how it works, what each of us must study, and what goes into the PPT
and the abstract.** Every number in it was produced by running the code in this
folder (see section 9 to reproduce).

> Work split (section 7): **You (team lead)** — pre-processing, rhythm analysis,
> the app and the live demo · **Arjit** — feature extraction (FFT, MFCC) and
> MATLAB · **Arin** — datasets, machine learning and evaluation.

---

## 1. The project in three sentences

1. We built **one digital-signal-processing (DSP) front end** — filtering,
   FFT, MFCC and spectral features — that turns *any* audio recording into the
   same 53 numbers.
2. Those numbers feed **separate classifiers** for three real problems:
   **music genre** (classical / jazz / rock / pop / hip-hop),
   **heart-sound screening** (normal / abnormal) and
   **lung-sound screening** (normal / wheeze / crackle).
3. A desktop app records or opens a file, **auto-detects what kind of sound it
   is**, shows every DSP stage visually, measures **heart rate / breathing rate
   / tempo**, classifies it and exports a **one-page PDF report**.

All three branches are trained and tested on **real, public, peer-reviewed
datasets** — GTZAN, PhysioNet/CinC 2016 and ICBHI 2017.

---

## 2. The use case (say this first in the PPT and the viva)

### Problem

- India has a **severe shortage of cardiologists and chest specialists**,
  worst in rural areas (Arin: find a current, citable figure before putting a
  number on the slide). Rheumatic and congenital heart disease, asthma, COPD
  and pneumonia are common and are often first noticed through the **stethoscope**.
- Auscultation (listening with a stethoscope) needs years of training. A nurse
  or ASHA worker at a Primary Health Centre can place a stethoscope but usually
  cannot judge whether a murmur or a wheeze is abnormal.
- Digital stethoscopes are increasingly affordable and record a `.wav` file,
  but a recording alone does not tell the health worker *what to do next*.

### Our solution

```
 health worker records          our app (laptop)                  doctor / tele-consult
 heart or lung sound   ───▶  auto-detect type → filter → FFT/MFCC ───▶  PDF report:
 (digital stethoscope)       → classifier → heart/breathing rate        "ABNORMAL – refer",
                                                                        72 bpm, spectrogram
```

- **Screening, not diagnosis:** the output is *"normal"* or *"abnormal – refer
  to a doctor"* plus heart rate / breathing rate. It is a triage aid that helps
  decide **who must see a specialist first**. We never claim a clinical
  diagnosis, and the app says so on every medical result.
- The **PDF report** is designed for tele-medicine (e.g. e-Sanjeevani): the
  doctor sees the waveform, spectrogram, rate and confidence without needing
  the audio player.

### Why music is in the same project

The music branch proves the **front end is domain-agnostic**: the *same* code
classifies songs and stethoscope sounds — only the sampling rate and the
filter passband change (section 4.3). Music genre tagging is itself a real
application (auto-tagging a music library or a streaming catalogue — our
**batch-folder mode** does exactly that). Adding a new domain (e.g. machine
fault sounds, bird calls) is one entry in `config.py`.

### Who uses it

| User | What they do | What they get |
|---|---|---|
| Nurse / ASHA worker at a PHC | Records heart and lung sounds | Normal / refer decision, heart & breathing rate, PDF for the doctor |
| Doctor (remote) | Opens the PDF | Waveform, spectrogram, confidence, rate |
| Medical student | Loads example recordings | Sees *why* a murmur or wheeze looks different (spectrogram, MFCC) |
| Music library / app | Points batch mode at a folder | A CSV of genre tags |

---

## 3. What is in the app (the demo)

Run `python app_gui.py`. Top bar: **Signal type** (Auto-detect / Music / Heart /
Respiratory) · **Open audio** (wav, mp3, flac, m4a, ogg, au) · **● Record 6 s**
(laptop microphone) · **Demo signal** · **▶ Play** · **Classify** · **Export
PDF report** · **Batch folder…**

| Tab | What it shows | DSP concept it demonstrates |
|---|---|---|
| **Signal analysis** | Waveform, FFT spectrum, spectrogram, MFCC map | Time vs frequency domain, STFT, cepstrum |
| **DSP pipeline** | Raw vs filtered waveform; raw vs filtered spectrum (Welch PSD); Butterworth filter response; mel filter bank | Filtering, noise reduction, filter design, mel scale |
| **Rhythm & timeline** | Envelope with detected beats/breaths; autocorrelation with the chosen period; per-segment class probabilities | Envelope detection, autocorrelation, periodicity |
| **Model performance** | Confusion matrix and classifier comparison of the loaded branch | Evaluation metrics |

Right-hand panel: auto-detected signal type, prediction with confidence bars,
**ground truth ✓/✗** for demo clips, heart rate / breathing rate / tempo, and
the DSP feature values.

**Features added beyond the synopsis** (point these out — they make the project
stand out):

1. **Auto-detect (signal-type router)** — an extra classifier decides whether
   the input is music, a heart sound, a lung sound **or none of these**
   ("unknown sound", trained on 2 000 everyday sounds from ESC-50) and routes
   it to the right branch. 94.6 % accurate; rejected 61/70 Windows system
   sounds it had never heard.
2. **Heart-rate / breathing-rate / tempo estimation** — the *same*
   envelope + autocorrelation algorithm gives bpm for a heart, breaths/min for
   lungs and beats/min for a song. Pure DSP, no machine learning.
3. **Whole-recording classification** — every 5 s segment is classified and the
   probabilities are averaged; the timeline shows how the decision evolves.
4. **Live microphone recording.**
5. **One-page PDF screening report.**
6. **Batch mode** — classifies a whole folder, writes a CSV, reports accuracy if
   the true labels are in the file names.
7. **Honest warnings** — unknown sounds, uncertain routing and borderline
   results (e.g. 55 % abnormal) are flagged in the status bar instead of being
   reported as confident answers.
8. **MATLAB mirror** — the whole DSP front end is also written in MATLAB and
   verified against Python to numerical precision.

---

## 4. How it works

### 4.1 The pipeline

```
 audio file ─▶ M1 read ─▶ M2 pre-process ─────────────▶ M3 features ──────────────▶ M4 classify ─▶ M5 evaluate
 (.wav .mp3    (ffmpeg,   1 mono                         frame 25–50 ms, Hamming      SVM (RBF)       accuracy, F1,
  .au …)       AU reader) 2 resample (polyphase)         FFT → power spectrum         k-NN (k=5)      sensitivity,
                          3 spectral-subtraction denoise  mel filter bank → log → DCT Random Forest   specificity,
                          4 Butterworth band-pass,        = 13 MFCC + 13 ΔMFCC                        confusion matrix,
                            order 4, zero-phase           centroid, bandwidth,                        ablation
                          5 peak-normalise                roll-off, flatness,
                          6 fixed-length clip             ZCR, energy, RMS
                                                          → mean & std → 53 numbers
```

Side paths: **router** (which branch?) and **rhythm** (what rate?) reuse M2/M3.

### 4.2 The 53-number feature vector

| Group | Contents | Count |
|---|---|---|
| MFCC | 13 MFCC mean + 13 MFCC std + 13 ΔMFCC mean | 39 |
| Spectral | centroid, bandwidth, roll-off (85 %), flatness — mean & std | 8 |
| Temporal | zero-crossing rate, short-time energy, RMS — mean & std | 6 |

MFCC is **written from first principles** (pre-emphasis → framing → Hamming →
FFT → mel filter bank → log → DCT-II). We deliberately did *not* call a
library `mfcc()` function, because that computation *is* the DSP deliverable.

### 4.3 Per-branch settings (only these change)

| Branch | Sampling rate | Band-pass | Frame / hop | Mel filters | Why this band |
|---|---|---|---|---|---|
| Music | 22 050 Hz | 30 Hz – 10 kHz | 25 / 10 ms | 40 | Full musical range |
| Heart | 2 000 Hz | 25 – 400 Hz | 50 / 25 ms | 26 | S1/S2 < 150 Hz, murmurs up to ~400 Hz |
| Lung | 4 000 Hz | 80 – 1 800 Hz | 50 / 25 ms | 30 | Breath noise, wheezes (100–1000 Hz), crackles |

### 4.4 The rate estimator (heart rate / breathing rate / tempo)

```
signal → envelope → autocorrelation → strongest peak in an allowed lag window → rate = 60 / lag
```

- Heart & lung: **homomorphic envelope** = exp(low-pass(log|x|)) — low-pass at
  8 Hz (heart) or 1 Hz (breathing).
- Music: **spectral-flux onset envelope** — the frame-to-frame rise in log
  power, which spikes at every drum hit / note onset.
- Search windows: heart 40–160 bpm, breathing 8–40 /min, tempo 60–180 bpm.
- Tested on signals with a known rate: error < 0.3 bpm from 45 to 150 bpm.

### 4.5 The one design decision to defend: group-wise splitting

A 30 s song is cut into six 5 s clips. If clip 1 is in training and clip 2 in
testing, the model is tested on a song it has already heard → accuracy is
inflated by 10–15 points. **We split by group**: by track (music), by recording
(heart) and by **patient** (lungs). No song, recording or patient is ever on
both sides of the split (`GroupShuffleSplit`, `StratifiedGroupKFold`).

### 4.6 Class imbalance

PhysioNet is ~80 % normal and ICBHI ~55 % normal. A model that always says
"normal" would score 80 % accuracy and be useless. So we use
`class_weight="balanced"`, choose the best classifier by **balanced accuracy**,
and report **sensitivity** (abnormal caught) and **specificity** (normal
cleared) — the ICBHI challenge score is their average.

---

## 5. Results (all on held-out test data never seen in training)

### 5.1 Headline (shipped, tuned models)

| Branch | Dataset (real) | Test clips | Classes | Best model | Accuracy | **Balanced acc.** | Other key metric |
|---|---|---|---|---|---|---|---|
| **Music** | GTZAN | 747 clips / 125 songs | 5 (chance 20 %) | Random Forest | 82.2 % | **81.2 %** | per song: 83.2 % |
| **Heart** | PhysioNet 2016 | 2 017 clips / 810 recordings | 2 (chance 50 %) | Random Forest | 85.7 % | **83.3 %** | sensitivity 79.0 %, specificity 87.6 % |
| **Lungs** | ICBHI 2017 | 1 737 cycles / 32 patients | 3 (chance 33 %) | SVM (RBF) | 50.5 % | **50.4 %** | ICBHI score 51.2 % (Se 57.3 %, Sp 45.1 %) |
| **Router** | all three + ESC-50 | 691 clips | 4 (music/heart/lung/other) | Random Forest | **94.6 %** | — | rejects 61/70 never-seen Windows sounds as "unknown" |

"Balanced accuracy" = average of per-class recall. It is the honest number when
classes are imbalanced (heart is 80 % normal).

### 5.2 Classifier comparison (default hyper-parameters, `run_experiment.py`)

| Branch | SVM (RBF) | k-NN (k=5) | Random Forest |
|---|---|---|---|
| Music — test acc / balanced | 0.803 / 0.799 | 0.735 / 0.721 | **0.818 / 0.807** |
| Heart — test acc / balanced | 0.853 / 0.795 | 0.842 / 0.725 | **0.866 / 0.825** |
| Lungs — test acc / balanced | 0.525 / 0.426 | 0.533 / 0.398 | 0.537 / 0.426 |

Grid-search tuning (`train_model.py --tune`) then raised the lung SVM from
0.426 to **0.504** balanced accuracy (C 10 → 1, γ → 0.001 — a smoother
boundary that stopped it over-fitting: train 0.95 → 0.57, test up).

### 5.3 Feature ablation (5-fold CV accuracy, change vs all 53 features)

| Configuration | Music | Heart | Lungs |
|---|---|---|---|
| All 53 features | 0.790 | 0.873 | 0.538 |
| without MFCC | **−0.127** | −0.016 | **−0.073** |
| without spectral | −0.001 | +0.002 | −0.003 |
| without temporal | −0.019 | −0.006 | −0.021 |
| MFCC only | −0.020 | −0.012 | −0.030 |
| spectral only | −0.189 | −0.037 | −0.121 |
| temporal only | −0.218 | −0.036 | −0.101 |

**MFCC carries every branch** — removing it costs the most, and MFCC alone is
within 1–3 points of the full set. This is the empirical justification for
making MFCC the core DSP deliverable.

### 5.4 Over/under-fitting (`analyze_fit.py`, shipped models)

| Branch | Train | 5-fold CV | Test | Verdict |
|---|---|---|---|---|
| Music | 1.000 | 0.772 | 0.822 | generalises (RF always fits train 100 %; CV ≈ test) |
| Heart | 0.958 | 0.868 | 0.857 | well fitted |
| Lungs | 0.567 | 0.491 | 0.505 | not over-fitted; limited by the features (see 5.6) |

### 5.5 Engineering checks

| Check | Result |
|---|---|
| MATLAB vs Python, all 53 features, 3 branches, with and without denoising | max relative error **≤ 6 × 10⁻⁸** |
| Heart-rate estimator on signals with known rate (45–150 bpm) | error < 0.3 bpm |
| Demo clips (held-out) correctly classified by the app | music 19/20, heart 6/8, lungs 7/12 |
| MATLAB classifier (k-NN on exported model) on the same clips | 32/40 |
| Automated tests | `tests/test_dsp.py` 19/19, `tests/test_system.py` 52/52 (incl. GUI, mic recording, PDF) |

Two real bugs were found and fixed by the MATLAB cross-check: Python's
`round(220.5)` = 220 (banker's rounding) vs MATLAB's 221, and MATLAB's `stft`
padding differing from SciPy's. Before the fix the MATLAB classifier got
12/40 demo clips right; after it, 32/40.

### 5.6 Comparison with published work (verify the citations before putting them on a slide)

| Dataset | Published | Ours | Comment |
|---|---|---|---|
| GTZAN | Tzanetakis & Cook 2002: 61 % on **10** genres | 82 % on 5 genres | not directly comparable (fewer genres) — our split is track-disjoint, many papers' are not |
| PhysioNet 2016 | Top challenge entries ≈ 85–86 % (mean of Se and Sp) on the hidden test set | 83.3 % balanced (per clip), 81.1 % (per recording) | close to the top entries with only 53 features and no deep learning |
| ICBHI 2017 | Classical-feature methods on patient-disjoint splits ≈ 40–50 % score; deep CNNs ≈ 55–65 % | 51.2 % score | in line with classical methods; crackles last < 20 ms and get averaged away in clip-level features (`results/RESPIRATORY_EXPERIMENTS.md`) |

We also tried four alternative pre-processing settings for the lungs (no
denoising, 8 kHz and a wider band, 5 s clips). None improved beyond noise. This is
a documented negative result, and an honest one to present.

---

## 6. Datasets

| Branch | Dataset | Size used | Labels | Source |
|---|---|---|---|---|
| Music | **GTZAN Genre Collection** (Tzanetakis & Cook, 2002) | 500 tracks × 30 s (5 of 10 genres) → 2 995 clips | folder per genre | kaggle.com/datasets/carlthome/gtzan-genre-collection |
| Heart | **PhysioNet/CinC Challenge 2016** (Liu et al., 2016) | 3 240 recordings, 6 hospitals/devices → 8 147 clips | `REFERENCE.csv`: 1 abnormal, −1 normal | physionet.org/content/challenge-2016 |
| Lungs | **ICBHI 2017 Respiratory Sound Database** (Rocha et al., 2019) | 920 recordings, 126 patients → 6 392 respiratory cycles | per-cycle `.txt`: start, end, crackle, wheeze | bhichallenge.med.auth.gr |
| Router "other" | **ESC-50** environmental sounds (Piczak, 2015) | 800 of 2 000 clips (16 per category × 50) | category per clip | github.com/karolpiczak/ESC-50 |

---

## 7. Who studies what

Each person owns one part end-to-end: **understand it, explain it in the viva,
and make its PPT slides.** Everyone must be able to give the 1-minute overview
in section 1 and the use case in section 2.

### 7.1 You (team lead) — Pre-processing, rhythm analysis, the app and the demo

**Owns:** M1 reading, M2 pre-processing, the rate estimator, the router,
`app_gui.py`, the live demo, overall story.

| Topic to study | What you must be able to explain | Where in our code |
|---|---|---|
| Sampling, Nyquist theorem, aliasing | Why heart uses 2 kHz and music 22.05 kHz | `config.py` |
| Resampling (polyphase, up/down by L/M) | `resample_poly`, anti-alias filter inside it | `src/preprocess.py: resample_to` |
| IIR filters — Butterworth | Maximally flat passband, order 4, −3 dB cut-offs | `bandpass` |
| Zero-phase filtering | `filtfilt` runs forward + backward → no phase distortion, squared magnitude | `bandpass` |
| SOS (second-order sections) | Why SOS is numerically stable vs one high-order polynomial | `butter(..., output="sos")` |
| Spectral subtraction denoising | Estimate noise from quietest 10 % frames, subtract, spectral floor | `denoise` |
| Normalisation, fixed-length clips | Peak normalisation, why each clip must be the same length | `normalize`, `fix_length` |
| Envelope detection | Rectify + low-pass; homomorphic envelope (log domain) | `src/rhythm.py` |
| Autocorrelation | Periodic signal → peaks at multiples of the period; picking the right peak | `src/rhythm.py: autocorr, estimate_rate` |
| The router | Same features at 4 kHz; Random Forest picks music / heart / lung / *other*; "other" learned from ESC-50 everyday sounds; plus a nearest-neighbour novelty score | `src/router.py` |
| Python app basics | Tkinter, Matplotlib embedding, how the buttons call the pipeline | `app_gui.py` |

**Resources**
- Oppenheim & Schafer, *Discrete-Time Signal Processing* — ch. 4 (sampling, resampling), ch. 7 (filter design).
- SciPy docs: `scipy.signal.butter`, `sosfiltfilt`, `resample_poly`.
- Schmidt et al. 2010, "Segmentation of heart sound recordings by a duration-dependent HMM" (homomorphic envelope idea).
- Our `DSP_EXPLAINED.md` Parts 2–3 and `HOW_IT_WORKS.md`.

**PPT slides you make:** Title · Problem & use case · System architecture ·
Pre-processing (with the *DSP pipeline* tab screenshot) · Rate estimation
(with the *Rhythm* tab screenshot) · App walkthrough / live demo · Conclusion.

**Viva questions you answer**
- *Why band-pass 25–400 Hz for heart sounds?* S1/S2 energy is below ~150 Hz,
  murmurs reach ~400 Hz; everything else is noise for this task.
- *Why zero-phase filtering?* A normal IIR filter shifts each frequency by a
  different delay and smears the sharp S1/S2 and crackle transients; filtering
  forward and backward cancels the phase.
- *How do you get heart rate without ML?* Envelope → autocorrelation → the lag
  of the strongest peak is one beat period → bpm = 60 / period.
- *What if the file is not music/heart/lung?* The router has a fourth class,
  "other", trained on 2 000 environmental sounds (ESC-50). The app shows
  UNKNOWN SOUND in orange and says the label is not meaningful. Try it live
  with speech or clapping into the mic.

### 7.2 Arjit — Feature extraction (FFT, MFCC, spectral features) and MATLAB

**Owns:** M3 feature extraction, the MATLAB mirror, the MATLAB-vs-Python
verification.

| Topic to study | What you must be able to explain | Where in our code |
|---|---|---|
| DFT and FFT | Definition, O(N log N), frequency resolution = fs / N | `src/features.py: stft_power` |
| Framing and windowing | Why 25–50 ms frames; Hamming window reduces spectral leakage | `frame_signal` |
| STFT and spectrogram | Time–frequency trade-off | Spectrogram plot |
| Mel scale | mel = 2595·log10(1 + f/700); why human hearing is log-like | `hz_to_mel`, `mel_filterbank` |
| Mel filter bank | Triangular filters, wider at high frequency | `mel_filterbank` |
| Log + DCT = cepstrum | Log compresses dynamic range; DCT decorrelates → MFCC | `mfcc` |
| Pre-emphasis | y[n] = x[n] − 0.97 x[n−1] boosts high frequencies | `pre_emphasis` |
| Delta MFCC | Regression slope over ±2 frames (how fast MFCC changes) | `delta` |
| Spectral centroid, bandwidth, roll-off, flatness | "Brightness", spread, where 85 % of energy lies, noise-like vs tonal | `spectral_*` |
| ZCR, short-time energy, RMS | Time-domain noisiness and loudness | `zero_crossing_rate`, `short_time_energy` |
| MATLAB implementation | `fft`, `dct`, `butter`, `filtfilt`, `spectrogram` | `matlab/sa_features.m`, `sa_preprocess.m` |
| Cross-verification | Same file, two languages, relative error per feature | `matlab/sa_verify.m` |

**Resources**
- "Mel Frequency Cepstral Coefficient (MFCC) tutorial" — practicalcryptography.com (best single explanation).
- 3Blue1Brown, "But what is the Fourier Transform?" (YouTube) for intuition.
- Oppenheim & Schafer ch. 8–10 (DFT, FFT, spectral analysis with the DFT).
- MATLAB docs: `spectrogram`, `dct`, `designfilt`.
- Our `DSP_EXPLAINED.md` Part 4 (every feature with its formula).

**PPT slides you make:** Framing & windowing · FFT & spectrogram · MFCC
step-by-step (one slide with the 7-step diagram) · Spectral & temporal
features · MATLAB implementation + verification table · Feature ablation
(why MFCC matters).

**Viva questions you answer**
- *Why Hamming and not rectangular?* Rectangular has high side lobes → energy
  leaks into neighbouring bins; Hamming's first side lobe is −43 dB.
- *Why log before DCT?* Turns the multiplicative source × vocal-tract (filter)
  model into an addition so the DCT can separate them; also matches loudness
  perception.
- *Why only 13 MFCCs?* The low coefficients describe the spectral envelope
  (timbre); higher ones carry fine pitch detail that is not needed.
- *How do you know your MFCC is correct?* Two independent implementations
  (Python and MATLAB) agree to within 6 × 10⁻⁸ relative error on all 53 features.

### 7.3 Arin — Datasets, machine learning and evaluation

**Owns:** M1 datasets, M4 classifiers, M5 evaluation, the results slides, the
abstract.

| Topic to study | What you must be able to explain | Where in our code |
|---|---|---|
| The three datasets | Size, labels, how they were recorded, class balance | `src/datasets.py`, section 6 |
| Feature standardisation | z-score; why SVM and k-NN need it | `StandardScaler` in `src/train.py` |
| SVM with RBF kernel | Maximum-margin, kernel trick, C and γ | `make_models` |
| k-Nearest Neighbours | Distance vote, choice of k | `make_models` |
| Random Forest | Bagging + random feature subsets, majority vote | `make_models` |
| Train/test split, cross-validation | 75/25 hold-out, 5-fold CV, **group-wise** (no leakage) | `split`, `cv_splitter` |
| Class imbalance | `class_weight="balanced"`, balanced accuracy | `src/train.py` |
| Metrics | Accuracy, precision, recall, F1, **sensitivity, specificity**, confusion matrix | `evaluate_branch`, `screening_scores` |
| Hyper-parameter tuning | Grid search over C, γ, k, depth | `train_model.py --tune` |
| Over/under-fitting | Train vs CV vs test, learning curves | `analyze_fit.py` |
| Feature ablation | Remove one feature group, measure the drop | `src/ablation.py` |

**Resources**
- scikit-learn user guide: "Support Vector Machines", "Nearest Neighbors", "Ensemble methods", "Cross-validation", "Model evaluation".
- StatQuest (YouTube): SVM, Random Forest, ROC & sensitivity/specificity.
- Dataset papers: Tzanetakis & Cook 2002 (GTZAN); Liu et al. 2016, *Physiol. Meas.* (PhysioNet 2016); Rocha et al. 2019, *Physiol. Meas.* (ICBHI).
- Our `STATUS_REPORT.md` and `results/FIT_ANALYSIS.md`.

**PPT slides you make:** Datasets · Classifiers used · Evaluation method
(group split, imbalance) · Results tables · Confusion matrices · Over/under-fitting
& learning curves · Comparison with literature · Future scope.
**Also writes the abstract** (draft in section 8).

**Viva questions you answer**
- *Why not deep learning?* Datasets are a few thousand clips; classical
  classifiers on good DSP features train in seconds, are explainable, and the
  synopsis asks for SVM / k-NN / RF. A CNN is listed as future work.
- *Your train accuracy is 100 % — isn't that overfitting?* No: Random Forest and
  RBF-SVM fit training data perfectly by construction. What matters is that
  cross-validation and the held-out test agree (they do, within ~0.05).
- *Why is heart accuracy higher than balanced accuracy?* The data is 80 %
  normal; balanced accuracy averages the per-class recall, so it is the honest
  number.
- *How does this compare with published work?* See section 5 — comparable to
  classical-feature baselines on the same datasets.

---

## 8. PPT outline and abstract draft

### 8.1 Slide-by-slide outline (~16 slides, 12–15 min)

| # | Slide | Owner | Figure to use (in `results/figures/`) |
|---|---|---|---|
| 1 | Title, team, guide | You | — |
| 2 | Problem: auscultation gap in India + music tagging | You | — |
| 3 | Our solution & use case (the flow diagram in section 2) | You | `ui_heart.png` |
| 4 | System architecture (M1–M5 + router + rhythm) | You | — (draw from section 4.1) |
| 5 | Datasets | Arin | — (table from section 6) |
| 6 | Pre-processing: resample, denoise, Butterworth | You | `ui_dsp.png` |
| 7 | Framing, windowing, FFT, spectrogram | Arjit | `overview_heart_abnormal.png` |
| 8 | MFCC step by step + mel filter bank | Arjit | `melbank_music.png` |
| 9 | Spectral & temporal features; 53-D vector | Arjit | `pca_music.png` |
| 10 | Classifiers & evaluation method (group split, imbalance) | Arin | — |
| 11 | Results: three branches | Arin | `classifier_comparison.png` |
| 12 | Confusion matrices | Arin | `confusion_heart.png`, `confusion_respiratory.png`, `confusion_music.png` |
| 13 | Ablation & over/under-fitting | Arin + Arjit | `ablation.png`, `learning_curves.png` |
| 14 | Rate estimation + auto-detect + PDF report | You | `ui_rhythm.png` |
| 15 | MATLAB vs Python verification | Arjit | `matlab_demo.png` |
| 16 | Conclusion, limitations, future scope | Arin | — |
| — | **Live demo** (4 min, script in `DEMO_SCRIPT.md`) | You | — |

**Future scope (slide 16):** CNN on log-mel spectrograms; murmur timing via S1/S2
segmentation (HMM); deploy on a phone with a ₹500 stethoscope adaptor; more
lung classes (pneumonia, COPD — the ICBHI diagnosis file); clinical validation
with a hospital.

**Limitations (be upfront, examiners like this):** screening not diagnosis;
datasets recorded in Europe/US/Brazil, not Indian clinics; lung sounds are hard —
published classical methods on ICBHI also stay in the 50–60 % score range;
music uses only 5 Western genres.

### 8.2 Abstract (draft, about 250 words; Arin to polish)

Auscultation, listening to heart and lung sounds with a stethoscope, is the
first screening step for many cardiac and respiratory diseases, but it needs a
trained ear that is scarce in primary health care. This project presents
*Smart Audio Diagnostics*, a unified digital-signal-processing and machine-learning
framework that classifies both medical and music audio with a single,
domain-agnostic front end. Each recording is resampled, denoised by spectral
subtraction, band-pass filtered with a zero-phase Butterworth filter and
normalised. A 53-dimensional feature vector is then computed per clip: 13
mel-frequency cepstral coefficients (MFCC) implemented from first principles,
their deltas, and spectral (centroid, bandwidth, roll-off, flatness) and
temporal (zero-crossing rate, energy) descriptors. Only the sampling rate and
pass-band change between domains. SVM, k-NN and Random Forest classifiers are
compared under patient/track-disjoint splits with class-balanced training. On
real public datasets the system reaches 83.3 % balanced accuracy for
normal/abnormal heart sounds (PhysioNet 2016; sensitivity 79 %, specificity
88 %), 81.2 % for five-genre music classification (GTZAN) and a 51.2 % ICBHI
score for normal/wheeze/crackle lung sounds (ICBHI 2017), in line with
classical-feature baselines. A feature-ablation study shows that MFCCs carry
most of the discriminative information in every domain. The desktop
application adds automatic signal-type detection (94.6 %, including rejection
of unknown sounds), heart-rate, breathing-rate and tempo estimation by envelope
autocorrelation, live recording and a one-page PDF screening report. The DSP
front end is mirrored in MATLAB and agrees with Python to 10⁻⁷. The system is
a screening aid, not a diagnostic device.

**Keywords:** digital signal processing, MFCC, phonocardiogram, respiratory
sounds, music genre classification, SVM, Random Forest, screening.

---

## 9. Running everything (for whoever operates the laptop)

```bash
cd C:\dsp
pip install -r requirements.txt

python app_gui.py                       # the demo app

python tests/test_dsp.py                # DSP unit checks
python tests/test_system.py             # loaders, models, router, report, GUI
python train_model.py --tune            # retrain all three branches (+ grid search)
python -m src.router                    # retrain the auto-detect router
python run_experiment.py                # full metrics, ablation, all figures
python analyze_fit.py                   # over/under-fitting + learning curves
python scripts/make_demo_wavs.py        # demo clips from the held-out test split
python scripts/classify_file.py song.mp3          # auto-detects the branch
```

MATLAB (from the `C:\dsp` folder):

```matlab
addpath('matlab');
sa_demo('data/demo_wavs/heart_abnormal_01.wav', 'heart');
sa_verify('data/demo_wavs/heart_abnormal_01.wav', 'heart');
```

---

## 10. Glossary

| Term | Meaning |
|---|---|
| S1 / S2 | The two normal heart sounds ("lub" – "dub"): valves closing |
| Murmur | Extra whooshing sound between S1 and S2 from turbulent blood flow — the main "abnormal" cue |
| Wheeze | Continuous musical (tonal) lung sound, 100–1000 Hz, > 100 ms — narrowed airways (asthma, COPD) |
| Crackle | Short explosive clicking lung sound (< 20 ms) — fluid or reopening airways (pneumonia, fibrosis) |
| Respiratory cycle | One inhale + exhale; the labelled unit in ICBHI |
| MFCC | Mel-frequency cepstral coefficients — compact description of the spectral envelope (timbre) |
| Sensitivity | Of all truly abnormal cases, the fraction we flagged (recall of abnormal) |
| Specificity | Of all truly normal cases, the fraction we cleared |
| Balanced accuracy | Mean of the per-class recalls — fair on imbalanced data |
| Group split | Train/test split where no track / recording / patient appears on both sides |
| Router | Our classifier that decides *which branch* a recording belongs to, or that it is an unknown sound |
