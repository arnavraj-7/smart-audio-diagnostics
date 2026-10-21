# How It Works — Plain English

**Smart Audio Diagnostics · Group 09 · DSP · ECE, MNNIT Allahabad**

No jargon. Read this before the viva so you can explain the project to anyone.

---

## 1. The one-line version

Sound goes in → the computer turns it into **53 numbers** → those numbers get
compared against thousands of examples it has already seen → it says which
category the sound belongs to.

Everything else in this document is detail about how you get those 53 numbers.

---

## 2. The journey of one audio file

Think of it as a factory line with five stations.

### Station 1 — Reading the file

A `.wav` or `.au` file is just a long list of numbers: the position of a
speaker cone, measured 22,050 times every second. A 5-second clip is 110,250
numbers. Nothing meaningful yet — it is raw material.

### Station 2 — Cleaning (pre-processing)

Four things happen, in order:

| Step | What it does | Why |
|---|---|---|
| **Resample** | Force every file to the same speed | So a phone recording and a studio recording are comparable |
| **Band-pass filter** | Throw away frequencies we don't care about | Heart sounds live between 25–400 Hz. Everything above that is noise the model would have to learn to ignore |
| **Denoise** | Find the quietest moments, assume that is background hiss, subtract it | Removes the recording room, keeps the signal |
| **Normalise** | Scale so the loudest peak is exactly 1.0 | A quiet recording and a loud one now look the same |

### Station 3 — Chopping into frames

The clip is cut into slices about 25 milliseconds long, slightly overlapping.

Sound changes constantly, so describing the whole clip with one set of numbers
would blur everything together. Instead we describe each tiny slice separately.

### Station 4 — Measuring each slice (this is the actual DSP)

For every slice we measure:

| Measurement | In plain words |
|---|---|
| Spectral centroid | How bright or sharp does it sound? |
| Spectral bandwidth | How spread out are the frequencies? |
| Spectral roll-off | Below which pitch is most of the sound? |
| Spectral flatness | Is it a clear tone, or noisy hiss? |
| Zero-crossing rate | How often the wave flips sign — rough noisiness indicator |
| Short-time energy | How loud |
| **13 MFCC numbers** | The "fingerprint" of the sound — see section 5 |

Then we take the **average** and the **variation** of each measurement across
all the slices. That collapses everything into **one list of 53 numbers per
clip**.

This list is called the **feature vector**, and it is the only thing the
classifier ever sees. The audio is gone by this point.

### Station 5 — Deciding

The 53 numbers go to a trained model, which answers *classical / jazz / rock /
pop / hiphop*, or *normal / abnormal*, or *normal / wheeze / crackle* — with a
confidence percentage for each.

---

## 3. Where MATLAB fits — read this part carefully

**Honest answer: audio does NOT flow through MATLAB and then into the Python
model.** They are two separate implementations of Stations 2–4 that produce the
*same* 53 numbers.

```
              MATLAB path                      Python path

  audio ──► sa_preprocess.m          audio ──► src/preprocess.py
            sa_features.m                      src/features.py
                  │                                  │
                  └────────► 53 numbers ◄────────────┘
                                  │
                        should be identical
                        sa_verify.m checks this
```

### Why build it twice on purpose?

**Because that is the proof the DSP code is correct.**

`sa_verify.m` computes the 53 numbers in MATLAB, compares them against the
Python numbers for the same file, and prints the difference for every one. If
two independently written programs agree to numerical precision, the maths is
right.

That is a real DSP result to present — not plumbing, not busywork.

### MATLAB's three jobs

1. **The DSP front end** — the course tool, using `butter`, `filtfilt`, `fft`,
   `spectrogram`, `dct`
2. **The verification** — `sa_verify.m`, above
3. **A GUI** — the App Designer app

### Python's one job

**The machine learning.** scikit-learn handles thousands of recordings, gives
SVM / k-NN / Random Forest and all the metrics in a few lines, and needs no
MATLAB toolbox licence.

### How they connect when you DO want MATLAB to classify

`scripts/export_for_matlab.py` writes the trained model out as plain CSV files
into `model_export/`. `sa_classify.m` reads those CSVs and does the comparison
itself.

So MATLAB can produce a prediction without re-implementing the classifier — it
just needs the exported numbers.

---

## 4. Where is the UI?

**There are two, and only one is live.**

| | Python UI | MATLAB UI |
|---|---|---|
| File | `app_gui.py` | `SmartAudioDiagnostics.m` |
| Status | **Working — this is what runs** | Built, but still placeholder logic |
| Classifier behind it | Real trained model | Hard-coded thresholds |

The MATLAB app currently **fakes** the classification. Its code says roughly
*"if the sound is dark, call it acoustic; if bright, call it electronic"* —
fixed thresholds, no learning involved.

`matlab/PATCH_ClassifySignal.m` is the written replacement that swaps in the
real trained model. It has not been pasted into the app yet.

> **For the demo, use the Python UI.** Run `python app_gui.py`. It is the one
> with the real model behind it.

---

## 5. The four graphs — what they actually mean

All four are drawn by the Python UI and computed **live** from the file you
loaded. They are not images loaded from disk — they are calculated the moment
you open a file.

### Top-left — Time-domain waveform

**"How loud, moment to moment."**

- Left to right = time
- Up and down = how hard the speaker pushes
- Fat sections = loud, thin sections = quiet

On a heart sound you can literally count the beats — each pair of spikes is one
*lub-dub*. On music you can see the song building up.

> This is the raw sound. It tells you almost nothing about *what* the sound is.

### Top-right — FFT magnitude spectrum

**"Which pitches are in this clip, overall."**

- Left to right = pitch, low to high
- Height = how much of that pitch is present

Think of it as: if you played the whole clip through a piano-detector, this is
every note it found and how loud each one was.

- A pure whistle → one sharp spike
- Static or hiss → flat across everything
- Music → a bumpy landscape

> Its weakness: it mashes the whole clip together. A song that goes low-then-high
> looks identical to one that goes high-then-low.

### Bottom-left — Spectrogram

**"Which pitches, at which moment."** This fixes the FFT's weakness.

- Left to right = time
- Bottom to top = pitch, low to high
- Brightness = how strong

It is a heat map of sound. Bright yellow and white mean lots of energy there.

> **This is your best demo graph** — you can point at the evidence.
>
> On the wheeze file there is a bright **horizontal line** at about 300 Hz: that
> is the wheeze, a single note held through the breath.
>
> On the heart file you see **vertical stripes** — each stripe is one heartbeat.

### Bottom-right — MFCC

**"The computer's fingerprint of the sound."**

This one is not meant to look like anything to a human. It is 13 numbers per
slice that summarise the *shape* of the spectrum — what people loosely call
timbre, or the "colour" of a sound.

Why it matters: a violin and a trumpet playing the **same note** have the same
pitch but sound completely different. MFCC captures that difference in 13
numbers instead of thousands.

**How it is built** — this is the main DSP deliverable, written from scratch
rather than calling a library function:

1. **Pre-emphasis** — boost the high frequencies slightly
2. **Framing** — cut into slices
3. **Windowing** — taper each slice's edges so the maths does not glitch
4. **FFT** — find the pitches in each slice
5. **Mel filter bank** — group nearby pitches into about 40 bands, spaced the
   way human hearing works (we easily tell 100 Hz from 200 Hz, but not 8000 Hz
   from 8100 Hz)
6. **Log** — convert to a loudness scale, like decibels
7. **DCT** — compress those 40 bands down to 13 numbers

**The ablation study proved this graph is the one that matters** on the music
branch:

| What we used | Accuracy |
|---|---|
| All 53 features | 0.787 |
| Everything **except** MFCC | 0.665 (−12 points) |
| **MFCC only** (39 features) | 0.770 (−2 points) |
| Spectral only (8 features) | 0.604 |
| Temporal only (6 features) | 0.573 |

Remove MFCC and accuracy collapses. Use only MFCC and you are within 2 points
of using everything.

---

## 6. The right-hand panel of the UI

| Section | What it shows |
|---|---|
| **FILE** | What you loaded, its sample rate, its length |
| **PREDICTION** | The answer, with a confidence bar per class. If two bars are close, the model is genuinely unsure — that is honest behaviour, not a bug |
| **DSP FEATURES** | A readable sample of the 53 numbers, so you can see what the model is actually looking at |
| **Status bar** | The result in one sentence, plus the *"screening result, not a diagnosis"* warning on medical files |

---

## 7. Where the training happened

**Nothing is trained when you press Classify.** That already happened, earlier,
once.

1. `run_experiment.py` walked all 500 GTZAN tracks through Stations 1–4
2. Each 30-second track was cut into six 5-second clips →
   **2,995 clips × 53 numbers**
3. `train_model.py` fed those to three classifiers (SVM, k-NN, Random Forest),
   picked the best, and saved it to `models/music.joblib`
4. The UI simply loads that saved file

### The critical detail

The train/test split was made **by track**, not randomly. All six clips from
one song stay together on the same side.

Otherwise the model sees clip 1 of a song in training and clip 2 in testing —
practically the same audio — and the score jumps 10 to 15 points for free.
That is the standard way GTZAN results get exaggerated, and we avoided it.

### What is trained on what

| Branch | Data | Clips | Best model | Balanced accuracy |
|---|---|---|---|---|
| Music | **Real GTZAN** | 2,995 from 500 tracks | Random Forest | **0.812** |
| Heart | **Real PhysioNet 2016** | 8,147 from 3,240 recordings | Random Forest | **0.833** (Se 0.79, Sp 0.88) |
| Respiratory | **Real ICBHI 2017** | 6,392 cycles from 126 patients | SVM (RBF) | **0.504** (ICBHI score 0.512) |

All three branches are trained on real data as of 21 Oct 2026. The synthetic
generator in `src/synth.py` is no longer used by any shipped model.

---

## 8. Per-branch settings

Only the sample rate and the frequency band change between branches. The code
path is identical — that is the whole point of the project.

| Branch | Sample rate | Band-pass | Mel filters | Classes |
|---|---|---|---|---|
| Music | 22,050 Hz | 30 Hz – 10 kHz | 40 | 5 genres |
| Heart | 2,000 Hz | 25 – 400 Hz | 26 | normal, abnormal |
| Respiratory | 4,000 Hz | 80 – 1,800 Hz | 30 | normal, wheeze, crackle |

The heart band comes from the physiology: S1 and S2 sit below about 150 Hz and
murmurs reach roughly 400 Hz, so 2 kHz sampling is plenty.

---

## 9. One-page summary

```
                        audio file
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
       MATLAB            Python           4 graphs
   sa_preprocess.m    preprocess.py      drawn live
   sa_features.m      features.py        from the audio
          │                 │
          └──► 53 numbers ◄─┘
                    │
          sa_verify.m checks
          both sides match
                    │
                    ▼
        trained model  models/*.joblib
        trained earlier by train_model.py
        on 2,995 real GTZAN clips
                    │
                    ▼
           "CLASSICAL — 92.3%"
```

---

## 10. If someone asks you one question, answer this way

**"What does your project actually do?"**

> "It turns any sound into 53 numbers that describe its shape in time and
> frequency, then uses those numbers to classify it. The same measuring code
> works on music and on heart and lung sounds — only the frequency range
> changes. That is the point we are proving: one DSP front end, two very
> different applications."
