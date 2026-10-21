# Every DSP Step Explained — MATLAB and Python, Side by Side

**Smart Audio Diagnostics · Group 09 · DSP · ECE, MNNIT Allahabad**
Anushka Patil · Arin Saraf · Arjit Abhitaj · Arnav Raj

Open this file during the viva. Every stage has: what it is, why we do it, how
it works, and **exactly which file and which line** in both MATLAB and Python.

---

# Part 0 — First, the question everyone asks

> **"Does MATLAB compute the features and send them to the Python model?"**

**No. And that is deliberate.** There is no live bridge between MATLAB and
Python. They are **two complete, independent implementations of the same
53-number specification.**

```
        MATLAB path (standalone)                Python path (standalone)
        ────────────────────────                ────────────────────────
        audio file                              audio file
            │                                       │
    sa_preprocess.m   ← same 5 steps →      preprocess.py: preprocess()
            │                                       │
    sa_features.m     ← same 53 numbers →   features.py: extract()
            │                                       │
    sa_classify.m                           train.py model (.joblib)
            │                                       │
        prediction                              prediction

                    ▲                                ▲
                    └────── sa_verify.m ─────────────┘
                    compares the 53 numbers from both
                    sides on the SAME file, one by one
```

**Both sides can classify on their own.** MATLAB is not "only making features" —
`matlab/sa_classify.m` produces a label too. It reads the trained model from
`model_export/*.csv`, which `scripts/export_for_matlab.py` wrote after Python
finished training.

### Why build it this way instead of piping MATLAB → Python?

**1. Two independent implementations agreeing is a proof. Plumbing is not.**
If MATLAB just handed numbers to Python, we would have written the MFCC code
once and would have no way of knowing it was correct. Because we wrote it
*twice*, independently, from the same textbook equations — and the two agree —
that agreement is evidence the maths is right. That is what `sa_verify.m` is
for, and it is a genuine DSP result, not glue code.

**2. Each toolchain stays usable alone.** MATLAB does not need Python installed.
Python does not need a MATLAB licence.

**3. A runtime bridge would be fragile and prove nothing.** Calling Python from
MATLAB mid-demo is exactly the kind of thing that breaks in front of an
examiner.

### What is true right now (be honest about this)

| | Status |
|---|---|
| `app_gui.py` — the UI we demo | **Fully Python.** Preprocessing, features and classification all in Python. MATLAB is not in this path. |
| `sa_preprocess.m` + `sa_features.m` | **Written and complete.** The MATLAB DSP front end. |
| `sa_classify.m` | **Written.** Classifies in MATLAB using the exported model. |
| `sa_verify.m` | **Run on 21 Oct.** MATLAB and Python agree to ≤ 6×10⁻⁸ relative error on all 53 features, all three branches, with and without denoising. It found two real bugs (rounding of 220.5, STFT padding), both fixed. |
| `sa_demo.m` | **New.** Full MATLAB path: read → pre-process → features → classify (exported model) → plot. |
| `SmartAudioDiagnostics.m` (group's App Designer GUI) | Still uses the old placeholder `if centroid < 1000` thresholds. `matlab/PATCH_ClassifySignal.m` replaces that with the real trained model — not pasted in yet. |

**Say this out loud:** *"Our Python path is complete and is what we demo. Our
MATLAB path mirrors it and has been verified against Python to 10⁻⁷. That
cross-check is our correctness proof for the MFCC implementation."*

---

# Part 1 — The whole pipeline in one picture

```
  ┌─────────────────────────────────────────────────────────────┐
  │  M1  DATA        read the audio file into numbers           │
  ├─────────────────────────────────────────────────────────────┤
  │  M2  PRE-PROCESSING   clean the signal                      │
  │      1. mono          2. resample     3. denoise            │
  │      4. band-pass     5. normalise    6. fixed length       │
  ├─────────────────────────────────────────────────────────────┤
  │  M3  FEATURES    turn the signal into 53 numbers            │
  │      framing → windowing → FFT → mel bank → log → DCT       │
  │      + centroid, bandwidth, roll-off, flatness, ZCR, energy │
  ├─────────────────────────────────────────────────────────────┤
  │  M4  CLASSIFIER  53 numbers → a label                       │
  ├─────────────────────────────────────────────────────────────┤
  │  M5  EVALUATION  accuracy, F1, confusion matrix, ablation   │
  └─────────────────────────────────────────────────────────────┘
```

**The core idea of the whole project:** M2 and M3 are *identical* for music and
for medical signals. Only two numbers change — the sampling rate and the
frequency band. That is what "one DSP front end, two branches" means.

---

# Part 2 — M1: Reading the audio

**What it is.** A sound file on disk is a list of numbers. Each number is the
air pressure at one instant. 22 050 numbers per second means the microphone
measured 22 050 times a second.

**Why we need custom readers.** Our three datasets come in three formats.

| Format | Where | Code |
|---|---|---|
| `.wav` | PhysioNet, ICBHI, our demo clips | `scipy.io.wavfile` |
| `.au` | **GTZAN ships as Sun AU** | `read_au()` — **we wrote this** |
| `.mp3`, `.flac`, `.m4a` | any song from the internet | `read_compressed()` — pipes through ffmpeg |

📁 **Python:** `src/preprocess.py`
- `read_au()` — line 16
- `read_compressed()` — line 39
- `read_audio()` — line 68 (picks the right one by file extension)

📁 **MATLAB:** built-in `audioread()`

> **Worth mentioning:** Python removed the `sunau` module in version 3.13, so we
> had to write the Sun AU decoder ourselves — read the 24-byte header, get the
> data offset, encoding, sample rate and channel count, all big-endian, then
> read 16-bit samples. It is 20 lines and it is real DSP file-format work.

---

# Part 3 — M2: Pre-processing, step by step

📁 **Python:** `src/preprocess.py` → `preprocess()` at **line 154** calls all six.
📁 **MATLAB:** `matlab/sa_preprocess.m` — the same six, lines 5–35.

---

## Step 1 — Stereo to mono

**What.** Average the left and right channels into one.

**Why.** Our features describe *one* signal. Stereo would give two different
answers for the same recording. Also, medical recordings are mono anyway, so
this keeps every branch consistent.

**How.** `mono = (left + right) / 2`

📁 Python: `to_mono()` — `src/preprocess.py:88`
📁 MATLAB: `sa_preprocess.m:6` — `x = mean(x,2)`

---

## Step 2 — Resampling

**What.** Change how many samples per second the signal has.

**Why.** Two reasons, and both matter:

1. **Consistency.** GTZAN is 22 050 Hz, your MP3 was 44 100 Hz. If we did not
   resample, the same song at two rates would produce different feature values
   and the model would break.
2. **Throwing away useless bandwidth.** Heart sounds live below 400 Hz. Storing
   them at 44 100 Hz wastes 98 % of the data on frequencies that contain
   nothing but noise. We drop heart audio to **2 000 Hz**.

**How — and this is the DSP point.** You cannot just delete samples. By the
**Nyquist theorem**, a signal sampled at *f*s can only represent frequencies up
to *f*s/2. If you downsample without first removing everything above the new
Nyquist limit, those high frequencies **fold back** and appear as fake low
frequencies. That is **aliasing**, and it is unrecoverable.

So proper resampling = **low-pass filter first, then decimate**. Both
`scipy.signal.resample_poly` and MATLAB's `resample` do this internally — they
are *polyphase* resamplers, which combine the filtering and the rate change
into one efficient operation.

We use `gcd` to reduce the ratio: 44 100 → 22 050 becomes "up by 1, down by 2".

📁 Python: `resample_to()` — `src/preprocess.py:92`
📁 MATLAB: `sa_preprocess.m:9-12`

> **If asked "what is aliasing?"** — *"A frequency above half the sampling rate
> gets mirrored down and shows up as a lower frequency that was never there. A
> 3 kHz tone sampled at 4 kHz looks like a 1 kHz tone. It cannot be undone
> afterwards, so you must low-pass before you downsample."*

---

## Step 3 — Noise reduction (spectral subtraction)

**What.** Estimate the constant background hiss and subtract it.

**Why.** Stethoscope and lung recordings have room noise and rubbing. That
noise shifts the spectral centroid and corrupts the MFCCs. Removing it makes
features depend on the *signal*, not the recording room.

**How.**
1. Cut the signal into overlapping frames and FFT each one (this is the STFT).
2. Find the **quietest 10 % of frames** — assume those are noise-only.
3. Average their magnitude spectrum → that is our noise estimate.
4. Subtract it from every frame: `clean = |X| − 1.5 × noise`
5. Never let it go below 5 % of the original (`floor_gain`), because subtracting
   too hard creates a warbling artefact called **musical noise**.
6. Keep the **original phase**, rebuild with inverse STFT.

> **The clever bit worth pointing out:** most textbook implementations assume the
> first half-second is silence. Ours does not — it *searches* for the quietest
> frames anywhere in the recording. That works on real recordings that start
> mid-sound.

**Why keep the original phase?** Human hearing and our features are driven by
the magnitude spectrum. Estimating clean phase is a hard, separate problem, and
reusing the noisy phase is standard practice and sounds fine.

📁 Python: `denoise()` — `src/preprocess.py:110`
📁 MATLAB: `sa_denoise()` — `matlab/sa_preprocess.m:38`

---

## Step 4 — Band-pass filtering

**What.** Keep only frequencies inside a chosen range; remove everything else.

**Why — this is where domain knowledge enters the DSP.** Each signal type has a
physically meaningful band:

| Branch | Band | Reason |
|---|---|---|
| Music | 30 Hz – 10 kHz | Full musical range; below 30 Hz is rumble, above 10 kHz is near the Nyquist limit |
| Heart | 25 – 400 Hz | S1/S2 energy is below ~150 Hz; murmurs reach ~400 Hz. **Everything above 400 Hz is noise.** |
| Respiratory | 80 – 1800 Hz | Wheezes are 100–1000 Hz; crackles are short bursts up to ~1500 Hz |

Set in `config.py` under `DOMAINS`.

**How.** A **Butterworth** filter of order 4.

- **Why Butterworth?** It has a *maximally flat* passband — no ripple. Other
  designs (Chebyshev, elliptic) cut off more sharply but ripple in the
  passband, which would distort the very amplitudes our features measure.
- **Why order 4?** Steepness vs stability trade-off. Higher order cuts more
  sharply but becomes numerically unstable.
- **Why SOS (second-order sections)?** A 4th-order filter written as one big
  polynomial accumulates rounding errors. Splitting it into cascaded 2nd-order
  biquads is numerically stable. That is what `zp2sos` / `output="sos"` do.

**The important part — zero-phase filtering.** We use `filtfilt`
(`sosfiltfilt` in Python), which runs the filter **forward and then backward**.

- A normal filter delays different frequencies by different amounts — **phase
  distortion**. That would smear the sharp S1/S2 heart clicks and the crackle
  transients, which are exactly what we are trying to detect.
- Running it forwards then backwards makes the delays cancel **exactly**. Zero
  phase distortion. The cost is that it doubles the effective filter order and
  it is non-causal (needs the whole signal) — fine, because we work offline.

📁 Python: `bandpass()` — `src/preprocess.py:99`
📁 MATLAB: `sa_preprocess.m:18-25`

> **Say this:** *"We use filtfilt, not filter, so there is zero phase
> distortion. In a heart sound the timing of S1 and S2 carries the information —
> if we smeared it we would destroy the thing we are measuring."*

---

## Step 5 — Amplitude normalisation

**What.** Divide by the loudest sample, so the signal spans −1 to +1.

**Why.** Somebody records loudly, somebody records quietly. That is a property
of the *microphone*, not the music or the heartbeat. Without this, the model
would learn "loud = rock" instead of learning actual timbre.

**How.** `x = x / max(|x|)`. We add a tiny `1e-12` so an all-zero file does not
divide by zero.

📁 Python: `normalize()` — `src/preprocess.py:127`
📁 MATLAB: `sa_preprocess.m:27`

---

## Step 6 — Fixed length

**What.** Make every clip exactly the same duration (5 s for music and heart,
3 s for respiratory).

**Why.** The classifier needs a **fixed-size** input — always 53 numbers. Longer
clips are centre-cropped (the middle is usually the most representative part;
songs often fade in and out). Shorter clips are zero-padded.

📁 Python: `fix_length()` — `src/preprocess.py:146`
📁 MATLAB: `sa_preprocess.m:29-35`

---

# Part 4 — M3: Feature extraction, step by step

📁 **Python:** `src/features.py` → `extract()` at **line 115**
📁 **MATLAB:** `matlab/sa_features.m` → `sa_features()` line 1

---

## Step 1 — Framing

**What.** Chop the signal into short overlapping pieces. Music: 25 ms frames,
10 ms hop. Heart/respiratory: 50 ms frames, 25 ms hop.

**Why.** Audio is **non-stationary** — its frequency content changes constantly.
One FFT over 5 seconds would blur a whole song into an average and tell you
nothing about *when* anything happened. But over 25 ms, the signal is
approximately steady ("quasi-stationary"), so an FFT of that frame is
meaningful.

**Why overlap?** With a hop shorter than the frame, consecutive frames share
samples. Without overlap, an event landing on a frame boundary would be split
and weakened. Overlap guarantees every event sits nicely inside some frame.

**Why longer frames for medical?** Heart sounds are low-frequency. To resolve a
25 Hz component you need a window long enough to contain a full cycle of it —
25 Hz has a 40 ms period, so a 25 ms frame is too short. This is the
**time–frequency resolution trade-off**: long frames give good frequency
resolution and poor time resolution; short frames the opposite.

📁 Python: `frame_signal()` — `src/preprocess.py:134`
📁 MATLAB: `sa_frame()` — `matlab/sa_features.m:68`

---

## Step 2 — Windowing (Hamming)

**What.** Multiply each frame by a bell-shaped curve that tapers to near zero at
both ends.

**Why — this is a classic exam question.** The FFT assumes the frame repeats
forever. If a frame starts at +0.8 and ends at −0.6, the imagined repetition has
a sudden jump at the join. A sharp jump contains *every* frequency, so it
splatters false energy across the whole spectrum. That is **spectral leakage**.

Tapering the edges to zero removes the discontinuity and the leakage.

**Why Hamming specifically?** `w[n] = 0.54 − 0.46·cos(2πn/(N−1))`. It is chosen
to cancel the largest side-lobe, giving about −43 dB side-lobe suppression —
better than Hann for leakage, at the cost of a slightly wider main lobe. It is
the standard choice for speech and audio, used in the original MFCC paper
(Davis & Mermelstein, reference [1]).

📁 Python: `frame_signal(..., window="hamming")` — `src/preprocess.py:134`
📁 MATLAB: `matlab/sa_features.m:74` — `F .* hamming(frameLen).'`

---

## Step 3 — FFT

**What.** Convert each frame from amplitude-over-time into
energy-over-frequency.

**Why.** Everything that separates our classes lives in frequency: a wheeze is a
tone at one frequency; classical music has energy concentrated low; cymbals put
energy high. In the time domain these all just look like wiggly lines.

**How.** The DFT is
`X[k] = Σ x[n]·e^(−j2πkn/N)`. Computing it directly costs N² operations. The
**FFT** is an algorithm that gets the identical answer in N·log N by
recursively splitting the sum into even and odd halves. For N = 512 that is
about 60× faster — and it is why real-time DSP is possible at all.

We use `rfft` (real FFT): audio is real-valued, so the spectrum is symmetric and
the negative half is redundant. We keep N/2+1 bins.

We then square the magnitude to get **power**: `power = |X|² / N`.

📁 Python: `stft_power()` — `src/features.py:47`
📁 MATLAB: `matlab/sa_features.m:14-18`

---

## Step 4 — MFCC (the main deliverable)

This is **Objective O2** of the synopsis. We compute it from scratch on both
sides — we deliberately did **not** call `librosa.mfcc()` or MATLAB's `mfcc()`,
because that function *is* the thing we were asked to implement.

📁 Python: `mfcc()` — `src/features.py:59`
📁 MATLAB: `matlab/sa_features.m:20-30`

### 4a. Pre-emphasis

**What.** `y[n] = x[n] − 0.97·x[n−1]` — a simple high-pass filter.

**Why.** Natural sounds roll off about −6 dB per octave; high frequencies are
much quieter than low ones. Without boosting them, the high-frequency detail
would be numerically swamped and the later log step would amplify its noise.
Pre-emphasis flattens the spectrum so all bands contribute meaningfully.

📁 Python: `pre_emphasis()` — `src/features.py:13`
📁 MATLAB: `matlab/sa_features.m:21`

### 4b. Mel filter bank

**What.** Instead of 257 individual FFT bins, group them into 26–40 overlapping
triangular bands, spaced according to the **mel scale**.

**Why the mel scale?** Human hearing is not linear in frequency. We easily hear
the difference between 200 Hz and 300 Hz, but 5000 Hz and 5100 Hz sound
identical — even though both gaps are 100 Hz. The mel scale compresses
frequency to match perception:

```
mel = 2595 · log10(1 + f/700)
```

So mel filters are **narrow at low frequencies and wide at high frequencies** —
you can see this in `results/figures/melbank_music.png`.

**Why does that help a classifier?** It puts resolution where the information
actually is, and it reduces 257 numbers to 40 — fewer features, less noise, less
overfitting.

**Why triangular and overlapping?** Triangles are a simple smooth weighting;
overlapping means a frequency sitting between two filter centres still gets
counted, so there are no blind spots.

We also apply **Slaney area normalisation** (`enorm`), so a wide high-frequency
filter does not automatically report more energy just because it is wider.

📁 Python: `mel_filterbank()` — `src/features.py:25`; `hz_to_mel()` line 17
📁 MATLAB: `sa_melbank()` — `matlab/sa_features.m:78`

### 4c. Logarithm

**What.** Take `log` of each mel band's energy.

**Why two reasons:**
1. **Perception.** Loudness perception is logarithmic — that is why we use
   decibels. Doubling the power is one step up, not twice as loud.
2. **The maths reason (this is the good answer).** A recorded sound is the
   source convolved with the room/microphone response. In the frequency domain
   convolution becomes **multiplication**. Taking the log turns multiplication
   into **addition** — so the source and the channel become *additive* instead
   of tangled together, and the next step can separate them.

### 4d. DCT — the "cepstrum"

**What.** Apply a Discrete Cosine Transform to the 26–40 log-mel values and keep
the first **13**.

**Why.** After the log, the source and the channel are added together, and they
vary at different *rates across frequency*. The DCT is a frequency analysis *of
the spectrum itself* — that is why it is called a **cepstrum** ("spectrum" with
the first syllable reversed, and the axis is called **quefrency**). Slowly
varying shape lands in the low coefficients; fast ripple lands in the high ones.

Keeping the **first 13** keeps the smooth spectral envelope — the **timbre**,
which is what distinguishes a violin from a trumpet playing the same note — and
discards pitch and fine detail.

The DCT also **decorrelates** the mel bands (neighbouring bands are highly
correlated because the filters overlap), which is exactly what SVM and k-NN
prefer.

**Which DCT?** Type-II, orthonormal. `scipy.fft.dct(type=2, norm="ortho")` in
Python and MATLAB's `dct()` are the same transform — that is what makes the two
implementations comparable.

📁 Python: `src/features.py:68` — `dct(log_mel, type=2, axis=1, norm="ortho")`
📁 MATLAB: `matlab/sa_features.m:27` — `dct(logMel, [], 2)`

### 4e. Delta-MFCC

**What.** How fast each MFCC is changing over time — a slope, computed over ±2
frames.

**Why.** A static MFCC describes one instant. Deltas capture *motion* — attack
transients, note changes, the sudden onset of a crackle. A crackle and a steady
noise can have similar average MFCCs but completely different deltas.

📁 Python: `delta()` — `src/features.py:70`
📁 MATLAB: `sa_delta()` — `matlab/sa_features.m:97`

---

## Step 5 — The spectral shape features

All four describe the *shape* of the power spectrum. Simple, cheap, and
interpretable — you can point at them in the UI.

### Spectral centroid

**What.** The centre of mass of the spectrum. The "average frequency".

**Formula.** `centroid = Σ(f · power) / Σ(power)` — a weighted average.

**Why.** This is the DSP measure of **brightness**. Classical music sits low;
cymbals and distorted guitar sit high. In the UI, classical measured 1117 Hz
while the heart sound measured 163 Hz.

📁 Python: `spectral_centroid()` — `src/features.py:79`
📁 MATLAB: `matlab/sa_features.m:33-34`

### Spectral bandwidth

**What.** The spread of energy around the centroid — the standard deviation of
the spectrum.

**Why.** Separates **tonal** from **noisy**. A pure flute note has almost all
energy at one frequency → small bandwidth. A cymbal crash spreads everywhere →
large bandwidth. Our unit test measures 33 Hz for a pure tone and 3180 Hz for
white noise.

📁 Python: `spectral_bandwidth()` — `src/features.py:84`
📁 MATLAB: `matlab/sa_features.m:35-36`

### Spectral roll-off (85 %)

**What.** The frequency below which 85 % of the total energy lies.

**Why.** A robust measure of "where does this sound stop". Less sensitive to a
single loud outlier bin than the centroid is. Good at separating
high-frequency-rich genres from mellow ones.

📁 Python: `spectral_rolloff()` — `src/features.py:90`
📁 MATLAB: `matlab/sa_features.m:37-38`

### Spectral flatness

**What.** Geometric mean ÷ arithmetic mean of the spectrum. Ranges 0 to 1.

**Why.** A direct **tone-vs-noise** measure. Pure white noise is flat across all
frequencies → both means are equal → flatness ≈ 1. A pure tone has one huge
peak and near-zero elsewhere → geometric mean collapses → flatness ≈ 0.

This is the single best feature for the respiratory branch: a **wheeze is
tonal** (low flatness), normal breath sound is noise-like (high flatness). Our
wheeze demo measured 0.0707 and the heart sound 0.0081.

📁 Python: `spectral_flatness()` — `src/features.py:97`
📁 MATLAB: `matlab/sa_features.m:39-40`

---

## Step 6 — The time-domain features

Computed on **un-windowed** frames, because a window would distort them.

### Zero-crossing rate (ZCR)

**What.** How often the waveform crosses zero, per frame.

**Why.** The cheapest possible pitch/noisiness estimate — no FFT needed. For a
pure tone it equals exactly `2f/fs` (two crossings per cycle), which is one of
our unit tests. High ZCR means high-frequency or noisy content; low ZCR means
low-frequency. Historically used to separate voiced from unvoiced speech.

📁 Python: `zero_crossing_rate()` — `src/features.py:103`
📁 MATLAB: `matlab/sa_features.m:44`

### Short-time energy

**What.** Mean of the squared samples in a frame — how loud that instant is.

**Why.** Captures **rhythm and dynamics**. A hip-hop track has huge energy spikes
on every beat; classical swells gradually. In heart sounds the energy envelope
*is* the S1–S2–pause pattern. Also gives RMS (`√energy`), the perceptual loudness
measure.

📁 Python: `short_time_energy()` — `src/features.py:107`
📁 MATLAB: `matlab/sa_features.m:45`

---

## Step 7 — Assembling the 53 numbers

Each feature above gives one value **per frame** — a 5 s music clip has about
500 frames. But the classifier needs a **fixed-size** input. So we take the
**mean and standard deviation** of each feature across all frames.

Mean says "what is this recording typically like"; standard deviation says "how
much does it vary" — which is itself informative, since steady and fluctuating
sounds differ.

| Group | Contents | Count |
|---|---|---|
| `mfcc` | 13 MFCC means + 13 MFCC std + 13 Δ-MFCC means | **39** |
| `spectral` | centroid, bandwidth, roll-off, flatness (mean + std each) | **8** |
| `temporal` | ZCR, energy, RMS (mean + std each) | **6** |
| | **Total** | **53** |

Each feature carries its group in its name (`mfcc::mfcc3_mean`), which is what
makes the ablation study a one-line mask instead of a rewrite.

📁 Python: `extract()` — `src/features.py:115`; `group_mask()` line 160
📁 MATLAB: `matlab/sa_features.m:47-66` — **same order, deliberately**

> **The order matters.** A model trained in Python expects centroid at index 39.
> If MATLAB put it at index 40, `sa_classify.m` would produce nonsense. That is
> why both files build the vector in the same sequence, and why `sa_verify.m`
> checks element by element.

---

# Part 5 — M4: Classification

📁 `src/train.py` — `make_models()` line 18

Three classical classifiers, as the synopsis specifies:

| Classifier | How it works, in one line | Best for |
|---|---|---|
| **SVM (RBF)** | Finds the boundary with the widest margin between classes, after bending the space with a Gaussian kernel | Heart (0.983) |
| **k-NN (k=5)** | Look at the 5 nearest training examples, take a majority vote | (weakest here) |
| **Random Forest** | 300 decision trees each voting, each trained on a random subset | Music (0.818), respiratory (0.767) |

Every one is wrapped in a `Pipeline` with `StandardScaler` first, which
rescales each of the 53 features to zero mean and unit variance.

**Why scaling is essential:** spectral centroid is ~1000 (Hz) and ZCR is ~0.1.
Without scaling, distance-based methods (SVM, k-NN) would let centroid dominate
entirely and ZCR would be ignored. Random Forest does not need it, but keeping
one pipeline for all three keeps the comparison fair.

**Where the model lives:** `train_model.py` fits it and saves
`models/<branch>.joblib`. The UI loads that file — **nothing trains when you
press Classify.**

📁 MATLAB equivalent: `matlab/sa_classify.m` implements k-NN directly against
`model_export/*.csv` — no Statistics Toolbox needed.

---

# Part 6 — What to point at in the UI

When you show the four graphs, this is what each one proves:

| Graph | Plain-English meaning | What to say |
|---|---|---|
| **Waveform** | How loud, moment to moment | "This is after pre-processing — resampled, band-passed, normalised" |
| **FFT spectrum** | Which frequencies, over the whole clip | "Single-sided magnitude spectrum. Its weakness is that it mashes all time together" |
| **Spectrogram** | Which frequencies **at which moment** | "This fixes that — it is an FFT of every short frame, stacked. **On the wheeze file you can see the wheeze as a horizontal line at 300 Hz**" |
| **MFCC** | The 13-number timbre fingerprint per frame | "Computed from scratch: pre-emphasis, framing, Hamming, FFT, mel bank, log, DCT" |

**The spectrogram is your best demo moment.** Load `respiratory_wheeze_01.wav` and
point at the horizontal band — you are showing the examiner the classifier's
evidence directly.

---

# Part 7 — Quick file index

| I need to show… | File | Where |
|---|---|---|
| Reading .au / .mp3 | `src/preprocess.py` | lines 16, 39, 68 |
| The whole M2 chain | `src/preprocess.py` | `preprocess()` line 154 |
| Resampling | `src/preprocess.py` | line 92 |
| Band-pass + filtfilt | `src/preprocess.py` | line 99 |
| Denoising | `src/preprocess.py` | line 110 |
| Framing / windowing | `src/preprocess.py` | line 134 |
| **MFCC from scratch** | `src/features.py` | line 59 |
| Mel filter bank | `src/features.py` | line 25 |
| Spectral features | `src/features.py` | lines 79–101 |
| ZCR / energy | `src/features.py` | lines 103–109 |
| The 53-vector assembly | `src/features.py` | line 115 |
| Classifiers | `src/train.py` | line 18 |
| Group-aware splitting | `src/train.py` | `split()` line 32 |
| Ablation study | `src/ablation.py` | line 11 |
| **MATLAB pre-processing** | `matlab/sa_preprocess.m` | whole file |
| **MATLAB features** | `matlab/sa_features.m` | whole file |
| **MATLAB classifier** | `matlab/sa_classify.m` | whole file |
| **MATLAB ↔ Python check** | `matlab/sa_verify.m` | whole file |
| The UI | `app_gui.py` | line 332 = the classify path |
| DSP unit tests | `tests/test_dsp.py` | 19 checks |

---

# Part 8 — The five questions you will be asked

**"Why implement MFCC yourself instead of using a library?"**
> "Because computing MFCC is Objective O2 of our synopsis. Calling
> `librosa.mfcc()` would have replaced the deliverable with one line. We use
> libraries for filter design, FFT and DCT — the standard building blocks — but
> the MFCC pipeline itself is ours, in both languages."

**"Why filtfilt and not filter?"**
> "Zero phase distortion. A normal filter delays different frequencies by
> different amounts, which would smear the S1/S2 timing in a heart sound —
> exactly the information we are measuring. filtfilt runs forwards then
> backwards so the delays cancel."

**"Why the mel scale?"**
> "Human hearing is logarithmic in frequency. 200 to 300 Hz is an obvious
> difference; 5000 to 5100 Hz is inaudible. Mel spacing puts the resolution
> where the information is, and it cuts 257 FFT bins down to 40."

**"Why take the log before the DCT?"**
> "Two reasons. Perception is logarithmic. And mathematically, a recording is
> the source convolved with the microphone response, which is multiplication in
> the frequency domain — the log turns it into addition, so the DCT can separate
> them. That is what makes it a cepstrum."

**"Does MATLAB feed the Python model?"**
> "No — they are two independent implementations of the same 53-feature
> specification, and `sa_verify.m` compares them element by element. Two
> independent implementations agreeing is our proof that the MFCC maths is
> correct. If we had written it once and piped the numbers across, we would have
> no way to check it."
