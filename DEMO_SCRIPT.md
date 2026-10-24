# Demo script — 6 minutes in front of the examiner

Before you walk in:

```bash
cd C:\dsp
python tests/test_system.py --no-mic   # 30 s sanity check: must end "checks passed"
python app_gui.py
```

Leave **Signal type = Auto-detect** (the default). Keep `TEAM_GUIDE.md`
section 5 open in a second window for the numbers. Turn the laptop volume up
(for ▶ Play) and keep the room reasonably quiet (for ● Record).

**Who speaks:** Arnav drives the laptop and does sections 1, 4, 5 and the
results questions. In section 2, **Arjit** explains the *DSP pipeline* and
*Rhythm* tabs (filters, noise reduction, heart rate) and **Arin** explains the
*Signal analysis* tab (FFT, spectrogram, MFCC). Arin and Arjit do section 6
(MATLAB).

Every demo clip is from the **held-out test split**: the model never trained on
these recordings. The panel shows **"Ground truth … ✓ correct"** under the
prediction, so the examiner can see it is not staged. All 31 clips in the
dropdown were opened one by one in the app and pass (right branch, right
class, no warnings, PDF exports); regenerate them with
`python scripts/make_demo_wavs.py --per-class 12 --verified`.

---

## 1. Architecture and use case (40 s)

> "Stethoscopes are cheap now, but interpreting heart and lung sounds needs a
> specialist, and rural PHCs don't have one. Our system takes a stethoscope
> recording, works out what it is, filters it, extracts 53 DSP features and
> flags *normal* or *refer*, with heart rate or breathing rate and a PDF report
> for the doctor. The same DSP front end also classifies music genres. That's
> the point of the project: one front end, separate classifiers, and only the
> sampling rate and the filter band change."

## 2. Heart — the main use case (90 s)

- Demo signal → **heart_abnormal_03.wav** → *AUTO-DETECTED: HEART SOUND* →
  **Classify** → *ABNORMAL 82 %, ✓ correct*, heart rate 54 bpm.
- **Signal analysis tab:** "S1 and S2 are the two spikes per beat; the energy
  between them in the spectrogram is the murmur."
- **DSP pipeline tab:** "Raw vs filtered. This is our 4th-order Butterworth
  band-pass, 25–400 Hz, run forwards and backwards (zero phase), so the sharp
  S1/S2 onsets are not smeared. Bottom right is the mel filter bank used for
  the MFCCs, which we coded from scratch."
- **Rhythm & timeline tab:** "Heart rate without any ML: envelope →
  autocorrelation → the first strong peak is one beat period → bpm. The bottom
  strip shows each 5 s segment's vote; we average them."
- Then **heart_normal_03.wav** → *NORMAL 96 %, ✓*, 59 bpm.
- **Export PDF report** → open it: "This is what goes to the doctor."

**Say the disclaimer out loud:** "Screening, not diagnosis. It decides who
should see a cardiologist first."

## 3. Lungs (60 s)

- **respiratory_wheeze_02.wav** → *WHEEZE 77 % ✓*. Point at the spectrogram:
  "Those bright lines are the wheeze. It's a musical tone, 100–1000 Hz."
- **respiratory_crackle_03.wav** → *CRACKLE 75 % ✓*, then
  **respiratory_normal_01.wav** → *NORMAL 75 % ✓*.
- **respiratory_wheeze_rec01.wav** (a full 20 s recording) → *WHEEZE ✓* →
  Rhythm tab → **20 breaths/min**, a normal adult rate. Every 3 s segment in
  the timeline votes wheeze.

## 4. Music — same front end, different domain (45 s)

- **music_classical_03.wav** → *CLASSICAL 89 % ✓*, **music_pop_03.wav** →
  *POP 95 % ✓*, **music_hiphop_01.wav** → *HIPHOP 95 % ✓*; Rhythm tab → tempo.
- "Same code path: only 22 kHz sampling and a 30 Hz–10 kHz band. 82 %
  accuracy over five genres, split by song so no song is in both train and test."

## 5. The "smart" bits (60 s)

- **Unknown sound:** Open audio → `C:\Windows\Media\Alarm04.wav` (an alarm
  tone) and then `C:\Windows\Media\Speech Disambiguation.wav` (a voice) →
  both show *AUTO-DETECTED: UNKNOWN SOUND* in orange. "A classifier always answers one
  of its classes. So the router has a fourth class trained on 2 000
  environmental sounds (ESC-50). It rejected 61 of 70 Windows sounds it had
  never heard."
- **Live recording:** ● Record 6 s, and let the examiner clap, talk or play a
  song from a phone. Music → routed to music; speech/claps → *unknown*.
- **Batch folder…** → pick `data\demo_wavs` → table of all 31 files + CSV +
  accuracy on files with a known label. "That's the music-library-tagging and
  daily-screening use case."
- **Model performance tab:** confusion matrix and classifier comparison for the
  current branch.

## 6. MATLAB (30 s, optional; needs ~1 min to start MATLAB)

```matlab
cd C:\dsp; addpath('matlab');
sa_verify('data/physionet2016/training-a/a0001.wav','heart', ...
          'results/tables/verify_python_heart.csv');
sa_demo('data/demo_wavs/heart_abnormal_02.wav','heart');   % ABNORMAL
sa_demo('data/demo_wavs/music_jazz_01.wav','music');        % JAZZ 93 %
```

> "Two independent implementations, MATLAB and Python, agree to within
> 6×10⁻⁸ on all 53 features. That cross-check found two real bugs: Python
> rounds 220.5 to 220 and MATLAB to 221, and the two STFTs pad differently."

---

## If they push

| Question | Answer |
|---|---|
| "Is 82 % inflated?" | No. Split by song/recording/patient (`GroupShuffleSplit`). A random split adds 10–15 points on GTZAN for free. |
| "Train accuracy is 100 %, isn't that overfitting?" | Random Forest fits training data perfectly by design. What matters is CV ≈ test: 0.77 vs 0.82 (music), 0.87 vs 0.86 (heart). |
| "Heart data is 80 % normal." | Hence `class_weight="balanced"`, selection by balanced accuracy, and reporting sensitivity 79 % / specificity 88 %. |
| "Which features matter?" | Ablation: removing MFCC costs 13 points on music and 7 on lungs; MFCC alone is within 1–3 points of all 53. |
| "Why not deep learning?" | A few thousand clips; explainable DSP features train in seconds and the synopsis specifies SVM/k-NN/RF. A CNN on log-mel spectrograms is future work. |
| "What is the real accuracy?" | On the full held-out test set: music 81 %, heart 83 % balanced, lungs 50 % balanced (chance 33 %, in line with classical methods on ICBHI; see `results/RESPIRATORY_EXPERIMENTS.md`). The demo clips are examples; the numbers come from thousands of test clips. |
| "Show me it isn't hard-coded." | Open any file they choose, or run `python tests/test_system.py` (53 checks incl. GUI, mic and PDF). |

## Backup if the GUI will not start

```bash
python scripts/classify_file.py data/demo_wavs     # auto-detects each file
python run_experiment.py --skip-figures
```

Screenshots of every tab are in `results/figures/ui_*.png`, and sample reports in
`results/reports/`.
