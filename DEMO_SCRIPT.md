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
prediction, so the examiner can see it is not staged.

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
  **Classify** → *ABNORMAL ~89 %, ✓ correct*.
- **Signal analysis tab:** "S1 and S2 are the two spikes per beat; the energy
  between them in the spectrogram is the murmur."
- **DSP pipeline tab:** "Raw vs filtered. This is our 4th-order Butterworth
  band-pass, 25–400 Hz, run forwards and backwards (zero phase), so the sharp
  S1/S2 onsets are not smeared. Bottom right is the mel filter bank used for
  the MFCCs, which we coded from scratch."
- **Rhythm & timeline tab:** "Heart rate without any ML: envelope →
  autocorrelation → the first strong peak is one beat period → bpm. The bottom
  strip shows each 5 s segment's vote; we average them."
- Then **heart_normal_03.wav** → *NORMAL ~98 %, ✓*.
- **Export PDF report** → open it: "This is what goes to the doctor."

**Say the disclaimer out loud:** "Screening, not diagnosis. It decides who
should see a cardiologist first."

## 3. Lungs (60 s)

- **respiratory_wheeze_01.wav** → *WHEEZE ✓*. Point at the spectrogram: "Those
  bright curving lines are the wheeze. It's a musical tone, 200–800 Hz."
- **respiratory_recording_01.wav** (a full 20 s ICBHI recording) → Rhythm tab →
  **≈ 12 breaths/min**, a normal adult rate.
- Be upfront: "Lungs are our hardest branch: 50 % balanced accuracy over three
  classes, chance is 33 %. That's in line with classical-feature methods on
  ICBHI. Crackles last under 20 ms and get averaged away in clip-level
  features. We tested four pre-processing variants and none helped. That's
  documented in `results/RESPIRATORY_EXPERIMENTS.md`, and event-level or CNN
  features are our future work."

## 4. Music — same front end, different domain (45 s)

- **music_classical_02.wav** → *CLASSICAL ~90 % ✓*, then **music_rock_01.wav**
  → *ROCK ✓*, Rhythm tab → tempo.
- "Same code path: only 22 kHz sampling and a 30 Hz–10 kHz band. 82 %
  accuracy over five genres, split by song so no song is in both train and test."

## 5. The "smart" bits (60 s)

- **Unknown sound:** Open audio → `C:\Windows\Media\Windows Notify Email.wav` →
  *AUTO-DETECTED: UNKNOWN SOUND* in orange. "A classifier always answers one
  of its classes. So the router has a fourth class trained on 2 000
  environmental sounds (ESC-50). It rejected 61 of 70 Windows sounds it had
  never heard."
- **Live recording:** ● Record 6 s, and let the examiner clap, talk or play a
  song from a phone. Music → routed to music; speech/claps → *unknown*.
- **Batch folder…** → pick `data\demo_wavs` → table of 42 files + CSV +
  accuracy on files with a known label. "That's the music-library-tagging and
  daily-screening use case."
- **Model performance tab:** confusion matrix and classifier comparison for the
  current branch.

## 6. MATLAB (30 s, optional; needs ~1 min to start MATLAB)

```matlab
cd C:\dsp; addpath('matlab');
sa_verify('data/physionet2016/training-a/a0001.wav','heart', ...
          'results/tables/verify_python_heart.csv');
sa_demo('data/demo_wavs/music_jazz_01.wav','music');
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
| "That one is wrong." | Yes. E.g. `heart_normal_01` comes out abnormal at 86 %. We report 83 % balanced accuracy, so about 1 in 6 is wrong, and we show the misses. |
| "Show me it isn't hard-coded." | Open any file they choose, or run `python tests/test_system.py` (52 checks incl. GUI, mic and PDF). |

## Backup if the GUI will not start

```bash
python scripts/classify_file.py data/demo_wavs     # auto-detects each file
python run_experiment.py --skip-figures
```

Screenshots of every tab are in `results/figures/ui_*.png`, and sample reports in
`results/reports/`.
