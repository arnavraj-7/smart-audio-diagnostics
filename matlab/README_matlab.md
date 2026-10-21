# MATLAB side

MATLAB is used for the **DSP front end, the verification of it, and the demo GUI**.
Python is used for **training and evaluating the classifiers**. Nothing is
duplicated for its own sake — the two toolchains meet at one 53-element
feature vector.

| File | Role |
|---|---|
| `sa_config.m` | DSP parameters per domain. Mirror of `config.py`. |
| `sa_preprocess.m` | M2: resample → spectral-subtraction denoise → Butterworth band-pass (zero-phase) → normalise → fixed-length. |
| `sa_features.m` | M3: framing/Hamming → FFT → mel filter bank → log → DCT (MFCC from scratch), plus centroid, bandwidth, roll-off, flatness, ZCR, short-time energy. Same 53 features, same order as `src/features.py`. |
| `sa_classify.m` | M4: applies the model exported from Python (`model_export/`). |
| `sa_verify.m` | Cross-toolchain check: MATLAB feature vector vs Python feature vector on the same file. |
| `sa_demo.m` | End-to-end MATLAB demo: read → pre-process → features → classify every 5 s segment with the exported model → 4-panel figure. |
| `sa_round.m` | Round-half-to-even like Python, so frame/hop lengths match exactly (22 050 × 10 ms = 220.5 → 220). |
| `PATCH_ClassifySignal.m` | Drop-in replacement for the placeholder threshold classifier in `SmartAudioDiagnostics.m`. |

## Running the verification (the part worth showing in the viva)

```
python scripts/make_demo_wavs.py
python scripts/export_for_matlab.py --verify data/demo_wavs/heart_abnormal_01.wav --domain heart
```

then in MATLAB, from the repo root:

```matlab
addpath('matlab');
sa_verify('data/demo_wavs/heart_abnormal_01.wav', 'heart');
```

It prints the MATLAB value, the Python value and the relative error for all 53
features. Agreement here is the evidence that the MFCC/spectral implementation
is correct — two independent implementations of the same textbook equations
landing on the same numbers.

## Wiring the GUI to the trained model

1. `python scripts/export_for_matlab.py` (writes `model_export/*.csv`)
2. In `SmartAudioDiagnostics.m`, replace the `ClassifySignal` method body with
   `PATCH_ClassifySignal.m`.
3. Add `'Heart Sound'` and `'Respiratory Sound'` to `SignalTypeDropDown.Items`
   so the medical branch is split into its two tasks.

## Verified result (21 Oct 2026)

| Branch | File | Max relative error, no denoise | With spectral-subtraction denoise |
|---|---|---|---|
| music | `gtzan/genres/jazz/jazz.00000.au` | 3.9 × 10⁻⁸ | 5.6 × 10⁻⁸ |
| heart | `physionet2016/training-a/a0001.wav` | 3.2 × 10⁻⁸ | 6.0 × 10⁻⁸ |
| respiratory | `icbhi/.../101_1b1_Al_sc_Meditron.wav` | 1.9 × 10⁻⁸ | 1.6 × 10⁻⁸ |

Reproduce:

```bash
python scripts/export_for_matlab.py --verify data/physionet2016/training-a/a0001.wav --domain heart
python scripts/export_for_matlab.py --verify data/physionet2016/training-a/a0001.wav --domain heart --denoise
```
```matlab
sa_verify('data/physionet2016/training-a/a0001.wav','heart','results/tables/verify_python_heart.csv')
sa_verify('data/physionet2016/training-a/a0001.wav','heart','results/tables/verify_python_heart_dn.csv', true)
sa_demo('data/demo_wavs/heart_abnormal_03.wav','heart')
```

Two discrepancies were found and fixed on the way: MATLAB `round(220.5)` = 221
vs Python 220 (`sa_round.m`), and MATLAB `stft/istft` padding vs SciPy's
(`sa_denoise` now frames exactly like `scipy.signal.stft`). MATLAB k-NN
classification of the 40 demo clips: 32/40.
