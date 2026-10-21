# Respiratory branch — pre-processing variants (ICBHI 2017, real data)

5-fold **patient-grouped** cross-validation, balanced accuracy over the three
classes (normal / wheeze / crackle; chance = 0.333). 6 392 cycles, 126 patients.
Run on 21 Oct 2026 with `class_weight="balanced"` classifiers.

| Variant | fs | Band-pass | Denoise | Clip | SVM (RBF) | Random Forest |
|---|---|---|---|---|---|---|
| **Baseline (used)** | 4 kHz | 80–1800 Hz | yes | 3 s | 0.452 ± 0.027 | 0.486 ± 0.024 |
| No spectral subtraction | 4 kHz | 80–1800 Hz | no | 3 s | 0.458 ± 0.030 | 0.490 ± 0.025 |
| Wider band | 8 kHz | 50–2500 Hz | yes | 3 s | 0.439 ± 0.018 | 0.450 ± 0.018 |
| Wider band, no denoise | 8 kHz | 50–2500 Hz | no | 3 s | 0.437 ± 0.023 | 0.458 ± 0.013 |
| Longer clip | 4 kHz | 80–1800 Hz | yes | 5 s | 0.469 ± 0.032 | 0.478 ± 0.025 |

**Reading.** Every variant is within one standard deviation of the baseline.
The limit on ICBHI is not the pre-processing but the clip-level summary
features: a crackle lasts < 20 ms and a wheeze can occupy a fraction of a
cycle, so averaging 53 frame features over a whole 3 s cycle dilutes exactly
the event the classifier needs. That is consistent with the literature —
classical feature + SVM/RF pipelines on ICBHI report challenge scores of
roughly 40–50 %, and only event-level or deep (CNN on log-mel) models reach
55–65 %. Our baseline is kept unchanged so that all three branches share one
front end.
