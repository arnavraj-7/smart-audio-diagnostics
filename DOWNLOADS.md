# What to download, and where to put it

Three datasets. Total ≈ 3.4 GB. **All three are downloaded and in use.** The loaders in `src/datasets.py` are already
written for all three label formats, so once the folders below exist,
`python run_experiment.py --no-cache` picks them up automatically and the
`source` line in the output changes from `synthetic` to `real:...`.

**Folder names matter.** Kaggle zips unpack with their own top-level folder
names — rename to match the left column exactly.

---

## 1. Music branch — GTZAN Genre Collection ✔ DONE

Already downloaded (`archive.zip`, 1.23 GB) and extracted to
`data/gtzan/genres/`. It is the original Sun AU release, so the files are
`.au` not `.wav` — `src/preprocess.py` has a reader for that format, since
Python removed the `sunau` module in 3.13. The music branch is trained on it.

Source: <https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection>

Layout on disk (`.au` or `.wav` both work):

```
data/gtzan/genres/classical/classical.00000.au
data/gtzan/genres/jazz/...
data/gtzan/genres/rock/...
data/gtzan/genres/pop/...
data/gtzan/genres/hiphop/...
```

`genres_original` is accepted too. Only 5 of the 10 genres are used; the rest
stay on disk and are ignored. All 500 files in the 5 used genres loaded
without error.

---

## 2. Heart branch — PhysioNet/CinC Challenge 2016 (190 MB) ✔ DONE (21 Oct)

Downloaded `training.zip` (3 240 recordings, `training-a` … `training-f`) from
the PhysioNet open-data mirror on AWS, which is much faster than physionet.org:

```bash
curl -o data/physionet2016/training.zip https://physionet-open.s3.amazonaws.com/challenge-2016/1.0.0/training.zip
cd data/physionet2016 && unzip training.zip
```

Alternative source:

<https://www.kaggle.com/datasets/bjoernjostein/physionet-challenge-2016>

Unpack so that the `.wav` files and the `REFERENCE.csv` files sit anywhere
under:

```
data/physionet2016/
```

The original release is split into `training-a` … `training-f`, each with its
own `REFERENCE.csv`. Keep that structure — the loader searches recursively and
merges every `REFERENCE*.csv` it finds. The CSV is `recordname,label` where
`1` = abnormal and `-1` = normal.

---

## 3. Respiratory branch — ICBHI 2017 Respiratory Sound DB (1.98 GB) ✔ DONE (21 Oct)

Downloaded `ICBHI_final_database.zip` (920 recordings, 126 patients) from the
official challenge site and unzipped to `data/icbhi/ICBHI_final_database/`.
The site's HTTPS certificate had expired, so `curl -k` was needed:

```bash
curl -k -o data/icbhi/ICBHI_final_database.zip https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip
cd data/icbhi && unzip ICBHI_final_database.zip
```

The loader searches `data/icbhi/` recursively, so the folder name inside does
not matter. Alternative source:

<https://www.kaggle.com/datasets/husninm/icbhi-2017-challenge>

Unpack so the layout is:

```
data/icbhi/audio_and_txt_files/101_1b1_Al_sc_Meditron.wav
data/icbhi/audio_and_txt_files/101_1b1_Al_sc_Meditron.txt
...
```

Each `.txt` sits next to its `.wav` and has one line per respiratory cycle:

```
start_seconds  end_seconds  crackles(0/1)  wheezes(0/1)
```

The loader cuts each cycle out using those timestamps — the labelled unit is
the cycle, not the file. Cycles marked as both crackle and wheeze are dropped,
since the synopsis defines a 3-class task.

---

## Downloading

Browser download works fine. If you would rather use the CLI:

```bash
pip install kaggle
# put kaggle.json in C:\Users\<you>\.kaggle\   (Kaggle > Account > Create New API Token)

kaggle datasets download -d bjoernjostein/physionet-challenge-2016 -p data/physionet2016 --unzip
kaggle datasets download -d husninm/icbhi-2017-challenge          -p data/icbhi       --unzip
```

Then check the folder names against the layouts above before running.

---

## Verifying the data is picked up

```bash
python -c "import sys; sys.path.insert(0,'.'); from src.datasets import load_domain; \
[print(d, len(load_domain(d, 5)[0]), load_domain(d, 5)[1]) for d in ('music','heart','respiratory')]"
```

Each line must print `real:<path>`. If it still says `synthetic`, the folder
name or nesting does not match — check the layouts above.

Then:

```bash
python run_experiment.py --no-cache
```

Expect roughly 20–40 minutes for the first full feature extraction on the real
audio (it is cached afterwards in `results/cache/`, so later runs are seconds).
Use `--limit 100` for a fast first pass over a subset.

---

## Software (already installed on this machine)

```
pip install -r requirements.txt     # numpy, scipy, scikit-learn, matplotlib
```

MATLAB needs the **Signal Processing Toolbox** (for `butter`, `filtfilt`,
`spectrogram`, `resample`, `stft`, `dct`). It does **not** need the Statistics
and Machine Learning Toolbox — `sa_classify.m` implements the k-NN lookup
directly against the model exported from Python, so the GUI runs on a base
MATLAB + Signal Processing install.

Nothing else needs downloading. `librosa` is deliberately not used, and there
is no deep-learning dependency.
