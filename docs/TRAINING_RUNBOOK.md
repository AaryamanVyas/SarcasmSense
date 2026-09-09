# Complete training runbook

Every command needed to go from a fresh clone to four trained models, an
ablation table, an error analysis and a running web application — with the full
~63k corpus.

Run everything **from the project root** with your virtual environment active.

---

## Step 0 — Prerequisites

### Python version matters — check it first

```bash
python scripts/check_environment.py
```

This needs no dependencies and runs on any interpreter. **Do this before
`pip install`**, because the failure mode otherwise is deeply misleading: pip
reports a missing Fortran compiler while building scipy, when the actual problem
is that your Python is too new for the pinned wheels.

| Python | Status |
|---|---|
| 3.10, 3.11, 3.12 | Supported — **3.11 recommended** |
| 3.13, 3.14, ... | **Not supported.** numpy 1.26 / scipy 1.14 / torch 2.5 / faiss-cpu 1.9 publish no wheels; pip tries to compile from source and fails |
| 3.9 and older | Not supported |

Installing a compiler does **not** fix a too-new interpreter — `torch` has no
source build for 3.13+ either. Install Python 3.11 alongside whatever you have;
they coexist, and the virtual environment keeps this project pinned to it
without changing your system default.

### Windows (PowerShell)

```powershell
# 1. Install Python 3.11 next to your existing Python
winget install -e --id Python.Python.3.11
#    (or download the 64-bit installer from python.org)

# 2. Close and reopen PowerShell, then from the project root:
cd C:\Users\<you>\Downloads\sarcasm-framework
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
#    If that is blocked by execution policy:
#      Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Your prompt must now start with (.venv). Verify, then install:
python scripts/check_environment.py
python -m pip install --upgrade pip
pip install -r requirements-training.txt
```

### macOS / Linux

```bash
# macOS:         brew install python@3.11
# Debian/Ubuntu: sudo apt install python3.11 python3.11-venv

cd sarcasm-framework
python3.11 -m venv .venv
source .venv/bin/activate

python scripts/check_environment.py
python -m pip install --upgrade pip
pip install -r requirements-training.txt
```

### The virtual environment is not optional

If pip prints `Defaulting to user installation because normal site-packages is
not writeable`, you are **not** in the venv. Activate it and start over —
otherwise packages scatter into your user directory, version conflicts become
very hard to diagnose, and you risk breaking other Python projects on the
machine.

You are in the venv when your prompt starts with `(.venv)` and:

```bash
python -c "import sys; print(sys.prefix != sys.base_prefix)"   # must print True
```

### NVIDIA GPU

Install the CUDA build of PyTorch **first**, then the requirements:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-training.txt
```

Check what you got:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### Hardware

| | Minimum | Comfortable |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Disk | 15 GB | 30 GB (a full ConceptNet index is ~8 GB while building) |
| GPU | none (CPU works, slowly) | 16 GB T4 / L4 / RTX 3090 |

---

## Step 1 — Place the two manual datasets

The other two corpora download themselves. These two are licence-gated, so you
place them by hand. **The filenames matter** — the loaders look for these exact
names.

```
sarcasm-framework/
└── data/
    └── raw/
        ├── news_headlines/
        │   └── Sarcasm_Headlines_Dataset_v2.json      ← News-Headlines (v1 also accepted)
        └── sarc/
            └── train-balanced-sarcasm.csv             ← SARC, Kaggle layout
```

Create the directories and copy the files:

```bash
mkdir -p data/raw/news_headlines data/raw/sarc

cp ~/Downloads/Sarcasm_Headlines_Dataset_v2.json  data/raw/news_headlines/
cp ~/Downloads/train-balanced-sarcasm.csv         data/raw/sarc/
```

Windows PowerShell:

```powershell
mkdir data\raw\news_headlines, data\raw\sarc -Force
copy $HOME\Downloads\Sarcasm_Headlines_Dataset_v2.json data\raw\news_headlines\
copy $HOME\Downloads\train-balanced-sarcasm.csv        data\raw\sarc\
```

### Accepted variants

| You have | Put it at | Note |
|---|---|---|
| `Sarcasm_Headlines_Dataset_v2.json` | `data/raw/news_headlines/` | 28,619 rows — preferred |
| `Sarcasm_Headlines_Dataset.json` (v1) | `data/raw/news_headlines/` | 26,709 rows — also fine |
| `train-balanced-sarcasm.csv` (Kaggle) | `data/raw/sarc/` | 1.01M rows, one parent turn |
| `train-balanced.csv` **+** `comments.json` (Princeton) | `data/raw/sarc/` | Full ancestor chains — up to 3 context turns. **Both files required.** |

Do **not** rename anything else, do not nest the files in a subfolder, and do
not leave them zipped.

---

## Step 2 — Verify the files parse before doing anything else

```bash
python scripts/download_datasets.py
```

This fetches FigLang + MUStARD automatically and validates all four. You want to
see four `──` report blocks and no `✗ NOT AVAILABLE`. Each block reports the row
count, class balance, duplicates, missing values and context availability, and
checks the row count against the published figure:

```
── News Headlines Dataset for Sarcasm Detection ────────────────
  rows                 : 28,619
  class distribution   : Not Sarcastic=16,135 (56.4%), Sarcastic=12,484 (43.6%)
  published row count  : 28,619 (loaded 28,619, delta +0) -> OK
```

If a corpus shows `NOT AVAILABLE`, the script prints the exact path it looked
for. Fix the path and re-run — everything downstream depends on this passing.

---

## Step 3 — Build the corpus and the split

```bash
python scripts/prepare_datasets.py --sarc-max-rows 25000
```

**Measured: ~75 seconds** for a 63k corpus on 2 CPU cores.

`--sarc-max-rows 25000` caps SARC by **seeded random sampling** so its 1.01M
rows don't drown out the other three corpora. The cap is a real sample, not the
first N rows — SARC's CSV is roughly date-ordered, so a head slice would quietly
restrict you to the earliest slice of Reddit history. Use `--sarc-max-rows 0`
for no cap (expect a ~1.05M-row corpus and much longer training).

Expected output:

```
  ✓ figlang2020_twitter      5,000 records
  ✓ figlang2020_reddit       4,400 records
  ✓ mustard                    690 records
  ✓ news_headlines          28,619 records
  ✓ sarc                    25,000 records   {... 'sampled_from': 1010826 ...}

  corpus: 63,709 -> 63,462 records            (duplicates removed)
  with context         : 34,843 (54.9%)
  per-domain           : dialogue=690, news=28,619, reddit=29,400, twitter=4,753

  train n= 44,424   val n=  9,521   test n=  9,517

Leakage check: PASS
```

**The leakage check must say PASS.** The script exits non-zero if it doesn't;
do not train on a failing split.

Writes: `data/processed/dataset.jsonl`, `data/splits/`, `results/dataset_report.json`.

---

## Step 4 — Build the commonsense knowledge index

Pick one. The index is built **once** and reused by every training run.

**Option A — WordNet only** (no large download, ~5 minutes, ~250k facts):

```bash
python scripts/build_knowledge_index.py --download-wordnet --source wordnet
```

**Option B — ConceptNet** (~1.2 GB download, 30–60 minutes, the canonical source):

```bash
python scripts/build_knowledge_index.py --download-conceptnet --source conceptnet wordnet
```

**Option C — ConceptNet + ATOMIC/COMET** (the full proposed configuration):

```bash
# ATOMIC-2020 from https://allenai.org/data/atomic-2020, unzipped to knowledge/raw/atomic2020/
python scripts/build_knowledge_index.py \
    --source conceptnet atomic wordnet \
    --atomic-tsv knowledge/raw/atomic2020/train.tsv
```

Option B is the sensible default for a project submission — it is what the
proposal specifies and what the citations refer to.

The script ends with a sanity probe. Check that the facts it prints are actually
relevant to the demo utterance; if they are nonsense, the index is broken and
you should not proceed:

```
  sanity probe -> top-3 facts for a demo utterance:
    +0.612  [ConceptNet/Causes] a computer crash causes data loss
```

Also confirm `encoder_is_fallback: False` in the summary. If it says `True`,
`all-MiniLM-L6-v2` failed to download and retrieval will be lexical rather than
semantic — fix your network and rebuild before training.

---

## Step 5 — Train the four systems

Run all four. The comparison between them *is* the result; the full model's
number alone proves nothing.

```bash
python scripts/train.py --config configs/baseline_roberta.yaml   # Baseline 1
python scripts/train.py --config configs/context_model.yaml      # Baseline 2
python scripts/train.py --config configs/knowledge_model.yaml    # Baseline 3
python scripts/train.py --config configs/full_model.yaml         # PROPOSED MODEL
```

The first run downloads `roberta-base` (~500 MB) and `all-MiniLM-L6-v2`
(~90 MB) and builds the retrieval cache (**measured: ~47 s for 63k records**);
later runs reuse both.

### How long

The proposal's planning estimate was 2–4 GPU-hours for the context model and 3–5
more for the retrieval+fusion model. **Treat those as estimates, not promises** —
your actual wall-clock time is measured and written to
`results/experiments.json` as `train_seconds`. What I can tell you concretely:
44,424 training examples at batch 16 is 2,777 optimiser steps per epoch, and the
default schedule is 4 epochs with early stopping.

### If you run out of GPU memory

```bash
python scripts/train.py --config configs/full_model.yaml \
    --override training.batch_size=8 training.gradient_accumulation_steps=4
```

That keeps the effective batch at 32 while halving peak memory. Still tight?
Add `model.freeze_encoder_layers=6`.

### Other useful overrides

```bash
--override training.epochs=6                 # train longer
--override data.max_train_samples=5000       # quick trial run on a subset
--device cuda:1                              # pick a GPU
--seed 1337                                  # a second seed, for a variance check
--skip-test                                  # leave the test split untouched
```

Each run writes `checkpoints/<name>/best_model/` (weights + tokenizer + the
resolved config), `history.json`, `metrics.json`, figures under
`results/figures/`, and appends a row to `results/experiments.json`.

**Watch for warnings.** The script prints a `⚠` block at startup for anything
degraded — a sample index, a fallback encoder, a development checkpoint. A clean
research run should print none.

---

## Step 6 — Evaluate and analyse errors

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/full_model/best_model \
    --split test --error-analysis --error-analysis-samples 300
```

Gives accuracy, macro-F1, per-class precision/recall, confusion matrix, ROC-AUC —
overall and broken down **per dataset**, **per domain** and **per context
length**. The last one is the interesting table: it shows whether the model
actually benefits from having 1, 2 or 3 turns of context.

Writes `results/error_analysis_full_model_test.csv`, one row per example with
the text, context, both labels, confidence, retrieved evidence, top tokens,
attention shares, the generated explanation, and the assigned error category.
That CSV is the single most useful artefact for the "error analysis" section of
a report.

---

## Step 7 — Ablation study

```bash
python scripts/run_ablation.py --base configs/full_model.yaml --explanation-agreement
```

Trains all ten variants and prints one comparison table. **This retrains ten
models** — start it when you can leave it alone. To trim it:

```bash
python scripts/run_ablation.py --base configs/full_model.yaml \
    --variants baseline1_no_context_no_knowledge baseline2_context_only \
               baseline3_knowledge_only full_cross_attention fusion_concat \
    --skip-existing
```

`--skip-existing` reuses checkpoints you already trained in Step 5 instead of
redoing them.

`--explanation-agreement` additionally runs SHAP against LIME on a fixed-seed
sample of the test set and reports how much they agree.

Writes `results/ablation_results.{json,csv}` and
`results/figures/ablation_comparison.png`.

---

## Step 8 — Benchmark

```bash
python scripts/benchmark_inference.py \
    --checkpoint checkpoints/full_model/best_model \
    --device cpu --runs 50 --explanation-methods occlusion lime shap
```

Use `--device cpu` explicitly: the `<300 ms` target in the proposal is a **CPU**
target, and the script only prints MET/NOT MET when it actually measured on CPU.

---

## Step 9 — Run the application

Two terminals:

```bash
# terminal 1
MODEL_PATH=checkpoints/full_model/best_model uvicorn backend.main:app --port 8000

# terminal 2
cd frontend && npm install && npm run dev
```

Open <http://localhost:5173>. Check the **Model information** tab first — it
should show your encoder, your index's fact count, and **no** red
"development/test checkpoint" banner. The **Performance dashboard** tab will
now show your real runs from `results/experiments.json`.

Or with Docker:

```bash
cp .env.example .env      # set MODEL_PATH=/app/checkpoints/full_model/best_model
docker compose up --build
```

---

## The whole thing, condensed

```bash
# once
pip install -r requirements-training.txt
mkdir -p data/raw/news_headlines data/raw/sarc
cp ~/Downloads/Sarcasm_Headlines_Dataset_v2.json data/raw/news_headlines/
cp ~/Downloads/train-balanced-sarcasm.csv        data/raw/sarc/

python scripts/download_datasets.py
python scripts/prepare_datasets.py --sarc-max-rows 25000
python scripts/build_knowledge_index.py --download-conceptnet --source conceptnet wordnet

# the experiments
python scripts/train.py --config configs/baseline_roberta.yaml
python scripts/train.py --config configs/context_model.yaml
python scripts/train.py --config configs/knowledge_model.yaml
python scripts/train.py --config configs/full_model.yaml

python scripts/evaluate.py --checkpoint checkpoints/full_model/best_model --error-analysis
python scripts/run_ablation.py --base configs/full_model.yaml --skip-existing --explanation-agreement
python scripts/benchmark_inference.py --checkpoint checkpoints/full_model/best_model --device cpu

# the app
MODEL_PATH=checkpoints/full_model/best_model uvicorn backend.main:app --port 8000
cd frontend && npm install && npm run dev
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ERROR: Unknown compiler(s): ... gfortran ...` while building scipy | Python too new (3.13+); no wheels, so pip compiles from source | Use Python 3.10-3.12. `python scripts/check_environment.py` diagnoses it. Installing a Fortran compiler will NOT help |
| `Defaulting to user installation because normal site-packages is not writeable` | Not inside a virtual environment | `py -3.11 -m venv .venv` then `.\\.venv\\Scripts\\Activate.ps1` (Windows) or `source .venv/bin/activate` |
| `Microsoft Visual C++ 14.0 or greater is required` | Same root cause: no wheel for your Python | Same fix: use Python 3.10-3.12 |
| `NOT AVAILABLE` in Step 2 | Wrong filename or directory | The script prints the exact path it looked for — match it |
| `LEAKAGE DETECTED` in Step 3 | Should not happen with the shipped splitter | Do not train; open an issue with `results/dataset_report.json` |
| `Saved split does not match the corpus` | You re-ran `prepare_datasets.py` after training, or edited the corpus | Re-run `prepare_datasets.py`, then retrain — the split is fingerprinted on purpose |
| `encoder_is_fallback: True` | `all-MiniLM-L6-v2` didn't download | Fix network, delete `knowledge/index/`, rebuild |
| `is_sample_index: true` warning | You built with `--sample` | Rebuild without `--sample` |
| `Retrieval cache does not match` | You rebuilt the index after caching | Delete `data/processed/*.npz`, or pass `--rebuild-retrieval-cache` |
| CUDA out of memory | Batch too large | `--override training.batch_size=8 training.gradient_accumulation_steps=4` |
| `Checkpoint directory not found` on the API | `MODEL_PATH` wrong | Point it at `checkpoints/<name>/best_model` (not the parent) |
| UI shows a red dev-checkpoint banner | Serving a development checkpoint | Set `MODEL_PATH` to a real trained one |
| SHAP request times out in the UI | Expected — SHAP is audit-mode | Use LIME or token occlusion for interactive use |

---

## Google Colab

```python
!git clone <your-repo-url> && %cd sarcasm-framework
!pip install -q -r requirements-training.txt

from google.colab import files          # upload the two Kaggle files
!mkdir -p data/raw/news_headlines data/raw/sarc
uploaded = files.upload()
!mv Sarcasm_Headlines_Dataset_v2.json data/raw/news_headlines/
!mv train-balanced-sarcasm.csv        data/raw/sarc/

!python scripts/download_datasets.py
!python scripts/prepare_datasets.py --sarc-max-rows 25000
!python scripts/build_knowledge_index.py --download-wordnet --source wordnet
!python scripts/train.py --config configs/full_model.yaml
```

Colab sessions disconnect. Copy `checkpoints/` and `results/` to Drive when a
run finishes, and use `--override training.resume_from=checkpoints/full_model/last`
to pick up an interrupted run.
