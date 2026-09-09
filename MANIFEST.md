# What is in this archive

## The project

```
README.md  MANIFEST.md  LICENSE  Dockerfile  docker-compose.yml  .dockerignore
.gitignore  .env.example  pytest.ini  requirements.txt
requirements-inference.txt  requirements-training.txt
configs/   src/   backend/   frontend/   scripts/   tests/   docs/
```

Plus the corpora that download without credentials, already fetched and
normalised, so the first run has real data to work with:

| Path | What it is |
|---|---|
| `data/raw/figlang2020_twitter/`, `figlang2020_reddit/`, `mustard/` | The three auto-downloadable corpora (9,400 + 690 rows) |
| `data/processed/dataset.jsonl` | 9,843 records in the common schema |
| `data/splits/` | The leakage-verified conversation-level 70/15/15 split |
| `results/dataset_report.json`, `results/dataset_acquisition_report.json` | The quality and acquisition reports for the above |
| `results/dev/` | Development-scale results — **read `results/dev/README.md` first** |
| `docs/*.png` | The three generated architecture diagrams |

## Not included (regenerated in about five minutes)

`frontend/node_modules`, the WordNet corpus, the FAISS index, the development
checkpoints and the retrieval cache are all machine-generated and are excluded
by `.gitignore`. One command rebuilds every one of them:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-training.txt

python scripts/quickstart.py
```

That runs, in order: dataset download + validation, corpus preparation, the
DEVELOPMENT/TEST-ONLY tiny encoder, a WordNet sample index, a ~1 minute
development training run, and the 11-check smoke test. Expect
`11 pass, 0 warn, 0 fail`.

Then:

```bash
pytest                                              # 324 tests
MODEL_PATH=checkpoints/dev_smoke/best_model uvicorn backend.main:app --port 8000
cd frontend && npm install && npm run dev           # http://localhost:5173
```

## Also not included, and why

* **News-Headlines** and **SARC** — licence-gated behind a Kaggle account.
  `scripts/download_datasets.py` prints the exact steps, expected filenames and
  target directories, then validates whatever you place there. See README §9.
* **A research knowledge index** — build it with
  `scripts/build_knowledge_index.py --download-conceptnet --source conceptnet wordnet`.
* **A research checkpoint** — there is none yet. Nothing in this archive claims
  a research result; the only trained artefacts are development-scale and are
  labelled as such everywhere they appear. Train the real thing with
  `python scripts/train.py --config configs/full_model.yaml` (README §11).
