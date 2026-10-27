# Contributors

This document records who contributed what to the SarcasmSense project, so that
contribution is visible in the repository even though the original development
history was collapsed into a single commit.

> **How to read this document.** The Git history contains exactly one commit
> (`cd5294e`), authored by Person A. Git cannot and does not establish who
> originally typed any particular line of code, because there is no separate
> per-person commit history. The attributions below are based on inspecting the
> implemented modules, their purpose, dependencies, and callers, and mapping
> them onto the agreed division of labour between Person A and Person B. Files
> marked *shared* contain work from both people or cross-cutting infrastructure
> that cannot be cleanly assigned.

## Agreed division of labour

| Area | Owner |
|---|---|
| Representation and backend — dataset integration and corpus adapters, data schema/preprocessing/tokenisation, dataset splitting/leakage prevention/corpus fingerprinting, reservoir sampling, knowledge-base construction, retrieval encoders/FAISS index/retrieval/caching, model architecture/encoder/cross-attention fusion/sarcasm model, FastAPI backend and API-serving orchestration | Person A |
| Learning, analysis, and interface — training pipeline/trainer/training scripts/hyperparameter configuration/checkpoints, evaluation metrics/plots/qualitative error analysis, explainability (occlusion, LIME, SHAP), incongruity detection, ablation experiments and their configuration, React/Vite frontend and user-facing analysis panels, tests/regression tests/coverage/benchmarking/deployment tooling where implemented by Person B | Person B |

## Person B — implemented components

The following components were implemented by Person B. Confidence reflects how
cleanly the file maps to Person B's scope after inspecting the implementation,
not git authorship metadata.

### Training pipeline

| File | Purpose | Evidence | Confidence |
|---|---|---|---|
| `src/training/trainer.py` | Training loop: fixed seed, gradient accumulation, mixed precision, linear warmup + linear decay, discriminative LRs (encoder vs head), gradient clipping, early stopping on validation macro-F1, checkpoint save/resume, per-epoch history logging | 561-line `Trainer` class; matches discriminative LRs, balanced class weights (via collator), gradient accumulation, warmup/decay, early stopping, macro-F1 | High |
| `src/training/pipeline.py` | Assembles an experiment from one `Config`: class weights, retrieval cache, collator/dataset assembly, split-fingerprinting integration | `build_experiment()` wires `compute_class_weights("balanced")`, `ensure_retrieval_cache`, and `SarcasmCollator`; orchestrates Person A's data/retrieval/model modules | High |
| `src/training/experiment.py` | Append-only experiment run log (`results/experiments.json` + CSV), run IDs, `git_commit` tracking | `ExperimentTracker.record()` appends JSON + CSV; `make_run_id()`, `model_type_from_config()` | High |

### Evaluation metrics and error analysis

| File | Purpose | Evidence | Confidence |
|---|---|---|---|
| `src/evaluation/metrics.py` | Macro-F1 primary metric, per-class precision/recall/F1, ROC/PR-AUC, grouped metrics per dataset/domain/context length | `PRIMARY_METRIC = "macro_f1"`, `compute_grouped_metrics()`, `roc_points()` | High |
| `src/evaluation/error_analysis.py` | Seven-category error analysis with explicit, priority-ordered, hand-checkable rules | `ErrorCategory` enum with seven categories; `categorise()` + `build_error_rows()` + `summarise_errors()` | High |

### Explainability and incongruity detection

| File | Purpose | Evidence | Confidence |
|---|---|---|---|
| `src/explainability/base.py` | `BaseExplainer`, `ExplanationResult`, `TokenAttribution`, `agreement()` (SHAP/LIME agreement), `normalise_weights()` | Used by all explainers and the SHAP-vs-LIME agreement study | High |
| `src/explainability/explainers.py` | Occlusion, LIME, SHAP explainers; honest fallback chain that always reports what actually ran | `OcclusionExplainer`, `LimeExplainer`, `ShapExplainer`, `ExplainerRegistry`; model-agnostic explainability | High |
| `src/explainability/explanation.py` | Human-readable explanation generation; incongruity detection via SentiWordNet polarity clash | `detect_incongruity()`, `IncongruitySignal`, `SentimentLexicon`, `generate_explanation()` | High |
| `src/inference/service.py` | Orchestrates predictor + explainers + explanation generator into one analysis service shared by API, benchmark, and error-analysis scripts | `SarcasmAnalysisService`; `analyze()` ties prediction to attribution and explanation | Medium-High |

### Training / evaluation / ablation / benchmarking scripts

| File | Purpose | Evidence | Confidence |
|---|---|---|---|
| `scripts/train.py` | Train one config; evaluates val/test; writes metrics, plots, predictions, experiment-tracking record | Orchestrates `Trainer` + `ExperimentTracker` + `plot_*` + CSV `results.csv` | High |
| `scripts/evaluate.py` | Evaluate a checkpoint; optional seven-category error analysis over a sampled split | Calls `Trainer.evaluate()` + `build_error_rows()` + `summarise_errors()` | High |
| `scripts/run_ablation.py` | Ten ablation variants plus SHAP-vs-LIME agreement study | `VARIANTS` dict (10 entries); `explanation_agreement_study()` | High |
| `scripts/benchmark_inference.py` | Measured CPU < 300 ms inference benchmark (retrieval + model, explanations separate) | `TARGET_MS = 300.0`; per-stage timing; `meets_300ms_target` | High |

### Hyperparameter configuration (training half)

| File | Person B settings | Person A settings | Confidence |
|---|---|---|---|
| `configs/*.yaml` | `learning_rate`/`head_learning_rate` (discriminative LRs), `gradient_accumulation_steps`, `warmup_ratio`, `early_stopping_patience`, `class_weights: balanced`, `metric_for_best: macro_f1`, `mixed_precision` | `encoder_name`, `fusion`, `use_context`, `use_knowledge`, `freeze_encoder_layers`, model architecture | Medium — mixed ownership; editing one person's half risks affecting the other |

### Frontend (React + Vite)

| File | Purpose | Evidence | Confidence |
|---|---|---|---|
| `frontend/src/App.jsx` | Root UI: Analyse / Model information / Performance dashboard tabs; health polling; backend-unavailable/degraded banners | Owns all cross-panel UI state; matches "user-facing analysis panels" | High |
| `frontend/src/api.js` | Typed API client with timeouts (incl. 180 s SHAP timeout), `ApiError` kinds, same-origin proxy via `VITE_API_BASE_URL` | SHAP-aware timeouts; no hard-coded `localhost` | High |
| `frontend/src/components/ResultPanel.jsx` | Renders verdict, confidence meter, explanation text, incongruity check, highlighted tokens, per-turn context influence, retrieved evidence, timings | Directly surfaces explainability + incongruity outputs | High |
| `frontend/src/components/DashboardPanel.jsx`, `InputPanel.jsx`, `ModelInfoPanel.jsx`, `common.jsx` | Dashboard, input form, model-info panel, shared UI primitives | User-facing panels | High |
| `frontend/src/styles.css`, `frontend/index.html`, `frontend/vite.config.js`, `frontend/Dockerfile`, `frontend/nginx.conf.template`, `frontend/package.json`, `frontend/package-lock.json`, `frontend/.env.example`, `frontend/.eslintrc.json` | Vite build, styling, container and proxy config, dependency lockfile | Frontend tooling and deployment | High |

### Tests for Person B's components

| File | Covers | Confidence |
|---|---|---|
| `tests/test_training.py` | Training loop, AMP, early stopping, checkpoint/resume, plots, experiment tracking | Medium-High |
| `tests/test_inference_and_explain.py` | Predictor, explainers, explanation generation, error analysis, agreement | Medium-High |
| `tests/test_api.py` | FastAPI contract incl. explainability metadata (`fusion`, `uses_knowledge`) | Medium — exercises both people's modules |

## Shared or ambiguous components (not attributed solely to Person B)

These files contain work from both people, cross-cutting infrastructure, or
integration glue, and are deliberately **not** assigned to Person B alone:

- `src/inference/predictor.py` — Person A's model forward path wrapped with retrieval/tokenisation; shared with Person B's service layer
- `src/utils/*` (`config.py`, `io.py`, `logging_utils.py`, `paths.py`, `seeding.py`) — shared infrastructure
- `tests/conftest.py`, `tests/test_model.py` — fixtures and model-shape tests spanning both modules
- `configs/*.yaml` — mixed ownership (see above)
- `README.md`, `MANIFEST.md`, `docs/*`, `Dockerfile`, `docker-compose.yml`, `requirements*.txt`, `pytest.ini`, `.gitignore` — project documentation, configuration, and deployment shared by both
- `src/data/*`, `src/retrieval/*`, `src/models/*`, `backend/*`, `scripts/download_datasets.py`, `scripts/prepare_datasets.py`, `scripts/build_knowledge_index.py`, `scripts/make_diagrams.py`, `scripts/make_dev_assets.py`, `scripts/quickstart.py`, `scripts/smoke_test.py`, `scripts/find_demo_examples.py` — Person A's representation/backend scope and shared glue

## What this document does not claim

- Git does not prove who originally typed any line. The single commit is by Person A; attributions above are based on module inspection against the agreed division of labour.
- No Person A work is attributed to Person B.
- No source code is duplicated here; this document only records attribution.
- This document does not change any tracked source file, configuration, or generated artefact.

## Verification

- Original commit: `cd5294e` (`Context-aware explainable sarcasm detection framework`, author Person A, no parent) — preserved unchanged.
- This document is a new, additive, follow-up record; the project's final tracked code tree changes only by the addition of this file.
