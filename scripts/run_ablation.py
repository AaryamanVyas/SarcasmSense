#!/usr/bin/env python
"""Run the ablation study and print a single clean comparison table.

Variants (all eight required by the specification, plus the four systems)::

    baseline1_no_context_no_knowledge   context OFF, commonsense OFF
    baseline2_context_only              context ON,  commonsense OFF
    baseline3_knowledge_only            context OFF, commonsense ON
    full_cross_attention                context ON,  commonsense ON, cross-attention
    fusion_concat                       context ON,  commonsense ON, concatenation
    topk_3                              top-3 retrieved evidence
    topk_5                              top-5 retrieved evidence
    context_1_turn / context_2_turns / context_3_turns

Every variant is trained from the same base config with only the listed
overrides changed, on the same split, with the same seed. Results are appended
to ``results/ablation_results.json`` and rendered as a table.

Examples
--------
    python scripts/run_ablation.py --base configs/full_model.yaml
    python scripts/run_ablation.py --base configs/dev_smoke.yaml \\
        --variants full_cross_attention fusion_concat --epochs 1
    python scripts/run_ablation.py --base configs/full_model.yaml \\
        --explanation-agreement-only --checkpoint checkpoints/full_model/best_model
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import MetricResult, render_comparison_table  # noqa: E402
from src.evaluation.plots import plot_model_comparison  # noqa: E402
from src.explainability.base import agreement  # noqa: E402
from src.inference.service import SarcasmAnalysisService  # noqa: E402
from src.models.dev_assets import is_development_asset  # noqa: E402
from src.training.pipeline import build_experiment  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.io import write_json  # noqa: E402
from src.utils.logging_utils import configure_logging  # noqa: E402
from src.utils.paths import ensure_dir, resolve  # noqa: E402
from src.utils.seeding import set_global_seed  # noqa: E402

logger = logging.getLogger("run_ablation")
RULE = "=" * 78

#: ``variant name -> (description, dotted config overrides)``
VARIANTS: dict[str, tuple[str, list[str]]] = {
    "baseline1_no_context_no_knowledge": (
        "Ablation 1+2 combined: context OFF, commonsense OFF (Baseline 1)",
        ["model.use_context=false", "model.use_knowledge=false", "model.fusion=none",
         "retrieval.enabled=false"],
    ),
    "baseline2_context_only": (
        "Ablation 2: commonsense OFF, context ON (Baseline 2)",
        ["model.use_context=true", "model.use_knowledge=false",
         "model.fusion=cross_attention", "retrieval.enabled=false"],
    ),
    "baseline3_knowledge_only": (
        "Ablation 1: context OFF, commonsense ON (Baseline 3)",
        ["model.use_context=false", "model.use_knowledge=true",
         "model.fusion=cross_attention", "retrieval.enabled=true"],
    ),
    "full_cross_attention": (
        "Ablation 3+5: context ON, commonsense ON, cross-attention fusion (proposed)",
        ["model.use_context=true", "model.use_knowledge=true",
         "model.fusion=cross_attention", "retrieval.enabled=true"],
    ),
    "fusion_concat": (
        "Ablation 4: concatenation fusion instead of cross-attention",
        ["model.use_context=true", "model.use_knowledge=true",
         "model.fusion=concat", "retrieval.enabled=true"],
    ),
    "topk_3": (
        "Ablation 6: top-3 retrieved evidence",
        ["model.use_context=true", "model.use_knowledge=true",
         "model.fusion=cross_attention", "retrieval.enabled=true", "retrieval.top_k=3"],
    ),
    "topk_5": (
        "Ablation 7: top-5 retrieved evidence",
        ["model.use_context=true", "model.use_knowledge=true",
         "model.fusion=cross_attention", "retrieval.enabled=true", "retrieval.top_k=5"],
    ),
    "context_1_turn": (
        "Ablation 8: context truncated to the single most recent turn",
        ["model.use_context=true", "model.use_knowledge=true",
         "model.fusion=cross_attention", "data.max_context_turns=1"],
    ),
    "context_2_turns": (
        "Ablation 8: context truncated to the two most recent turns",
        ["model.use_context=true", "model.use_knowledge=true",
         "model.fusion=cross_attention", "data.max_context_turns=2"],
    ),
    "context_3_turns": (
        "Ablation 8: full three-turn context",
        ["model.use_context=true", "model.use_knowledge=true",
         "model.fusion=cross_attention", "data.max_context_turns=3"],
    ),
}

DEFAULT_VARIANTS = [
    "baseline1_no_context_no_knowledge",
    "baseline2_context_only",
    "baseline3_knowledge_only",
    "full_cross_attention",
    "fusion_concat",
    "topk_3",
    "topk_5",
    "context_1_turn",
    "context_2_turns",
    "context_3_turns",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", default="configs/full_model.yaml")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    parser.add_argument("--output", default="results/ablation_results.json")
    parser.add_argument("--csv", default="results/ablation_results.csv")
    parser.add_argument("--checkpoints-root", default="checkpoints/ablation")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="key.sub=value",
        help=(
            "Extra dotted config overrides applied to EVERY variant. Applied "
            "before the variant's own overrides, so a variant's defining setting "
            "always wins."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse a variant's checkpoint if it already exists instead of retraining.",
    )
    parser.add_argument(
        "--explanation-agreement",
        action="store_true",
        help="Also run the SHAP-vs-LIME agreement study on a controlled sample.",
    )
    parser.add_argument("--explanation-agreement-only", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for the agreement study.")
    parser.add_argument("--agreement-samples", type=int, default=30)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def run_variant(
    name: str,
    base_config_path: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Train and evaluate one ablation variant; returns its result payload."""
    description, overrides = VARIANTS[name]
    # Shared overrides first so a variant's own defining settings always win.
    all_overrides = [*args.override, *overrides]
    if args.epochs is not None:
        all_overrides.append(f"training.epochs={args.epochs}")
    output_dir = f"{args.checkpoints_root}/{name}"
    all_overrides.append(f"experiment.output_dir={output_dir}")
    all_overrides.append(f"experiment.name=ablation_{name}")
    all_overrides.append(f"experiment.seed={args.seed}")

    config = Config.from_yaml(base_config_path, overrides=all_overrides)
    set_global_seed(config.experiment.seed, deterministic=config.experiment.deterministic)

    print(f"\n{RULE}\nVARIANT: {name}\n  {description}\n  overrides: {overrides}\n{RULE}")

    best_dir = resolve(output_dir) / "best_model"
    context = build_experiment(
        config, device=args.device, include_train=True, local_files_only=args.local_files_only
    )
    trainer = Trainer(
        model=context.model,
        config=config,
        collator=context.collator,
        train_dataset=context.datasets["train"],
        val_dataset=context.datasets["val"],
        device=args.device,
        output_dir=ensure_dir(output_dir),
    )

    train_seconds = 0.0
    history: dict[str, Any] = {}
    if args.skip_existing and best_dir.exists():
        print(f"  reusing existing checkpoint {best_dir}")
    else:
        started = time.perf_counter()
        history = trainer.train().as_dict()
        train_seconds = time.perf_counter() - started

    if best_dir.exists():
        from src.models.sarcasm_model import SarcasmModel

        trainer.model = SarcasmModel.from_pretrained(
            best_dir, map_location=str(trainer.device),
            local_files_only=args.local_files_only,
        ).to(trainer.device)

    val_metrics, _ = trainer.evaluate(context.datasets["val"])
    test_metrics, _ = trainer.evaluate(context.datasets["test"])
    print(f"  val : {val_metrics.summary()}")
    print(f"  test: {test_metrics.summary()}")

    return {
        "variant": name,
        "description": description,
        "overrides": overrides,
        "config": config.to_dict(),
        "checkpoint": str(best_dir),
        "val": val_metrics.as_dict(),
        "test": test_metrics.as_dict(),
        "train_seconds": round(train_seconds, 2),
        "history": history,
        "warnings": context.warnings,
        "is_development_asset": is_development_asset(config.model.encoder_name),
        "_test_metric_object": test_metrics,
    }


def explanation_agreement_study(
    checkpoint: str,
    config_path: str | None,
    n_samples: int,
    seed: int,
    device: str | None,
) -> dict[str, Any]:
    """Compare SHAP and LIME attributions on a controlled sample.

    The sample is drawn from the **test split** with a fixed seed so the study is
    reproducible, and both explainers see exactly the same inputs, the same
    retrieved evidence and the same model.
    """
    print(f"\n{RULE}\nEXPLANATION AGREEMENT STUDY (SHAP vs LIME)\n{RULE}")
    config = Config.from_yaml(config_path or (resolve(checkpoint) / "training_config.yaml"))
    service = SarcasmAnalysisService.from_checkpoint(checkpoint, device=device, config=config)
    availability = service.explainers.availability()
    if not (availability.get("shap") and availability.get("lime")):
        message = f"Both SHAP and LIME are required; availability={availability}"
        print(f"  ⚠  {message}")
        return {"available": False, "reason": message, "availability": availability}

    context = build_experiment(config, device=device, include_train=False)
    rng = random.Random(seed)
    pool = context.splits["test"]
    sample = rng.sample(pool, min(n_samples, len(pool)))
    print(f"  sample: {len(sample)} test examples (seed={seed})")

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(sample, start=1):
        lime_result = service.analyze(
            record.text, record.context, explanation_method="lime"
        )
        shap_result = service.analyze(
            record.text, record.context, explanation_method="shap"
        )
        scores = agreement(lime_result.explanation, shap_result.explanation, top_k=5)
        rows.append(
            {
                "id": record.id,
                "agreement": scores,
                "lime_top": [t.token for t in lime_result.explanation.top(5, False)],
                "shap_top": [t.token for t in shap_result.explanation.top(5, False)],
                "lime_ms": round(lime_result.explanation.latency_ms, 1),
                "shap_ms": round(shap_result.explanation.latency_ms, 1),
                "same_prediction": lime_result.prediction.label == shap_result.prediction.label,
            }
        )
        if index % 5 == 0:
            print(f"    {index}/{len(sample)}")

    import numpy as np

    def _mean(key: str) -> float:
        values = np.array([r["agreement"][key] for r in rows], dtype=float)
        values = values[~np.isnan(values)]
        return float(values.mean()) if values.size else float("nan")

    summary = {
        "available": True,
        "n_samples": len(rows),
        "seed": seed,
        "mean_jaccard_top5": round(_mean("jaccard_topk"), 4),
        "mean_rank_correlation": round(_mean("rank_correlation"), 4),
        "mean_sign_agreement": round(_mean("sign_agreement"), 4),
        "mean_lime_ms": round(float(np.mean([r["lime_ms"] for r in rows])), 1),
        "mean_shap_ms": round(float(np.mean([r["shap_ms"] for r in rows])), 1),
        "rows": rows,
    }
    print(
        f"\n  mean Jaccard(top-5)     : {summary['mean_jaccard_top5']:.4f}\n"
        f"  mean rank correlation   : {summary['mean_rank_correlation']:.4f}\n"
        f"  mean sign agreement     : {summary['mean_sign_agreement']:.4f}\n"
        f"  mean latency LIME / SHAP: {summary['mean_lime_ms']:.1f} ms / "
        f"{summary['mean_shap_ms']:.1f} ms"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a POSIX exit code."""
    args = parse_args(argv)
    configure_logging(args.log_level)

    payload: dict[str, Any] = {"base_config": args.base, "seed": args.seed, "variants": []}

    if not args.explanation_agreement_only:
        results: dict[str, MetricResult] = {}
        for name in args.variants:
            try:
                outcome = run_variant(name, args.base, args)
            except Exception as exc:  # noqa: BLE001 - one bad variant must not kill the sweep
                logger.exception("Variant %s failed", name)
                payload["variants"].append({"variant": name, "error": str(exc)})
                continue
            results[name] = outcome.pop("_test_metric_object")
            payload["variants"].append(outcome)

        if results:
            print(f"\n{RULE}\nABLATION RESULTS (test split)\n{RULE}")
            print(render_comparison_table(results))
            # Figures go next to the results file, so a development sweep writing
            # to results/dev/ does not drop plots into the research results tree.
            figures = ensure_dir(Path(args.output).parent / "figures")
            plot_model_comparison(
                results, figures / "ablation_comparison.png",
                title="Ablation study (test set)",
            )
            csv_path = ensure_dir(Path(args.csv).parent) / Path(args.csv).name
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["variant", "description", "n", "accuracy", "macro_f1",
                     "sarcastic_precision", "sarcastic_recall", "roc_auc", "train_seconds"]
                )
                for entry in payload["variants"]:
                    if "error" in entry:
                        continue
                    test = entry["test"]
                    writer.writerow(
                        [
                            entry["variant"], entry["description"], test["n_examples"],
                            round(test["accuracy"], 4), round(test["macro_f1"], 4),
                            round(test["sarcastic_precision"], 4),
                            round(test["sarcastic_recall"], 4),
                            round(test["roc_auc"], 4) if test["roc_auc"] is not None else "",
                            entry["train_seconds"],
                        ]
                    )
            print(f"\n  table -> {csv_path}")

    if args.explanation_agreement or args.explanation_agreement_only:
        checkpoint = args.checkpoint
        if checkpoint is None:
            successful = [v for v in payload["variants"] if "checkpoint" in v]
            checkpoint = successful[-1]["checkpoint"] if successful else None
        if checkpoint is None:
            print("  ⚠  No checkpoint available for the agreement study; pass --checkpoint.")
        else:
            payload["explanation_agreement"] = explanation_agreement_study(
                checkpoint, None, args.agreement_samples, args.seed, args.device
            )

    path = write_json(args.output, payload)
    print(f"\n  full results -> {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
