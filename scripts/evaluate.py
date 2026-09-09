#!/usr/bin/env python
"""Evaluate a trained checkpoint and (optionally) run the error analysis.

Examples
--------
    python scripts/evaluate.py --checkpoint checkpoints/full_model/best_model
    python scripts/evaluate.py --checkpoint checkpoints/full_model/best_model \\
        --split test --error-analysis --error-analysis-samples 300

Bulk metrics use the batched trainer path (fast). The error analysis re-runs a
*sample* of the split through the full inference service so that every exported
row carries the same retrieval, attention and explanation information a user
would see in the web app -- which is what makes the categories checkable by hand.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.error_analysis import (  # noqa: E402
    CATEGORY_DESCRIPTIONS,
    build_error_rows,
    summarise_errors,
)
from src.evaluation.metrics import compute_grouped_metrics, render_comparison_table  # noqa: E402
from src.evaluation.plots import plot_confusion_matrix, plot_metric_by_group, plot_roc  # noqa: E402
from src.inference.service import SarcasmAnalysisService  # noqa: E402
from src.models.dev_assets import is_development_asset  # noqa: E402
from src.training.pipeline import build_experiment  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.io import write_json  # noqa: E402
from src.utils.logging_utils import configure_logging  # noqa: E402
from src.utils.paths import ensure_dir, resolve  # noqa: E402
from src.utils.seeding import set_global_seed  # noqa: E402

logger = logging.getLogger("evaluate")
RULE = "=" * 78


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory.")
    parser.add_argument("--config", default=None, help="Override the bundled config.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--error-analysis", action="store_true")
    parser.add_argument(
        "--error-analysis-samples",
        type=int,
        default=300,
        help="Examples to run through the full explanation path (0 = all).",
    )
    parser.add_argument(
        "--explanation-method", default="occlusion", choices=["lime", "shap", "occlusion"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a POSIX exit code."""
    args = parse_args(argv)
    configure_logging(args.log_level)
    set_global_seed(args.seed)

    checkpoint = resolve(args.checkpoint)
    config_path = args.config or (checkpoint / "training_config.yaml")
    if not Path(config_path).exists():
        print(f"Config not found at {config_path}; pass --config explicitly.")
        return 1
    config = Config.from_yaml(config_path)

    print(f"\n{RULE}\nEVALUATION: {checkpoint.name}\n{RULE}")
    dev_asset = is_development_asset(config.model.encoder_name) or is_development_asset(checkpoint)
    if dev_asset:
        print(
            "  ⚠  DEVELOPMENT/TEST checkpoint: the numbers below describe the "
            "software pipeline, not the proposed architecture.\n"
        )

    context = build_experiment(config, device=args.device, include_train=False)
    for warning in context.warnings:
        print(f"  ⚠  {warning}")

    from src.models.sarcasm_model import SarcasmModel

    model = SarcasmModel.from_pretrained(checkpoint, map_location="cpu")
    trainer = Trainer(
        model=model,
        config=config,
        collator=context.collator,
        device=args.device,
        output_dir=checkpoint.parent,
    )
    dataset = context.datasets[args.split]
    records = context.splits[args.split]
    print(f"  split      : {args.split} ({len(dataset):,} examples)")

    metrics, predictions = trainer.evaluate(dataset)
    print(f"\n  overall    : {metrics.summary()}")

    by_id = {record.id: record for record in records}
    datasets_of = [by_id[i].dataset for i in predictions.record_ids]
    domains_of = [by_id[i].domain for i in predictions.record_ids]
    context_of = [f"{min(len(by_id[i].context), 3)}_turns" for i in predictions.record_ids]

    per_dataset = compute_grouped_metrics(
        predictions.y_true, predictions.y_pred, datasets_of, predictions.y_prob_sarcastic
    )
    per_domain = compute_grouped_metrics(
        predictions.y_true, predictions.y_pred, domains_of, predictions.y_prob_sarcastic
    )
    per_context = compute_grouped_metrics(
        predictions.y_true, predictions.y_pred, context_of, predictions.y_prob_sarcastic
    )
    print("\n  per dataset:")
    print("   " + render_comparison_table(per_dataset).replace("\n", "\n   "))
    print("\n  per context length:")
    print("   " + render_comparison_table(per_context).replace("\n", "\n   "))

    output_dir = ensure_dir(args.output_dir)
    figures = ensure_dir(output_dir / "figures")
    stem = f"{config.experiment.name}_{args.split}"
    plot_confusion_matrix(
        metrics.confusion_matrix, figures / f"{stem}_confusion_matrix.png",
        title=f"{config.experiment.name}: confusion matrix ({args.split})",
    )
    plot_roc(
        predictions.y_true, predictions.y_prob_sarcastic, figures / f"{stem}_roc.png",
        title=f"{config.experiment.name}: ROC ({args.split})",
    )
    plot_metric_by_group(
        per_dataset, figures / f"{stem}_macro_f1_by_dataset.png",
        title=f"{config.experiment.name}: macro-F1 by dataset ({args.split})",
    )

    payload: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "split": args.split,
        "is_development_asset": dev_asset,
        "warnings": context.warnings,
        "overall": metrics.as_dict(),
        "per_dataset": {k: v.as_dict() for k, v in per_dataset.items()},
        "per_domain": {k: v.as_dict() for k, v in per_domain.items()},
        "per_context_length": {k: v.as_dict() for k, v in per_context.items()},
    }

    # ------------------------------------------------------------ error analysis
    if args.error_analysis:
        print(f"\n{RULE}\nERROR ANALYSIS\n{RULE}")
        service = SarcasmAnalysisService.from_checkpoint(
            checkpoint, device=args.device, config=config
        )
        pool = list(records)
        if args.error_analysis_samples and args.error_analysis_samples < len(pool):
            rng = random.Random(args.seed)
            pool = rng.sample(pool, args.error_analysis_samples)
        print(f"  running full inference + {args.explanation_method} on {len(pool):,} examples ...")

        analyses = []
        for index, record in enumerate(pool, start=1):
            result = service.analyze(
                record.text,
                record.context,
                explanation_method=args.explanation_method,
            )
            analyses.append(result.to_dict())
            if index % 50 == 0:
                print(f"    {index}/{len(pool)}")

        domain_f1 = {name: result.macro_f1 for name, result in per_domain.items()}
        rows = build_error_rows(
            pool, analyses, domain_f1=domain_f1, overall_f1=metrics.macro_f1
        )
        summary = summarise_errors(rows)
        payload["error_analysis"] = summary
        payload["error_analysis_settings"] = {
            "samples": len(pool),
            "explanation_method": args.explanation_method,
            "seed": args.seed,
        }

        csv_path = output_dir / f"error_analysis_{stem}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].as_dict()))
            writer.writeheader()
            writer.writerows(row.as_dict() for row in rows)

        print(
            f"\n  examples={summary['n_examples']:,} errors={summary['n_errors']:,} "
            f"(rate {summary['error_rate']:.1%})"
        )
        print(f"  false positives={summary['false_positives']} "
              f"false negatives={summary['false_negatives']}")
        print("\n  error categories:")
        for category, count in summary["categories"].items():
            share = summary["category_share"][category]
            print(f"    {category:<32} {count:>5}  ({share:.1%})")
            description = CATEGORY_DESCRIPTIONS.get(category, "")
            if description:
                print(f"        {description}")
        print(f"\n  full CSV -> {csv_path}")

    metrics_path = write_json(output_dir / f"evaluation_{stem}.json", payload)
    print(f"\n  metrics  -> {metrics_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
