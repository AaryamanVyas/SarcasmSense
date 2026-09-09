#!/usr/bin/env python
"""Train one model variant described by a YAML config.

Examples
--------
    python scripts/train.py --config configs/baseline_roberta.yaml
    python scripts/train.py --config configs/full_model.yaml
    python scripts/train.py --config configs/full_model.yaml \\
        --override training.epochs=2 data.max_train_samples=2000

The test split is loaded but **only touched at the very end**, after the best
checkpoint has been selected on validation macro-F1. Pass ``--skip-test`` while
developing to keep it untouched entirely.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import LABEL_NAMES  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    compute_grouped_metrics,
    metrics_to_rows,
    render_comparison_table,
)
from src.evaluation.plots import (  # noqa: E402
    plot_class_distribution,
    plot_confusion_matrix,
    plot_metric_by_group,
    plot_roc,
    plot_training_curves,
)
from src.models.dev_assets import is_development_asset  # noqa: E402
from src.training.experiment import (  # noqa: E402
    ExperimentRecord,
    ExperimentTracker,
    make_run_id,
    model_type_from_config,
)
from src.training.pipeline import build_experiment  # noqa: E402
from src.training.trainer import Trainer, build_predictions_frame  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.io import environment_snapshot, write_json  # noqa: E402
from src.utils.logging_utils import configure_logging  # noqa: E402
from src.utils.paths import ensure_dir  # noqa: E402
from src.utils.seeding import set_global_seed  # noqa: E402

logger = logging.getLogger("train")
RULE = "=" * 78


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Path to the experiment YAML.")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="key.sub=value",
        help="Dotted config overrides applied after loading the YAML.",
    )
    parser.add_argument("--device", default=None, help="cpu / cuda / cuda:0 / mps")
    parser.add_argument("--seed", type=int, default=None, help="Overrides experiment.seed.")
    parser.add_argument("--skip-test", action="store_true", help="Do not touch the test split.")
    parser.add_argument("--rebuild-retrieval-cache", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Never contact the HuggingFace hub (offline / air-gapped runs).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a POSIX exit code."""
    args = parse_args(argv)
    config = Config.from_yaml(args.config, overrides=args.override)
    if args.seed is not None:
        config.experiment.seed = args.seed

    output_dir = ensure_dir(config.experiment.output_dir)
    configure_logging(args.log_level, log_file=output_dir / "train.log")
    set_global_seed(config.experiment.seed, deterministic=config.experiment.deterministic)

    print(f"\n{RULE}\nTRAINING: {config.experiment.name}\n{RULE}")
    print(f"  config      : {args.config}")
    print(f"  overrides   : {args.override or 'none'}")
    print(f"  seed        : {config.experiment.seed}")
    print(f"  output      : {output_dir}")

    context = build_experiment(
        config,
        device=args.device,
        include_train=True,
        rebuild_cache=args.rebuild_retrieval_cache,
        local_files_only=args.local_files_only,
    )
    dev_asset = is_development_asset(config.model.encoder_name)
    if dev_asset:
        print(
            "\n  ⚠  DEVELOPMENT/TEST ENCODER IN USE. This run exercises the software\n"
            "     pipeline with a tiny randomly-initialised RoBERTa. Its metrics are\n"
            "     NOT research results for the proposed architecture.\n"
        )
    for warning in context.warnings:
        print(f"  ⚠  {warning}")

    print(f"\n  model       : {context.model.describe()}")
    print(
        f"  splits      : train={len(context.datasets['train'])} "
        f"val={len(context.datasets['val'])} test={len(context.datasets['test'])}"
    )

    trainer = Trainer(
        model=context.model,
        config=config,
        collator=context.collator,
        train_dataset=context.datasets["train"],
        val_dataset=context.datasets["val"],
        device=args.device,
        output_dir=output_dir,
    )

    print(f"\n{RULE}\nTRAINING LOOP\n{RULE}")
    started = time.perf_counter()
    history = trainer.train()
    train_seconds = time.perf_counter() - started
    print(
        f"\n  best epoch {history.best_epoch} "
        f"({config.training.metric_for_best}={history.best_metric:.4f}) "
        f"in {train_seconds / 60:.1f} min"
    )

    # ---------------------------------------------------------------- evaluate
    print(f"\n{RULE}\nEVALUATION\n{RULE}")
    best_dir = output_dir / "best_model"
    if best_dir.exists():
        from src.models.sarcasm_model import SarcasmModel

        trainer.model = SarcasmModel.from_pretrained(
            best_dir,
            map_location=str(trainer.device),
            local_files_only=args.local_files_only,
        ).to(trainer.device)
        print(f"  loaded best checkpoint from {best_dir}")

    val_metrics, val_predictions = trainer.evaluate(context.datasets["val"])
    print(f"  validation : {val_metrics.summary()}")

    results: dict[str, object] = {
        "experiment": config.experiment.name,
        "config": config.to_dict(),
        "is_development_asset": dev_asset,
        "warnings": context.warnings,
        "validation": val_metrics.as_dict(),
        "history": history.as_dict(),
        "train_seconds": round(train_seconds, 2),
        "environment": environment_snapshot(),
    }
    test_metrics = None
    if not args.skip_test:
        test_metrics, test_predictions = trainer.evaluate(context.datasets["test"])
        print(f"  test       : {test_metrics.summary()}")
        test_records = context.splits["test"]
        by_id = {record.id: record for record in test_records}
        datasets_of = [by_id[i].dataset for i in test_predictions.record_ids]
        domains_of = [by_id[i].domain for i in test_predictions.record_ids]
        context_of = [
            f"{min(len(by_id[i].context), 3)}_turns" for i in test_predictions.record_ids
        ]
        per_dataset = compute_grouped_metrics(
            test_predictions.y_true,
            test_predictions.y_pred,
            datasets_of,
            test_predictions.y_prob_sarcastic,
        )
        per_domain = compute_grouped_metrics(
            test_predictions.y_true,
            test_predictions.y_pred,
            domains_of,
            test_predictions.y_prob_sarcastic,
        )
        per_context = compute_grouped_metrics(
            test_predictions.y_true,
            test_predictions.y_pred,
            context_of,
            test_predictions.y_prob_sarcastic,
        )
        results["test"] = test_metrics.as_dict()
        results["per_dataset"] = {k: v.as_dict() for k, v in per_dataset.items()}
        results["per_domain"] = {k: v.as_dict() for k, v in per_domain.items()}
        results["per_context_length"] = {k: v.as_dict() for k, v in per_context.items()}

        print("\n  per-dataset (test):")
        print("   " + render_comparison_table(per_dataset).replace("\n", "\n   "))

        # ------------------------------------------------------------- figures
        figures = ensure_dir(Path(config.experiment.results_dir) / "figures")
        stem = config.experiment.name
        plot_confusion_matrix(
            test_metrics.confusion_matrix,
            figures / f"{stem}_confusion_matrix.png",
            title=f"{stem}: confusion matrix (test)",
        )
        plot_roc(
            test_predictions.y_true,
            test_predictions.y_prob_sarcastic,
            figures / f"{stem}_roc.png",
            title=f"{stem}: ROC (test)",
        )
        plot_metric_by_group(
            per_dataset,
            figures / f"{stem}_macro_f1_by_dataset.png",
            title=f"{stem}: macro-F1 by dataset (test)",
        )
        plot_class_distribution(
            {
                name: {
                    LABEL_NAMES[0]: sum(1 for r in rows if r.label == 0),
                    LABEL_NAMES[1]: sum(1 for r in rows if r.label == 1),
                }
                for name, rows in context.splits.items()
            },
            figures / "class_distribution.png",
        )
        plot_training_curves(
            history.as_dict(), figures / f"{stem}_training_curves.png", title=stem
        )

        # ------------------------------------------------- predictions export
        frame = build_predictions_frame(test_predictions, test_records)
        write_json(
            Path(config.experiment.results_dir) / f"predictions_{stem}_test.json", frame
        )

    write_json(output_dir / "metrics.json", results)
    write_json(Path(config.experiment.results_dir) / f"metrics_{config.experiment.name}.json", results)

    # --------------------------------------------------------------- tracking
    # The run log lives under the experiment's own results_dir, so development
    # runs (results/dev) never contaminate the research run log (results/).
    results_root = Path(config.experiment.results_dir)
    tracker = ExperimentTracker(
        json_path=results_root / "experiments.json",
        csv_path=results_root / "experiments.csv",
    )
    record = ExperimentRecord(
        run_id=make_run_id(config.experiment.name, config.experiment.seed),
        experiment=config.experiment.name,
        model_type=model_type_from_config(config.to_dict()),
        config=config.to_dict(),
        seed=config.experiment.seed,
        datasets=sorted({r.dataset for r in context.records}),
        val_metrics=val_metrics.as_dict(),
        test_metrics=test_metrics.as_dict() if test_metrics else {},
        per_dataset_metrics=results.get("per_dataset", {}),  # type: ignore[arg-type]
        per_domain_metrics=results.get("per_domain", {}),  # type: ignore[arg-type]
        history=history.as_dict(),
        train_seconds=train_seconds,
        warnings=context.warnings,
        is_development_asset=dev_asset,
    )
    tracker.record(record)

    # results.csv keeps a flat per-split table for the README and the dashboard.
    rows = metrics_to_rows(
        {f"{config.experiment.name}/validation": val_metrics}
        | ({f"{config.experiment.name}/test": test_metrics} if test_metrics else {})
    )
    import csv as _csv

    csv_path = Path(config.experiment.results_dir) / "results.csv"
    ensure_dir(csv_path.parent)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\n  metrics  -> {output_dir / 'metrics.json'}")
    print(f"  results  -> {csv_path}")
    print(f"  run log  -> results/experiments.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
