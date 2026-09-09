#!/usr/bin/env python
"""Measure real inference latency, stage by stage.

The project specification targets **<300 ms per example on CPU** for
classification. That target applies to *retrieval + model*, not to explanation:
LIME and SHAP are hundreds of extra forward passes by construction and are
reported separately so the number is not quietly conflated.

Nothing here is estimated. Every figure printed is measured on this machine, and
``results/benchmark.json`` records the device it was measured on.

Examples
--------
    python scripts/benchmark_inference.py --checkpoint checkpoints/full_model/best_model
    python scripts/benchmark_inference.py --checkpoint checkpoints/full_model/best_model \\
        --runs 100 --explanation-methods lime occlusion shap
"""

from __future__ import annotations

import argparse
import logging
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.service import SarcasmAnalysisService  # noqa: E402
from src.models.dev_assets import is_development_asset  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.io import environment_snapshot, read_jsonl, write_json  # noqa: E402
from src.utils.logging_utils import configure_logging  # noqa: E402
from src.utils.paths import resolve  # noqa: E402
from src.utils.seeding import set_global_seed  # noqa: E402

logger = logging.getLogger("benchmark")
RULE = "=" * 78
TARGET_MS = 300.0

FALLBACK_INPUTS: list[tuple[str, list[str]]] = [
    ("Great, my laptop crashed five minutes before the presentation.", []),
    ("The train arrived exactly on schedule this morning.", []),
    (
        "Perfect timing, as always.",
        ["The delivery was supposed to arrive on Monday.", "It turned up nine days late."],
    ),
    ("Wow, what an amazing traffic jam. I only needed to get to work.", []),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None, help="cpu / cuda. Default: auto-detect.")
    parser.add_argument("--runs", type=int, default=50, help="Timed iterations per stage.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--explanation-methods",
        nargs="*",
        default=["occlusion", "lime"],
        choices=["lime", "shap", "occlusion"],
        help="Explanation methods to benchmark (SHAP is slow; add it deliberately).",
    )
    parser.add_argument(
        "--corpus",
        default="data/processed/dataset.jsonl",
        help="Sample real inputs from here; falls back to built-in demo inputs.",
    )
    parser.add_argument("--output", default="results/benchmark.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args(argv)


def _stats(samples: Sequence[float]) -> dict[str, float]:
    """Mean / median / p95 / min / max in milliseconds."""
    if not samples:
        return {"n": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "n": len(ordered),
        "mean_ms": round(statistics.fmean(ordered), 3),
        "median_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[index], 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
    }


def load_inputs(path: str, limit: int, seed: int) -> list[tuple[str, list[str]]]:
    """Sample real inputs from the processed corpus when it is available."""
    import random

    corpus_path = resolve(path)
    if not corpus_path.exists():
        logger.warning("Corpus %s not found; benchmarking on built-in demo inputs", corpus_path)
        return [FALLBACK_INPUTS[i % len(FALLBACK_INPUTS)] for i in range(limit)]
    rows = list(read_jsonl(corpus_path))
    rng = random.Random(seed)
    chosen = rng.sample(rows, min(limit, len(rows)))
    return [(row["text"], list(row.get("context") or [])) for row in chosen]


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a POSIX exit code."""
    args = parse_args(argv)
    configure_logging(args.log_level)
    set_global_seed(args.seed)

    checkpoint = resolve(args.checkpoint)
    config = Config.from_yaml(checkpoint / "training_config.yaml")
    service = SarcasmAnalysisService.from_checkpoint(checkpoint, device=args.device)
    device = str(service.predictor.device)
    dev_asset = is_development_asset(config.model.encoder_name) or is_development_asset(checkpoint)

    print(f"\n{RULE}\nINFERENCE BENCHMARK\n{RULE}")
    print(f"  checkpoint : {checkpoint}")
    print(f"  device     : {device}")
    print(f"  encoder    : {config.model.encoder_name} (hidden={service.predictor.model.hidden_size})")
    print(f"  fusion     : {config.model.fusion}")
    print(f"  runs       : {args.runs} (+{args.warmup} warm-up)")
    if dev_asset:
        print(
            "  ⚠  DEVELOPMENT encoder: these latencies are for the tiny test model and "
            "are NOT representative of RoBERTa-base."
        )

    inputs = load_inputs(args.corpus, args.runs + args.warmup, args.seed)

    # ---------------------------------------------------------------- warm-up
    for text, context in inputs[: args.warmup]:
        service.analyze(text, context, explain=False)

    # ------------------------------------------------- classification pipeline
    retrieval_ms: list[float] = []
    model_ms: list[float] = []
    total_ms: list[float] = []
    for text, context in inputs[args.warmup : args.warmup + args.runs]:
        started = time.perf_counter()
        result = service.analyze(text, context, explain=False)
        total_ms.append((time.perf_counter() - started) * 1000)
        retrieval_ms.append(result.timings_ms["retrieval"])
        model_ms.append(result.timings_ms["model"])

    stages: dict[str, Any] = {
        "retrieval": _stats(retrieval_ms),
        "model_forward": _stats(model_ms),
        "classification_total": _stats(total_ms),
    }

    # ---------------------------------------------------------- explanations
    explanation_stats: dict[str, Any] = {}
    for method in args.explanation_methods:
        samples: list[float] = []
        end_to_end: list[float] = []
        fallbacks: list[str] = []
        # SHAP is expensive; cap its iteration count so a benchmark run terminates.
        iterations = min(args.runs, 10 if method == "shap" else args.runs)
        for text, context in inputs[args.warmup : args.warmup + iterations]:
            started = time.perf_counter()
            result = service.analyze(text, context, explanation_method=method)
            elapsed = (time.perf_counter() - started) * 1000
            if result.explanation.method != method:
                # The registry fell back to another explainer for this input.
                # Timing it here would report a different method's latency under
                # this method's name, so it is excluded and counted instead.
                fallbacks.append(result.explanation.method)
                continue
            end_to_end.append(elapsed)
            samples.append(result.timings_ms["explanation"])
        explanation_stats[method] = {
            "explanation": _stats(samples),
            "end_to_end": _stats(end_to_end),
            "iterations": iterations,
            "measured": len(samples),
            "fallbacks": len(fallbacks),
            "fallback_methods": sorted(set(fallbacks)),
        }
        stages[f"explanation_{method}"] = _stats(samples)

    classification_p95 = stages["classification_total"]["p95_ms"]
    meets_target = classification_p95 <= TARGET_MS

    print(f"\n{RULE}\nRESULTS (milliseconds)\n{RULE}")
    header = f"  {'stage':<28}{'mean':>10}{'median':>10}{'p95':>10}{'min':>10}{'max':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, values in stages.items():
        print(
            f"  {name:<28}{values['mean_ms']:>10.2f}{values['median_ms']:>10.2f}"
            f"{values['p95_ms']:>10.2f}{values['min_ms']:>10.2f}{values['max_ms']:>10.2f}"
        )
        for method, payload in explanation_stats.items():
            if payload["fallbacks"]:
                print(
                f"\n  NOTE: {method} ran on {payload['measured']}/{payload['iterations']} "
                f"sampled inputs. {payload['fallbacks']} input(s) fell back to "
                f"{', '.join(payload['fallback_methods'])} and are EXCLUDED from the "
                f"{method} row above, so that row times {method} only."
            )
    print(f"\n  Target: classification (retrieval + model) < {TARGET_MS:.0f} ms on CPU")
    print(
        f"  Measured p95 on {device}: {classification_p95:.1f} ms -> "
        f"{'MET' if meets_target else 'NOT MET'}"
    )
    if device != "cpu":
        print(
            "  Note: the target is specified for CPU. Re-run with --device cpu to "
            "evaluate it directly."
        )
    print(
        "  Explanation latency is reported separately and is deliberately outside "
        "this target."
    )

    payload = {
        "checkpoint": str(checkpoint),
        "device": device,
        "is_development_asset": dev_asset,
        "n_runs": args.runs,
        "warmup": args.warmup,
        "encoder": config.model.encoder_name,
        "fusion": config.model.fusion,
        "retrieval_top_k": config.retrieval.top_k,
        "stages": stages,
        "explanations": explanation_stats,
        "target_ms": TARGET_MS,
        "meets_300ms_target": bool(meets_target) if device == "cpu" else None,
        "platform": platform.platform(),
        "environment": environment_snapshot(),
    }
    path = write_json(args.output, payload)
    print(f"\n  written -> {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
