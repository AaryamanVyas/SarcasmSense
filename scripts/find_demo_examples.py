"""Find test-set utterances this checkpoint confidently predicts as Sarcastic.

Use it to pick live-demo inputs that the trained model actually gets right,
instead of guessing at sentences and hoping. Every example printed is a real
record from the held-out test split together with the model's real output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.predictor import SarcasmPredictor  # noqa: E402
from src.training.pipeline import load_corpus, load_splits  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.paths import resolve  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a POSIX exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scan", type=int, default=400, help="How many records to score.")
    parser.add_argument("--show", type=int, default=15, help="How many hits to print.")
    parser.add_argument("--max-chars", type=int, default=160, help="Skip longer utterances.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    checkpoint = resolve(args.checkpoint)
    config = Config.from_yaml(checkpoint / "training_config.yaml")
    predictor = SarcasmPredictor.from_checkpoint(checkpoint, device=args.device, config=config)

    records = list(load_splits(config, load_corpus(config))[args.split])[: args.scan]
    hits = []
    for record in records:
        if record.label != 1 or len(record.text) > args.max_chars:
            continue
        result = predictor.predict(record.text, context=record.context)
        if result.label == 1:
            hits.append((result.confidence, record, result))

    hits.sort(key=lambda item: -item[0])
    print(f"\nScanned {len(records)} {args.split} records; {len(hits)} correct sarcastic hits.\n")
    for confidence, record, _result in hits[: args.show]:
        print(f"[{confidence:.1%} sarcastic]  {record.text}")
        for turn_index, turn in enumerate(record.context):
            print(f"      context {turn_index + 1}: {turn}")
        print(f"      source: {record.dataset}\n")

    if not hits:
        print(
            "No confident sarcastic predictions in the scanned window. Raise --scan, "
            "or the model is not usefully separating the classes yet."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())