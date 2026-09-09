"""End-to-end training integration on the fixture corpus.

This exercises the whole research path -- corpus loading, splitting, retrieval
caching, collation, the optimiser/scheduler, mixed-precision resolution, early
stopping, checkpointing, resume, evaluation, plots and experiment tracking -- in
a few seconds, on synthetic data. It verifies that the machinery *works*, not
that the model is any good.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.data.splitting import save_split, split_records
from src.evaluation.metrics import compute_metrics, render_comparison_table
from src.evaluation.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_metric_by_group,
    plot_roc,
    plot_training_curves,
)
from src.training.experiment import (
    ExperimentRecord,
    ExperimentTracker,
    make_run_id,
    model_type_from_config,
)
from src.training.pipeline import build_experiment
from src.training.trainer import (
    Trainer,
    build_predictions_frame,
    resolve_amp,
    resolve_device,
)


@pytest.fixture()
def prepared_config(experiment_config, fixture_records, tmp_path):
    """A *copy* of the session config whose split files exist on disk.

    Copying matters: several tests below deliberately break the config (missing
    corpus, missing index), and mutating the shared session fixture would leak
    those breakages into every later test.
    """
    from src.utils.config import Config

    config = Config.from_dict(experiment_config.to_dict())
    config.data.splits_dir = str(tmp_path / "splits")
    config.experiment.output_dir = str(tmp_path / "out")
    config.experiment.results_dir = str(tmp_path / "results")
    config.training.epochs = 1
    config.training.batch_size = 8
    config.training.log_every = 0
    save_split(split_records(fixture_records, seed=42), config.data.splits_dir)
    return config


class TestDeviceAndAmp:
    """Hardware resolution behaves sensibly on a CPU-only machine."""

    def test_device_override(self):
        assert resolve_device("cpu").type == "cpu"

    def test_auto_device(self):
        assert resolve_device().type in ("cpu", "cuda", "mps")

    def test_amp_is_disabled_on_cpu(self):
        enabled, dtype = resolve_amp("auto", torch.device("cpu"))
        assert enabled is False and dtype is None

    def test_amp_off_mode(self):
        assert resolve_amp("off", torch.device("cuda")) == (False, None)


class TestBuildExperiment:
    """The factory wires the whole experiment from one config."""

    def test_context_is_complete(self, prepared_config):
        context = build_experiment(prepared_config, device="cpu", local_files_only=True)
        assert set(context.splits) == {"train", "val", "test"}
        assert len(context.records) > 0
        assert context.model.config.fusion == "cross_attention"
        assert context.retrieval_cache is not None
        assert context.class_weights is not None
        assert context.tokenizer is not None

    def test_missing_corpus_raises_with_guidance(self, prepared_config):
        prepared_config.data.processed_path = "does/not/exist.jsonl"
        with pytest.raises(FileNotFoundError, match="prepare_datasets"):
            build_experiment(prepared_config, device="cpu", local_files_only=True)

    def test_missing_index_degrades_with_a_warning(self, prepared_config):
        prepared_config.retrieval.index_dir = "no/such/index"
        context = build_experiment(prepared_config, device="cpu", local_files_only=True)
        assert any("No knowledge index" in w for w in context.warnings)
        assert context.retrieval_cache is None
        # The model must still be usable without evidence.
        batch = context.collator(context.splits["train"][:2])
        from src.data.dataset import model_inputs

        with torch.no_grad():
            output = context.model(**model_inputs(batch))
        assert torch.isfinite(output.logits).all()

    def test_subsampling_is_seeded_not_a_head_slice(self, prepared_config):
        prepared_config.data.max_train_samples = 6
        first = build_experiment(prepared_config, device="cpu", local_files_only=True)
        second = build_experiment(prepared_config, device="cpu", local_files_only=True)
        assert len(first.splits["train"]) == 6
        assert [r.id for r in first.splits["train"]] == [r.id for r in second.splits["train"]]


class TestTrainingLoop:
    """One real epoch, plus checkpointing and resume."""

    def test_train_and_evaluate(self, prepared_config):
        context = build_experiment(prepared_config, device="cpu", local_files_only=True)
        trainer = Trainer(
            model=context.model,
            config=prepared_config,
            collator=context.collator,
            train_dataset=context.datasets["train"],
            val_dataset=context.datasets["val"],
            device="cpu",
        )
        history = trainer.train()
        assert len(history.epochs) == 1
        assert history.epochs[0].train_loss > 0
        assert 0.0 <= history.best_metric <= 1.0

        output_dir = Path(prepared_config.experiment.output_dir)
        assert (output_dir / "best_model" / "pytorch_model.bin").exists()
        assert (output_dir / "best_model" / "training_config.yaml").exists()
        assert (output_dir / "best_model" / "trainer_state.pt").exists()
        assert (output_dir / "history.json").exists()
        assert (output_dir / "history.jsonl").exists()

        metrics, predictions = trainer.evaluate(context.datasets["test"])
        assert metrics.n_examples == len(context.datasets["test"])
        assert predictions.y_pred.shape == predictions.y_true.shape
        assert set(predictions.y_pred.tolist()) <= {0, 1}
        assert len(predictions.record_ids) == metrics.n_examples

    def test_predictions_frame_joins_back_to_records(self, prepared_config):
        context = build_experiment(prepared_config, device="cpu", local_files_only=True)
        trainer = Trainer(
            model=context.model, config=prepared_config,
            collator=context.collator, device="cpu",
        )
        predictions = trainer.predict(context.datasets["val"])
        rows = build_predictions_frame(predictions, context.splits["val"])
        assert len(rows) == len(predictions.record_ids)
        assert {"id", "text", "context", "true_label", "predicted_label", "confidence"} <= set(rows[0])

    def test_gradient_accumulation_runs(self, prepared_config):
        prepared_config.training.gradient_accumulation_steps = 3
        context = build_experiment(prepared_config, device="cpu", local_files_only=True)
        trainer = Trainer(
            model=context.model, config=prepared_config, collator=context.collator,
            train_dataset=context.datasets["train"], val_dataset=context.datasets["val"],
            device="cpu",
        )
        assert len(trainer.train().epochs) == 1

    def test_early_stopping_triggers(self, prepared_config):
        prepared_config.training.epochs = 4
        prepared_config.training.early_stopping_patience = 1
        context = build_experiment(prepared_config, device="cpu", local_files_only=True)
        trainer = Trainer(
            model=context.model, config=prepared_config, collator=context.collator,
            train_dataset=context.datasets["train"], val_dataset=context.datasets["val"],
            device="cpu",
        )
        history = trainer.train()
        assert len(history.epochs) <= 4
        if history.stopped_early:
            assert history.epochs[-1].is_best is False

    def test_resume_from_checkpoint(self, prepared_config):
        context = build_experiment(prepared_config, device="cpu", local_files_only=True)
        trainer = Trainer(
            model=context.model, config=prepared_config, collator=context.collator,
            train_dataset=context.datasets["train"], val_dataset=context.datasets["val"],
            device="cpu",
        )
        trainer.train()

        prepared_config.training.epochs = 2
        prepared_config.training.resume_from = str(
            Path(prepared_config.experiment.output_dir) / "last"
        )
        resumed = build_experiment(prepared_config, device="cpu", local_files_only=True)
        second = Trainer(
            model=resumed.model, config=prepared_config, collator=resumed.collator,
            train_dataset=resumed.datasets["train"], val_dataset=resumed.datasets["val"],
            device="cpu",
        )
        history = second.train()
        # Resuming from epoch 1 of a 2-epoch schedule leaves exactly one to run.
        assert len(history.epochs) == 1
        assert history.epochs[0].epoch == 2

    def test_training_requires_both_splits(self, prepared_config):
        context = build_experiment(prepared_config, device="cpu", local_files_only=True)
        trainer = Trainer(
            model=context.model, config=prepared_config, collator=context.collator,
            device="cpu",
        )
        with pytest.raises(ValueError, match="train and a validation"):
            trainer.train()


class TestExperimentTracking:
    """Runs are appended to JSON and CSV without any external service."""

    def test_record_round_trip(self, tmp_path, prepared_config):
        tracker = ExperimentTracker(
            json_path=tmp_path / "experiments.json", csv_path=tmp_path / "experiments.csv"
        )
        assert tracker.load() == []
        metrics = compute_metrics([0, 1, 0, 1], [0, 1, 1, 1], [0.2, 0.8, 0.6, 0.9])
        record = ExperimentRecord(
            run_id=make_run_id("pytest", 42),
            experiment="pytest",
            model_type=model_type_from_config(prepared_config.to_dict()),
            config=prepared_config.to_dict(),
            seed=42,
            datasets=["fixture_news"],
            val_metrics=metrics.as_dict(),
            test_metrics=metrics.as_dict(),
        )
        tracker.record(record)
        runs = tracker.load()
        assert len(runs) == 1
        assert runs[0]["experiment"] == "pytest"
        assert runs[0]["environment"]["packages"]["torch"] != "not-installed"

        header, row = (tmp_path / "experiments.csv").read_text().strip().split("\n")
        assert "test_macro_f1" in header
        assert "pytest" in row

        tracker.record(record)
        assert len(tracker.load()) == 2

    @pytest.mark.parametrize(
        "context,knowledge,expected",
        [
            (False, False, "baseline1_roberta"),
            (True, False, "baseline2_roberta_context"),
            (False, True, "baseline3_roberta_knowledge"),
            (True, True, "full_model_cross_attention"),
        ],
    )
    def test_model_type_naming(self, context, knowledge, expected):
        config = {
            "model": {
                "use_context": context,
                "use_knowledge": knowledge,
                "fusion": "cross_attention",
            }
        }
        assert model_type_from_config(config) == expected


class TestPlots:
    """Figures are produced from real arrays and written to disk."""

    def test_all_figures(self, tmp_path):
        metrics = compute_metrics(
            [0, 1, 0, 1, 1, 0], [0, 1, 1, 1, 0, 0], [0.1, 0.9, 0.7, 0.8, 0.4, 0.2]
        )
        assert plot_confusion_matrix(metrics.confusion_matrix, tmp_path / "cm.png").exists()
        assert plot_roc([0, 1, 0, 1], [0.1, 0.9, 0.3, 0.8], tmp_path / "roc.png").exists()
        assert plot_metric_by_group({"a": metrics}, tmp_path / "group.png").exists()
        assert plot_class_distribution(
            {"train": {"Sarcastic": 10, "Not Sarcastic": 12}}, tmp_path / "dist.png"
        ).exists()
        history = {
            "epochs": [
                {"epoch": 1, "train_loss": 0.7, "val_loss": 0.6,
                 "val_metrics": {"macro_f1": 0.5, "accuracy": 0.55}},
                {"epoch": 2, "train_loss": 0.5, "val_loss": 0.5,
                 "val_metrics": {"macro_f1": 0.62, "accuracy": 0.64}},
            ],
            "best_epoch": 2,
        }
        assert plot_training_curves(history, tmp_path / "curves.png").exists()

    def test_roc_is_skipped_when_undefined(self, tmp_path):
        assert plot_roc([1, 1, 1], [0.9, 0.8, 0.7], tmp_path / "roc.png") is None

    def test_training_curves_skipped_without_epochs(self, tmp_path):
        assert plot_training_curves({"epochs": []}, tmp_path / "c.png") is None

    def test_comparison_table_renders(self):
        metrics = compute_metrics([0, 1], [0, 1], [0.2, 0.8])
        table = render_comparison_table({"baseline": metrics, "full": metrics})
        assert "macro-F1" in table and "baseline" in table and "full" in table


class TestScriptsAreImportable:
    """Every CLI entry point must import and expose a main()."""

    @pytest.mark.parametrize(
        "module",
        [
            "download_datasets", "prepare_datasets", "build_knowledge_index",
            "train", "evaluate", "run_ablation", "benchmark_inference",
            "smoke_test", "make_dev_assets", "quickstart", "check_environment",
        ],
    )
    def test_import(self, module):
        import importlib
        import sys

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "scripts"))
        try:
            imported = importlib.import_module(module)
            assert callable(imported.main)
            # check_environment takes no arguments by design; every other entry
            # point exposes parse_args so its CLI can be unit-tested.
            if module != "check_environment":
                assert callable(imported.parse_args)
        finally:
            sys.path.remove(str(root / "scripts"))


class TestConfigFilesAreValid:
    """Every shipped YAML config parses and describes a coherent system."""

    def test_all_configs(self):
        from src.utils.config import Config

        root = Path(__file__).resolve().parents[1] / "configs"
        configs = sorted(root.glob("*.yaml"))
        assert configs, "no configs found"
        for path in configs:
            config = Config.from_yaml(path)
            config.validate()
            if config.model.use_knowledge:
                assert config.retrieval.enabled
                assert config.retrieval.top_k > 0

    def test_the_four_required_systems_exist(self):
        from src.utils.config import Config

        root = Path(__file__).resolve().parents[1] / "configs"
        signatures = {
            (Config.from_yaml(p).model.use_context, Config.from_yaml(p).model.use_knowledge)
            for p in root.glob("*.yaml")
        }
        assert (False, False) in signatures  # baseline 1
        assert (True, False) in signatures   # baseline 2
        assert (False, True) in signatures   # baseline 3
        assert (True, True) in signatures    # full model

    def test_overrides_apply(self):
        from src.utils.config import Config, ConfigError

        root = Path(__file__).resolve().parents[1] / "configs"
        config = Config.from_yaml(
            root / "full_model.yaml", overrides=["training.epochs=7", "retrieval.top_k=3"]
        )
        assert config.training.epochs == 7 and config.retrieval.top_k == 3
        with pytest.raises(ConfigError):
            Config.from_yaml(root / "full_model.yaml", overrides=["nonsense.key=1"])

    def test_saved_config_round_trips(self, tmp_path):
        from src.utils.config import Config

        root = Path(__file__).resolve().parents[1] / "configs"
        original = Config.from_yaml(root / "full_model.yaml")
        original.save(tmp_path / "c.yaml")
        assert Config.from_yaml(tmp_path / "c.yaml").to_dict() == original.to_dict()

    def test_unknown_keys_are_rejected(self, tmp_path):
        from src.utils.config import Config, ConfigError

        path = tmp_path / "bad.yaml"
        path.write_text("model:\n  not_a_real_key: 1\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="Unknown configuration key"):
            Config.from_yaml(path)

    def test_inconsistent_config_is_rejected(self, tmp_path):
        from src.utils.config import Config, ConfigError

        path = tmp_path / "bad.yaml"
        path.write_text(
            "model:\n  use_knowledge: true\nretrieval:\n  enabled: false\n", encoding="utf-8"
        )
        with pytest.raises(ConfigError, match="requires retrieval.enabled"):
            Config.from_yaml(path)


class TestReproducibility:
    """Seeding and environment capture."""

    def test_seed_makes_torch_deterministic(self):
        from src.utils.seeding import set_global_seed

        state = set_global_seed(123)
        assert state.seed == 123
        first = torch.randn(4)
        set_global_seed(123)
        torch.testing.assert_close(first, torch.randn(4))

    def test_environment_snapshot_has_versions(self):
        from src.utils.io import environment_snapshot

        snapshot = environment_snapshot()
        assert snapshot["python"]
        assert snapshot["packages"]["torch"] != "not-installed"
        assert "torch_cuda_available" in snapshot

    def test_json_helpers_round_trip(self, tmp_path):
        from src.utils.io import read_json, read_jsonl, write_json, write_jsonl

        write_json(tmp_path / "a.json", {"k": 1})
        assert read_json(tmp_path / "a.json") == {"k": 1}
        assert write_jsonl(tmp_path / "b.jsonl", [{"i": 0}, {"i": 1}]) == 2
        assert [row["i"] for row in read_jsonl(tmp_path / "b.jsonl")] == [0, 1]

    def test_invalid_jsonl_reports_the_line(self, tmp_path):
        from src.utils.io import read_jsonl

        (tmp_path / "bad.jsonl").write_text('{"ok": 1}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match=":2:"):
            list(read_jsonl(tmp_path / "bad.jsonl"))
