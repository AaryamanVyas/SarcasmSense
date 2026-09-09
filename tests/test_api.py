"""FastAPI contract, validation, error envelopes and degraded modes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.settings import Settings


@pytest.fixture(scope="module")
def client(checkpoint_dir_module) -> TestClient:
    """A TestClient serving the fixture checkpoint."""
    settings = Settings(
        model_path=checkpoint_dir_module,
        results_dir="results",
        log_level="ERROR",
        cors_origins="http://localhost:5173",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def checkpoint_dir_module(request) -> str:
    """Module-scoped view of the session ``checkpoint_dir`` fixture."""
    return request.getfixturevalue("checkpoint_dir")


class TestSystemEndpoints:
    """Health, model info and the index route."""

    def test_index(self, client):
        body = client.get("/").json()
        assert body["docs"] == "/docs" and body["health"] == "/health"

    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] in ("ok", "degraded")
        assert body["model_loaded"] is True
        assert body["device"] == "cpu"
        assert body["uptime_seconds"] >= 0

    def test_model_info(self, client):
        body = client.get("/model/info").json()
        assert body["model"]["fusion"] == "cross_attention"
        assert body["task"] == "Binary sarcasm classification"
        assert body["input_modality"] == "text only"
        assert body["max_context_turns"] == 3
        assert "available_methods" in body["explanation"]

    def test_openapi_schema_is_served(self, client):
        schema = client.get("/openapi.json").json()
        assert "/predict" in schema["paths"]
        assert "/explain" in schema["paths"]
        assert "/retrieve" in schema["paths"]

    def test_request_id_header(self, client):
        response = client.get("/health")
        assert response.headers.get("X-Request-ID")
        assert response.headers.get("X-Process-Time-Ms")

    def test_supplied_request_id_is_echoed(self, client):
        response = client.get("/health", headers={"X-Request-ID": "my-trace-id"})
        assert response.headers["X-Request-ID"] == "my-trace-id"


class TestPredict:
    """The main endpoint's contract."""

    def test_full_response_shape(self, client):
        response = client.post(
            "/predict",
            json={
                "text": "Great, my laptop crashed before the presentation.",
                "context": ["I have to present at three."],
                "explanation_method": "occlusion",
            },
        )
        assert response.status_code == 200
        body = response.json()
        for key in (
            "prediction", "label", "analysed_text", "confidence", "probabilities",
            "important_tokens", "token_importance", "context_used", "context_influence",
            "retrieved_evidence", "retrieval_status", "evidence_attention",
            "explanation", "explanation_method", "incongruity", "warnings",
            "timings_ms", "request_id",
        ):
            assert key in body, f"missing key {key!r}"
        assert body["prediction"] in ("Sarcastic", "Not Sarcastic")
        assert body["label"] in (0, 1)
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["probabilities"]["sarcastic"] + body["probabilities"]["not_sarcastic"] == pytest.approx(1.0, abs=1e-3)
        assert body["context_used"] == ["I have to present at three."]

    def test_no_context(self, client):
        body = client.post("/predict", json={"text": "A plain headline.", "context": []}).json()
        assert body["context_used"] == []
        assert body["context_influence"] == []

    @pytest.mark.parametrize("n_turns", [0, 1, 2, 3])
    def test_every_allowed_context_length(self, client, n_turns):
        context = [f"turn number {i}" for i in range(n_turns)]
        response = client.post(
            "/predict",
            json={"text": "Target.", "context": context, "explanation_method": "occlusion"},
        )
        assert response.status_code == 200
        assert len(response.json()["context_used"]) == n_turns

    def test_token_importance_is_normalised(self, client):
        body = client.post(
            "/predict", json={"text": "Oh perfect timing.", "explanation_method": "occlusion"}
        ).json()
        weights = [abs(t["weight"]) for t in body["token_importance"]]
        assert max(weights) == pytest.approx(1.0, abs=1e-6)

    def test_evidence_carries_provenance(self, client):
        body = client.post("/predict", json={"text": "rain causes wetness"}).json()
        for fact in body["retrieved_evidence"]:
            assert fact["text"] and fact["source"] and fact["fact_id"]
            assert -1.0 <= fact["score"] <= 1.0

    def test_explain_endpoint_forces_attribution(self, client):
        body = client.post(
            "/explain",
            json={"text": "Oh wonderful.", "explain": False, "explanation_method": "occlusion"},
        ).json()
        assert body["explanation_method"] == "occlusion"
        assert body["token_importance"]

    def test_explanation_method_is_reported_truthfully(self, client):
        body = client.post(
            "/predict", json={"text": "Great news.", "explanation_method": "lime"}
        ).json()
        assert body["explanation_method"] in ("lime", "occlusion")

    def test_skipping_explanation(self, client):
        body = client.post(
            "/predict", json={"text": "Great news.", "explain": False}
        ).json()
        assert body["explanation_method"] == "none"
        assert body["token_importance"] == []


class TestValidation:
    """Malformed requests are rejected with a readable 422."""

    @pytest.mark.parametrize("payload", [
        {"text": ""},
        {"text": "   "},
        {"text": "\n\t"},
        {},
    ])
    def test_bad_text_is_rejected(self, client, payload):
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert response.json()["error"] == "validation_error"
        assert response.json()["detail"]

    def test_too_many_context_turns(self, client):
        response = client.post("/predict", json={"text": "x", "context": ["a", "b", "c", "d"]})
        assert response.status_code == 422
        assert "at most 3 turns" in response.json()["detail"]

    def test_text_too_long(self, client):
        response = client.post("/predict", json={"text": "x" * 5000})
        assert response.status_code == 422

    def test_unknown_explanation_method(self, client):
        response = client.post(
            "/predict", json={"text": "x", "explanation_method": "crystal_ball"}
        )
        assert response.status_code == 422

    def test_blank_context_turns_are_dropped_not_rejected(self, client):
        body = client.post(
            "/predict", json={"text": "target", "context": ["", "  ", "real turn"]}
        ).json()
        assert body["context_used"] == ["real turn"]

    def test_wrong_type_is_rejected(self, client):
        assert client.post("/predict", json={"text": 42}).status_code == 422

    def test_unknown_route_returns_the_error_envelope(self, client):
        response = client.get("/no-such-route")
        assert response.status_code == 404
        assert set(response.json()) == {"error", "detail", "request_id"}


class TestRetrieveEndpoint:
    """Retrieval-only inspection endpoint."""

    def test_returns_evidence(self, client):
        body = client.post("/retrieve", json={"text": "rain causes wetness", "top_k": 2}).json()
        assert body["status"] == "ok"
        assert 1 <= len(body["evidence"]) <= 2
        assert body["latency_ms"] >= 0

    def test_top_k_bounds_are_enforced(self, client):
        assert client.post("/retrieve", json={"text": "x", "top_k": 0}).status_code == 422
        assert client.post("/retrieve", json={"text": "x", "top_k": 99}).status_code == 422

    def test_empty_text_is_rejected(self, client):
        assert client.post("/retrieve", json={"text": "  "}).status_code == 422


class TestDemoAndDashboard:
    """Demo examples and the results endpoint."""

    def test_examples_are_returned(self, client):
        examples = client.get("/examples").json()
        assert len(examples) >= 3
        for example in examples:
            assert example["text"] and example["title"] and example["note"]
            assert len(example["context"]) <= 3
        # At least one context-dependent pair must exist for the viva demo.
        assert any(example["context"] for example in examples)

    def test_results_endpoint_always_answers(self, client):
        body = client.get("/results").json()
        assert "available" in body
        if not body["available"]:
            assert "No experiment results available yet" in body["message"]


class TestDegradedModes:
    """The API must fail loudly and usefully, not silently."""

    def test_no_checkpoint_yields_503_with_a_reason(self, tmp_path):
        settings = Settings(model_path=str(tmp_path / "missing"), log_level="ERROR")
        with TestClient(create_app(settings)) as offline:
            health = offline.get("/health").json()
            assert health["status"] == "error"
            assert health["model_loaded"] is False
            assert health["detail"]

            response = offline.post("/predict", json={"text": "anything"})
            assert response.status_code == 503
            assert response.json()["detail"]

            assert offline.get("/model/info").status_code == 503
            # Validation still runs before the readiness check.
            assert offline.get("/examples").status_code == 200

    def test_model_loading_can_be_disabled(self, checkpoint_dir_module):
        settings = Settings(
            model_path=checkpoint_dir_module,
            load_model_on_startup=False,
            log_level="ERROR",
        )
        with TestClient(create_app(settings)) as lazy:
            assert lazy.get("/health").json()["model_loaded"] is False
            assert lazy.post("/predict", json={"text": "x"}).status_code == 503


class TestSingleContainerMode:
    """SERVE_FRONTEND=true, the deployment path for single-service platforms."""

    @pytest.fixture()
    def bundled(self, tmp_path, checkpoint_dir_module):
        """An app serving a minimal fake frontend build alongside the API."""
        dist = tmp_path / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text("<!doctype html><title>app</title>", encoding="utf-8")
        (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
        settings = Settings(
            model_path=checkpoint_dir_module,
            serve_frontend=True,
            frontend_dist=str(dist),
            log_level="ERROR",
        )
        with TestClient(create_app(settings)) as client:
            yield client

    def test_index_serves_the_bundle(self, bundled):
        response = bundled.get("/")
        assert response.status_code == 200
        assert "<title>app</title>" in response.text

    def test_assets_are_served(self, bundled):
        assert bundled.get("/assets/app.js").status_code == 200

    def test_missing_asset_is_a_real_404(self, bundled):
        assert bundled.get("/assets/nope.js").status_code == 404

    def test_client_side_route_falls_back_to_index(self, bundled):
        response = bundled.get("/some/deep/route")
        assert response.status_code == 200
        assert "<title>app</title>" in response.text

    def test_api_still_wins_at_the_root(self, bundled):
        assert bundled.get("/health").json()["model_loaded"] is True

    def test_api_is_also_mounted_under_the_api_prefix(self, bundled):
        assert bundled.get("/api/health").status_code == 200
        assert bundled.post("/api/predict", json={"text": "Oh great."}).status_code == 200

    def test_missing_build_degrades_without_crashing(self, tmp_path, checkpoint_dir_module):
        settings = Settings(
            model_path=checkpoint_dir_module,
            serve_frontend=True,
            frontend_dist=str(tmp_path / "absent"),
            log_level="ERROR",
        )
        with TestClient(create_app(settings)) as client:
            # No bundle, so the JSON service descriptor is served instead.
            assert client.get("/").json()["health"] == "/health"
            assert client.get("/health").status_code == 200


class TestCors:
    """CORS is configured from the environment, not hard-coded."""

    def test_allowed_origin_is_echoed(self, client):
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_disallowed_origin_is_not_echoed(self, client):
        response = client.options(
            "/predict",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.headers.get("access-control-allow-origin") != "https://evil.example"
