"""
Integration tests for EvaluationOrchestrator.

All external dependencies (Ollama, Whisper, HTTP) are mocked.
Tests verify the full pipeline: test case ingestion → evaluation → report.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from pipeline.orchestrator import EvaluationOrchestrator
from pipeline.voice_ai_client import MockVoiceAIClient
from pipeline.audio_transcriber import MockTranscriber
from utils.determinism import seed_everything


MINIMAL_CONFIG = {
    "evaluation": {
        "seed": 42,
        "run_id_prefix": "test",
        "output_dir": None,  # Set per-test
        "parallel_workers": 1,
    },
    "whisper": {"model_size": "tiny", "device": "cpu"},
    "ollama": {"host": "http://localhost:11434", "model": "llama3"},
    "metrics": {
        "wer": {"enabled": True},
        "latency": {"enabled": True},
        "semantic_similarity": {"enabled": False},  # Disabled: needs model download
        "hallucination": {"enabled": False},         # Disabled: needs Ollama
    },
    "scoring": {
        "weights": {"wer": 0.5, "latency": 0.5},
        "latency_thresholds": {"excellent": 200, "good": 500, "acceptable": 1000, "poor": 2000},
    },
    "thresholds": {
        "min_composite_score": 0.50,
        "max_wer": 0.30,
        "max_latency_ms": 5000,
    },
}

SAMPLE_CASES = [
    {
        "id": "tc_001",
        "category": "factual",
        "input_text": "What is the capital of France?",
        "audio_file": None,
        "expected_response": "The capital of France is Paris.",
        "reference_facts": ["Paris is the capital of France"],
    },
    {
        "id": "tc_002",
        "category": "conversational",
        "input_text": "Hello!",
        "audio_file": None,
        "expected_response": "Hello! How can I help you today?",
        "reference_facts": [],
    },
]


@pytest.fixture
def config_with_output(tmp_path):
    config = {**MINIMAL_CONFIG, "evaluation": {**MINIMAL_CONFIG["evaluation"], "output_dir": str(tmp_path)}}
    return config


@pytest.fixture
def orchestrator(config_with_output):
    client = MockVoiceAIClient(fixed_latency_ms=150.0)
    transcriber = MockTranscriber()
    return EvaluationOrchestrator(
        config=config_with_output,
        voice_ai_client=client,
        transcriber=transcriber,
        mock=True,
    )


class TestOrchestratorInit:

    def test_seed_set_during_init(self, config_with_output, tmp_path):
        """Orchestrator must seed all RNG on construction."""
        import random
        import numpy as np

        orch = EvaluationOrchestrator(
            config=config_with_output,
            voice_ai_client=MockVoiceAIClient(),
            transcriber=MockTranscriber(),
        )
        # Seed 42 → first random.random() is deterministic
        r1 = random.random()
        seed_everything(42)
        r2 = random.random()
        assert r1 == r2

    def test_evaluators_built_from_config(self, orchestrator):
        names = [e.name for e in orchestrator.evaluators]
        assert "wer" in names
        assert "latency" in names
        # semantic_similarity and hallucination disabled in minimal config
        assert "semantic_similarity" not in names
        assert "hallucination" not in names


class TestOrchestratorRun:

    def test_run_produces_report_file(self, orchestrator, config_with_output):
        output_dir = Path(config_with_output["evaluation"]["output_dir"])
        report_path = orchestrator.run(SAMPLE_CASES)

        assert report_path.exists()
        assert report_path.suffix == ".json"

    def test_run_report_has_correct_case_count(self, orchestrator, config_with_output):
        report_path = orchestrator.run(SAMPLE_CASES)
        with open(report_path) as f:
            data = json.load(f)
        assert data["summary"]["total_cases"] == len(SAMPLE_CASES)

    def test_run_report_is_valid_json(self, orchestrator):
        report_path = orchestrator.run(SAMPLE_CASES)
        with open(report_path) as f:
            data = json.load(f)
        assert "run_id" in data
        assert "cases" in data
        assert len(data["cases"]) == len(SAMPLE_CASES)

    def test_run_case_ids_match_input(self, orchestrator):
        report_path = orchestrator.run(SAMPLE_CASES)
        with open(report_path) as f:
            data = json.load(f)
        reported_ids = {c["case_id"] for c in data["cases"]}
        expected_ids = {tc["id"] for tc in SAMPLE_CASES}
        assert reported_ids == expected_ids

    def test_run_latency_recorded(self, orchestrator):
        report_path = orchestrator.run(SAMPLE_CASES)
        with open(report_path) as f:
            data = json.load(f)
        for case in data["cases"]:
            if not case.get("error"):
                assert case["latency_ms"] is not None

    def test_run_metrics_present(self, orchestrator):
        report_path = orchestrator.run(SAMPLE_CASES)
        with open(report_path) as f:
            data = json.load(f)
        for case in data["cases"]:
            if not case.get("error"):
                metric_names = {m["metric"] for m in case["metrics"]}
                assert "wer" in metric_names
                assert "latency" in metric_names

    def test_run_composite_score_computed(self, orchestrator):
        report_path = orchestrator.run(SAMPLE_CASES)
        with open(report_path) as f:
            data = json.load(f)
        for case in data["cases"]:
            if not case.get("error"):
                assert case["composite_score"] is not None
                assert 0.0 <= case["composite_score"] <= 1.0

    def test_run_deterministic_two_runs_same_result(self, config_with_output):
        """Two runs with the same config and mock client must produce identical scores."""
        client = MockVoiceAIClient(fixed_latency_ms=150.0)

        orch1 = EvaluationOrchestrator(
            config=config_with_output,
            voice_ai_client=client,
            transcriber=MockTranscriber(),
        )
        report_path1 = orch1.run(SAMPLE_CASES)

        orch2 = EvaluationOrchestrator(
            config=config_with_output,
            voice_ai_client=client,
            transcriber=MockTranscriber(),
        )
        report_path2 = orch2.run(SAMPLE_CASES)

        with open(report_path1) as f:
            data1 = json.load(f)
        with open(report_path2) as f:
            data2 = json.load(f)

        # Scores should be identical across runs
        scores1 = [c.get("composite_score") for c in data1["cases"]]
        scores2 = [c.get("composite_score") for c in data2["cases"]]
        assert scores1 == scores2, f"Non-deterministic: {scores1} != {scores2}"

    def test_run_with_error_case_handled_gracefully(self, config_with_output):
        """A case that causes an exception should produce an error report, not crash."""
        # Client that raises on first call
        class FailingClient(MockVoiceAIClient):
            def query(self, text, **kwargs):
                raise RuntimeError("Voice AI unavailable")

        orch = EvaluationOrchestrator(
            config=config_with_output,
            voice_ai_client=FailingClient(),
            transcriber=MockTranscriber(),
        )
        report_path = orch.run([SAMPLE_CASES[0]])
        with open(report_path) as f:
            data = json.load(f)
        assert data["summary"]["error_cases"] == 1
        assert data["cases"][0]["error"] is not None


class TestOrchestratorCompositeScore:

    def test_composite_score_respects_weights(self, orchestrator):
        """Verify composite computation with known metric scores."""
        from utils.report_generator import MetricResult
        metrics = [
            MetricResult(metric="wer", score=1.0),
            MetricResult(metric="latency", score=0.5),
        ]
        score = orchestrator._composite_score(metrics)
        # weights: wer=0.5, latency=0.5 → (1.0*0.5 + 0.5*0.5) = 0.75
        assert score == pytest.approx(0.75, abs=1e-4)

    def test_composite_score_handles_missing_metric(self, orchestrator):
        """If a metric has an error, it's excluded and weights re-normalised."""
        from utils.report_generator import MetricResult
        metrics = [
            MetricResult(metric="wer", score=0.8),
            MetricResult(metric="latency", score=0.0, error="No data"),
        ]
        score = orchestrator._composite_score(metrics)
        # Only WER valid → score = 0.8
        assert score == pytest.approx(0.8, abs=1e-4)
