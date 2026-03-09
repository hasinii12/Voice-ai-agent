"""
Tests for Latency Evaluator and latency_to_score helper.
"""

import pytest
from evaluators.latency_evaluator import LatencyEvaluator, latency_to_score
from utils.determinism import seed_everything


@pytest.fixture(autouse=True)
def seed():
    seed_everything(42)


@pytest.fixture
def evaluator():
    return LatencyEvaluator()


def make_case(latency_ms: float, ttfb_ms: float | None = None) -> dict:
    return {
        "id": "lat_test",
        "input_text": "test",
        "expected_response": "test",
        "reference_facts": [],
        "_latency_ms": latency_ms,
        "_ttfb_ms": ttfb_ms,
    }


class TestLatencyToScore:

    def test_zero_latency_is_perfect(self):
        assert latency_to_score(0) == 1.0

    def test_excellent_threshold_exact(self):
        assert latency_to_score(200) == 1.0

    def test_just_above_excellent(self):
        score = latency_to_score(201)
        assert 0.75 < score < 1.0

    def test_good_threshold_exact(self):
        score = latency_to_score(500)
        assert score == pytest.approx(0.75, abs=1e-5)

    def test_acceptable_threshold_exact(self):
        score = latency_to_score(1000)
        assert score == pytest.approx(0.50, abs=1e-5)

    def test_poor_threshold_exact(self):
        score = latency_to_score(2000)
        assert score == pytest.approx(0.25, abs=1e-5)

    def test_above_poor_is_zero(self):
        assert latency_to_score(2001) == 0.0
        assert latency_to_score(10_000) == 0.0

    def test_monotonically_decreasing(self):
        """Higher latency should always give lower or equal score."""
        latencies = [0, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
        scores = [latency_to_score(ms) for ms in latencies]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score not monotonic at {latencies[i]}ms→{latencies[i+1]}ms: "
                f"{scores[i]:.4f} < {scores[i+1]:.4f}"
            )

    def test_score_always_in_range(self):
        for ms in [0, 1, 50, 199, 200, 201, 499, 500, 999, 1000, 1999, 2000, 2001, 5000]:
            score = latency_to_score(ms)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for {ms}ms"

    def test_custom_thresholds(self):
        custom = {"excellent": 100, "good": 300, "acceptable": 600, "poor": 1200}
        assert latency_to_score(100, custom) == 1.0
        assert latency_to_score(1201, custom) == 0.0


class TestLatencyEvaluator:

    def test_excellent_latency_scores_one(self, evaluator):
        tc = make_case(150.0)
        result = evaluator.evaluate(tc, "any response")
        assert result.score == pytest.approx(1.0)
        assert result.details["band"] == "excellent"

    def test_poor_latency_scores_near_zero(self, evaluator):
        tc = make_case(3000.0)
        result = evaluator.evaluate(tc, "any response")
        assert result.score == pytest.approx(0.0)
        assert result.details["band"] == "unacceptable"

    def test_no_latency_data_returns_error(self, evaluator):
        tc = {
            "id": "no_lat",
            "input_text": "test",
            "expected_response": "test",
            "reference_facts": [],
        }
        result = evaluator.evaluate(tc, "response")
        assert result.score == 0.0
        assert result.error is not None

    def test_raw_value_is_latency_ms(self, evaluator):
        tc = make_case(342.7)
        result = evaluator.evaluate(tc, "response")
        assert result.raw_value == pytest.approx(342.7, abs=0.1)
        assert result.unit == "ms"

    def test_ttfb_in_details_when_provided(self, evaluator):
        tc = make_case(300.0, ttfb_ms=80.0)
        result = evaluator.evaluate(tc, "response")
        assert "ttfb_ms" in result.details
        assert result.details["ttfb_ms"] == pytest.approx(80.0, abs=0.1)

    def test_metric_name(self, evaluator):
        tc = make_case(100.0)
        result = evaluator.evaluate(tc, "response")
        assert result.metric == "latency"

    def test_deterministic_for_same_latency(self, evaluator):
        tc = make_case(487.3)
        scores = [evaluator.evaluate(tc, "r").score for _ in range(5)]
        assert all(s == scores[0] for s in scores)

    def test_band_classification(self, evaluator):
        bands = [
            (100, "excellent"),
            (400, "good"),
            (800, "acceptable"),
            (1500, "poor"),
            (5000, "unacceptable"),
        ]
        for ms, expected_band in bands:
            tc = make_case(float(ms))
            result = evaluator.evaluate(tc, "r")
            assert result.details["band"] == expected_band, (
                f"{ms}ms should be '{expected_band}' but got '{result.details['band']}'"
            )
