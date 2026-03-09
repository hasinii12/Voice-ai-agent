"""
Latency Evaluator.

Measures end-to-end response latency and maps it to a [0, 1] score
using configurable thresholds. Optionally measures Time-To-First-Byte (TTFB).

The latency is recorded by the orchestrator during the actual API call;
this evaluator only converts the raw milliseconds into a normalised score.

Scoring function (piecewise linear):
  ≤ excellent_ms  → 1.0
  ≤ good_ms       → lerp(1.0, 0.75)
  ≤ acceptable_ms → lerp(0.75, 0.50)
  ≤ poor_ms       → lerp(0.50, 0.25)
  > poor_ms       → 0.0

Determinism: Pure function of numeric inputs — fully deterministic.
"""

from typing import Any

from evaluators.base_evaluator import BaseEvaluator, EvalResult
from utils.logger import get_logger

logger = get_logger(__name__)

# Default latency thresholds in milliseconds
DEFAULT_THRESHOLDS = {
    "excellent": 200,
    "good": 500,
    "acceptable": 1000,
    "poor": 2000,
}


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: a + t*(b-a), t in [0,1]."""
    return a + t * (b - a)


def latency_to_score(
    latency_ms: float,
    thresholds: dict[str, int] | None = None,
) -> float:
    """
    Map a latency value (ms) to a score in [0, 1].

    Uses a piecewise-linear decay through the threshold bands.
    """
    t = thresholds or DEFAULT_THRESHOLDS
    excellent = t["excellent"]
    good = t["good"]
    acceptable = t["acceptable"]
    poor = t["poor"]

    if latency_ms <= excellent:
        return 1.0
    elif latency_ms <= good:
        # interpolate from 1.0 → 0.75
        frac = (latency_ms - excellent) / (good - excellent)
        return round(_lerp(1.0, 0.75, frac), 6)
    elif latency_ms <= acceptable:
        frac = (latency_ms - good) / (acceptable - good)
        return round(_lerp(0.75, 0.50, frac), 6)
    elif latency_ms <= poor:
        frac = (latency_ms - acceptable) / (poor - acceptable)
        return round(_lerp(0.50, 0.25, frac), 6)
    else:
        return 0.0


class LatencyEvaluator(BaseEvaluator):
    """
    Converts raw latency measurements into a normalised [0, 1] score.

    Expects the test_case to contain '_latency_ms' (injected by orchestrator).
    Optionally uses '_ttfb_ms' for time-to-first-byte if available.

    Config keys
    -----------
    thresholds.excellent  : int ms (default 200)
    thresholds.good       : int ms (default 500)
    thresholds.acceptable : int ms (default 1000)
    thresholds.poor       : int ms (default 2000)
    measure_ttfb          : bool (default True)
    """

    name = "latency"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._thresholds = {
            **DEFAULT_THRESHOLDS,
            **self.config.get("thresholds", {}),
        }
        self._measure_ttfb = self.config.get("measure_ttfb", True)

    def evaluate(
        self,
        test_case: dict[str, Any],
        actual_response: str,  # noqa: ARG002  (not used for latency)
    ) -> EvalResult:
        """
        Score the latency for this test case.

        The orchestrator must inject '_latency_ms' into the test_case dict
        before calling this evaluator.
        """
        latency_ms = test_case.get("_latency_ms")
        ttfb_ms = test_case.get("_ttfb_ms")

        if latency_ms is None:
            logger.warning(
                f"[{test_case.get('id')}] No latency data — returning score=0"
            )
            return EvalResult(
                metric=self.name,
                score=0.0,
                error="No latency data (_latency_ms not set by orchestrator)",
            )

        score = latency_to_score(latency_ms, self._thresholds)

        details: dict[str, Any] = {
            "latency_ms": round(latency_ms, 2),
            "thresholds": self._thresholds,
            "band": self._get_band(latency_ms),
        }

        if ttfb_ms is not None and self._measure_ttfb:
            ttfb_score = latency_to_score(ttfb_ms, self._thresholds)
            details["ttfb_ms"] = round(ttfb_ms, 2)
            details["ttfb_score"] = round(ttfb_score, 6)

        logger.debug(
            f"[{test_case.get('id')}] latency={latency_ms:.1f}ms "
            f"score={score:.4f} band={details['band']}"
        )

        return EvalResult(
            metric=self.name,
            score=score,
            raw_value=round(latency_ms, 2),
            unit="ms",
            details=details,
        )

    def _get_band(self, latency_ms: float) -> str:
        t = self._thresholds
        if latency_ms <= t["excellent"]:
            return "excellent"
        elif latency_ms <= t["good"]:
            return "good"
        elif latency_ms <= t["acceptable"]:
            return "acceptable"
        elif latency_ms <= t["poor"]:
            return "poor"
        return "unacceptable"
